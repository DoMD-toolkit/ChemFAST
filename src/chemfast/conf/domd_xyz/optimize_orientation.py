import time
from collections import namedtuple

import numba as nb
import numpy as np
import torch
from scipy.optimize import minimize
from torch import optim

from chemfast.misc.logger import logger


def optimization_by_chunk(chunk_per_d, connections, pos, local_frame_idx, trans, box, rot, expand_radius=1):
    """Build spatial chunks augmented by a bonded graph neighborhood."""
    rot = rot.reshape(-1, 3, 3)
    shifted_trans = trans + 0.5 * box
    chunk_len = np.ceil(box / chunk_per_d)
    cgpos_cell_idx = shifted_trans // chunk_len
    cell_idx_set = sorted({tuple(idx) for idx in cgpos_cell_idx})
    idx_to_cid = {idx: cid for cid, idx in enumerate(cell_idx_set)}
    cgpos_cell_cid = np.array([idx_to_cid[tuple(idx)] for idx in cgpos_cell_idx], dtype=np.int64)
    cid_set = np.unique(cgpos_cell_cid)
    connection_u, connection_v = connections.T
    cid_hash, cid_meta = {}, {}
    for cid in cid_set:
        chunk_res_idx = np.flatnonzero(cgpos_cell_cid == cid)
        expanded_mask = np.zeros(len(trans), dtype=bool)
        expanded_mask[chunk_res_idx] = True
        for _ in range(expand_radius):
            expansion_edge_mask = expanded_mask[connection_u] | expanded_mask[connection_v]
            expanded_mask[connection_u[expansion_edge_mask]] = True
            expanded_mask[connection_v[expansion_edge_mask]] = True
        edge_mask = expanded_mask[connection_u] & expanded_mask[connection_v]
        expanded_res_idx = np.flatnonzero(expanded_mask)
        expanded_connections = connections[edge_mask]
        local_to_global = {i: gid for i, gid in enumerate(expanded_res_idx)}
        global_to_local = {gid: i for i, gid in enumerate(expanded_res_idx)}
        if len(expanded_connections) > 0:
            local_connections = np.array(
                [[global_to_local[i], global_to_local[j]] for i, j in expanded_connections], dtype=np.int64
            )
        else:
            local_connections = np.empty((0, 2), dtype=np.int64)
        chunk_local_idx = np.array([global_to_local[gid] for gid in chunk_res_idx], dtype=np.int64)
        cid_hash[cid] = edge_mask
        cid_meta[cid] = {
            'connections_': local_connections, 'local_frame_idx': local_frame_idx[edge_mask],
            'n_residue': len(expanded_res_idx), 'rot': rot[expanded_res_idx].ravel(),
            'local_to_global': local_to_global, 'connections': expanded_connections,
            'chunk_res_idx': chunk_res_idx, 'chunk_local_idx': chunk_local_idx,
            'expanded_res_idx': expanded_res_idx
        }
    return cid_set, cid_meta, cid_hash


@nb.jit(nopython=True, nogil=True)
def pbc(r, d):
    """Calculates periodic boundary condition differences using Numba.

        Args:
            r (np.ndarray): Coordinate difference vector.
            d (np.ndarray): Box dimensions.

        Returns:
            np.ndarray: Wrapped vector.
    """
    return r - d * np.rint(r / d)


def pbc_torch(r, d):
    """Calculates periodic boundary condition differences using PyTorch.

        Args:
            r (Tensor): Coordinate difference tensor.
            d (Tensor): Box dimensions tensor.

        Returns:
            Tensor: Wrapped tensor.
    """
    return r - d * torch.floor(r / d + 0.5)


def constraint_det(x):
    r"""Calculates the determinant constraint error for rotation matrices.

    Ensures that the determinant of each rotation matrix is 1 (proper rotation).

    .. math::
        \mathcal{L}_{det} = \sum_{i} (\det(R_i) - 1)^2

    Args:
        x (np.ndarray): Flattened array of rotation matrices (N*9,).

    Returns:
        float: Sum of squared errors of determinants.
    """
    N = x.size // 9
    R = x.reshape(N, 3, 3)

    r0, r1, r2 = R[:, 0], R[:, 1], R[:, 2]

    cross_12 = np.cross(r1, r2, axis=1)
    dets = np.einsum('ij,ij->i', r0, cross_12)

    return np.sum((dets - 1.0) ** 2)


def constraint_det_jac_(R):  # Only the second part
    R = R.reshape(-1, 3, 3)
    det = np.linalg.det(R)
    inv = np.linalg.pinv(R)
    return 2 * det[:, None, None] * (det[:, None, None] - 1) * np.swapaxes(inv, (1, 2))


def constraint_det_jac(x):
    r"""Calculates the analytical Jacobian of the determinant constraint.

    Computes gradients via the cross product rule for determinants.

    .. math::

       \frac{\partial \det(R)}{\partial r_0} = r_1 \times r_2

    Args:
        x (np.ndarray): Flattened array of rotation matrices (N*9,).

    Returns:
        np.ndarray: Flattened gradient vector (N*9,).
    """
    N = x.size // 9
    R = x.reshape(N, 3, 3)
    r0, r1, r2 = R[:, 0], R[:, 1], R[:, 2]

    grad_r0 = np.cross(r1, r2, axis=1)
    grad_r1 = np.cross(r2, r0, axis=1)
    grad_r2 = np.cross(r0, r1, axis=1)

    dets = np.einsum('ij,ij->i', r0, grad_r0)

    diff = dets - 1.0
    factor = 2 * diff[:, np.newaxis]  # (N, 1) 用于广播
    final_grad_r0 = factor * grad_r0
    final_grad_r1 = factor * grad_r1
    final_grad_r2 = factor * grad_r2

    return np.stack([final_grad_r0, final_grad_r1, final_grad_r2], axis=1).reshape(-1)


cons_det = {
    'type': 'eq',
    'fun': constraint_det,
    'jac': constraint_det_jac
}


def rot_cons(rot0):
    """Calculates the orthogonality constraint error.

        Ensures $R^T R = I$.

        Args:
            rot0 (np.ndarray): Flattened rotation matrices.

        Returns:
            float: Sum of squared errors of orthogonality.
    """
    rot = rot0.reshape(-1, 3, 3)
    a = np.sum((np.einsum('ikj,ikl->ijl', rot, rot) - np.eye(3)) ** 2)
    return a


def rot_cons_jac(rot0):
    """Calculates the Jacobian of the orthogonality constraint.

        Args:
            rot0 (np.ndarray): Flattened rotation matrices.

        Returns:
            np.ndarray: Flattened gradient vector.
    """
    rot = rot0.reshape(-1, 3, 3)
    b = np.einsum('ikj,ikl->ijl', rot, rot) - np.eye(3)
    a = 4 * np.einsum('ijk, ikl->ijl', rot, b).ravel()
    return np.nan_to_num(a, nan=0)  # 4 * np.einsum('ijk, ikl->ijl', rot, b).ravel()


cons = ({'type': 'eq', 'fun': rot_cons, 'jac': rot_cons_jac, }, cons_det)
# cons = ({'type': 'ineq', 'fun': rot_cons, },)

Meta = namedtuple("Meta", "bonds trans_v local_x atom_pos atom_res_id box")


def optimize_res_orientation(n_residue, meta, chunk_per_d=1, expand_radius=1):
    r"""Optimizes the orientation of residues to reconnect bonded atoms.

        This function finds the optimal rotation matrices $R_i$ for each residue $i$ such
        that the distance between bonded atoms $a \in i$ and $b \in j$ matches the
        expected bond length (implicitly minimized to zero offset in this context, assuming
        ideal geometry inputs).



        **Objective Function:**
        Minimize the sum of squared distances between bonded atoms across residue boundaries:

        .. math::
            \min_{\{R_k\}} \sum_{(i,j) \in \text{bonds}} || (R_{res(i)} \cdot x_{local, i} + T_{res(i)}) - (R_{res(j)} \cdot x_{local, j} + T_{res(j)}) ||^2

        **Constraints:**
        1. Orthogonality: $R_k^T R_k = I$
        2. Proper Rotation: $\det(R_k) = 1$

        Uses `scipy.optimize.minimize` with 'trust-constr' method. Supports splitting
        the problem into chunks for performance.

        Args:
            n_residue (int): Total number of residues.
            meta (Meta): NamedTuple containing topology and coordinate info.
            chunk_per_d (int, optional): Number of chunks per dimension.
                If > 1, optimization is performed locally on chunks. Defaults to 1.
            expand_radius (int, optional): Bonded-graph expansion radius around each
                spatial chunk. Defaults to 1.

        Returns:
            np.ndarray: The optimized rotation matrices of shape (n_residue, 3, 3).
    """
    total_start = time.perf_counter()
    total_n_residue = n_residue
    identity_rot = np.array([np.eye(3), ] * total_n_residue)
    r0 = (np.random.normal(0, 0.01, (
            n_residue * 9)))  # np.tile(np.eye(3), (n_residue, 1, 1)).ravel()# + (np.random.normal(0,0.01,(n_residue * 9)) )
    # r0 = np.concatenate((r0,r0),axis=0)
    connections = meta.bonds
    # print(connections,meta.local_x,meta.atom_pos)
    trans = meta.trans_v
    trans_torch = torch.tensor(trans)
    # trans = trans.astype(np.float16)
    local_frame_idx = meta.local_x
    box = meta.box  # * n_residue
    # box = box.astype(np.float16)
    pos = meta.atom_pos
    pos_torch = torch.tensor(meta.atom_pos)
    box_torch = torch.tensor(meta.box)
    # self.post = pos_torch
    # self.boxt = box_torch
    ##self.trant = trans_torch
    if n_residue == 1:
        logger.info("Orientation optimization skipped: only one residue.")
        return identity_rot
    if len(connections) == 0:
        logger.warning("Orientation optimization skipped: no inter-residue bonds.")
        return identity_rot
    if chunk_per_d > 1 and (not isinstance(expand_radius, (int, np.integer)) or expand_radius < 1):
        raise ValueError("expand_radius must be a positive integer.")
    # numba free
    # @nb.jit(nopython=True, nogil=True)
    # device = torch.device(0 if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:0')

    def _loss_jac_torch(rot0):
        start = time.time()
        rot = torch.tensor(rot0, requires_grad=True, device=device)
        ri = pbc_torch(
            (torch.einsum('ijk,ik->ij', rot[connections.T[0]], pos_torch[local_frame_idx.T[0]].to(device)) +
             trans_torch[
                 connections.T[0]].to(device)),
            box_torch.to(device)
        )
        rj = pbc_torch(
            (torch.einsum('ijk,ik->ij', rot[connections.T[1]], pos_torch[local_frame_idx.T[1]].to(device)) +
             trans_torch[
                 connections.T[1]].to(device)),
            box_torch.to(device)
        )
        rij = pbc_torch(rj - ri, box_torch.to(device))
        s = torch.sum(rij ** 2)
        s.backward()
        # print(rot)
        # print('JacobiTime',time.time()-start)
        return (rot.grad.detach().cpu().numpy().ravel())

    @nb.njit
    def matrix_multiply(matrices, vectors):
        N = matrices.shape[0]
        results = np.empty((N, 3))  # Initialize an array to hold the results
        for i in range(N):
            results[i] = matrices[i] @ vectors[i]  # Matrix-vector multiplication
        return results

    @nb.jit(nopython=True, nogil=True)
    def _loss_jac(rot):
        ri = (matrix_multiply(rot[connections.T[0]], pos[local_frame_idx.T[0]])).reshape(-1, 3) + trans[
            connections.T[0]]
        rj = (matrix_multiply(rot[connections.T[1]], pos[local_frame_idx.T[1]])).reshape(-1, 3) + trans[
            connections.T[1]]
        rij = pbc(rj - ri, box)
        r0j = pos[local_frame_idx.T[1]]
        grad = np.zeros_like(rot)
        g1_grad = np.zeros_like(rot)
        for i in range(len(rot)):
            g1 = np.sum(
                2 * (rij.reshape(-1, 1, 3) * pos[local_frame_idx.T[1]].reshape(-1, 3, 1))[connections.T[1] == i],
                axis=0)
            g2 = np.sum(
                2 * (rij.reshape(-1, 1, 3) * pos[local_frame_idx.T[0]].reshape(-1, 3, 1))[connections.T[0] == i],
                axis=0)
            # print(g1.shape)
            if g1.shape[0] == 0:
                g1 = np.zeros((3, 3))
            if g2.shape[0] == 0:
                g2 = np.zeros((3, 3))
            # print(g1.shape)
            # rint(g2.shape)
            grad[i] = ((g1 - g2).T)
            g1_grad[i] = g1
            # print(i)
        return grad.ravel()

    def _loss_jac_ana(rot):
        # print('loss_jac')
        rot = rot.reshape(-1, 3, 3)
        ri = (np.einsum('ijk, ipk->ipj', rot[connections_.T[0]], pos[local_frame_idx.T[0]].reshape(-1, 1, 3)) + trans[
            connections.T[0]].reshape(-1, 1, 3))
        rj = (np.einsum('ijk, ipk->ipj', rot[connections_.T[1]], pos[local_frame_idx.T[1]].reshape(-1, 1, 3)) + trans[
            connections.T[1]].reshape(-1, 1, 3))
        rij = pbc(rj - ri, box)
        g1 = np.zeros_like(rot).reshape(-1, 1, 3, 3)
        g2 = np.zeros_like(rot).reshape(-1, 1, 3, 3)
        np.add.at(g1, connections_.T[1],
                  2 * np.einsum('ijk,ijl->ijkl', rij, pos[local_frame_idx.T[1]].reshape(-1, 1, 3)))
        np.add.at(g2, connections_.T[0],
                  2 * np.einsum('ijk,ijl->ijkl', rij, pos[local_frame_idx.T[0]].reshape(-1, 1, 3)))
        a = g1 - g2
        return a.ravel()

    def _loss(rot):
        rot = rot.reshape(n_residue, 3, 3)
        ri = (np.einsum('ijk, ipk->ipj', rot[connections_.T[0]], pos[local_frame_idx.T[0]].reshape(-1, 1, 3)) + trans[
            connections.T[0]].reshape(-1, 1, 3))
        rj = (np.einsum('ijk, ipk->ipj', rot[connections_.T[1]], pos[local_frame_idx.T[1]].reshape(-1, 1, 3)) + trans[
            connections.T[1]].reshape(-1, 1, 3))
        rij = pbc(rj - ri, box)
        s = np.sum(rij ** 2)
        jac = _loss_jac_ana(rot)
        return s, jac

    def lfn(rot, pos, trans, box):
        ri = torch.einsum('ijk,ik->ij', rot[connections.T[0]], pos[local_frame_idx.T[0]]) + trans[connections.T[0]]
        rj = torch.einsum('ijk,ik->ij', rot[connections.T[1]], pos[local_frame_idx.T[1]]) + trans[connections.T[1]]
        rij = pbc_torch(rj - ri, box)
        s = torch.sum(rij ** 2)
        return s

    def Rot_cons(rot, eye):
        return torch.sum((torch.det(rot) - 1) ** 2)

    def torch_opt(rot, lam, rho, pos_torch, trans_torch, box_torch, maxiter=500):
        device = torch.device(1)
        rot = torch.tensor(rot.reshape(n_residue, 3, 3), requires_grad=True, device=device)
        lam = torch.tensor(lam, device=device)
        rho = torch.tensor(rho, device=device)
        done = False
        pos_torch = pos_torch.to(device)
        trans_torch = trans_torch.to(device)
        box_torch = box_torch.to(device)
        eye = torch.eye(3).to(device)
        inner_loop = 100
        outer_loop = maxiter
        for i in range(outer_loop):
            optimizerR = optim.Adam([rot], lr=1e-6)
            rot_ = rot.clone().detach()
            for j in range(inner_loop):
                optimizerR.zero_grad()
                if done:
                    break
                f = lfn(rot, pos_torch, trans_torch, box_torch)
                p = lam * Rot_cons(rot, eye) + 0.5 * rho * Rot_cons(rot, eye) ** 2
                loss = f + p
                loss.backward()
                optimizerR.step()
            constraint_val = Rot_cons(rot, eye).item()
            with torch.no_grad():
                lam += rho * constraint_val
            if lam > 1e9:
                rho *= 1.0001
            else:
                rho *= 1.5
            if torch.mean(torch.abs(rot.grad)) < 1e-2 and torch.mean(
                    torch.abs(rot_ - rot)) < 1e-4 and constraint_val < 5e-7:
                done = True
            if done:
                break
        return rot.detach().cpu().numpy().ravel(), lam, rho, done

    def finite_result(result):
        return np.all(np.isfinite(result.x)) and np.isfinite(result.fun)

    max_retries = 5
    if chunk_per_d <= 1:
        logger.info(f"Orientation optimization: residues={total_n_residue}, bonds={len(connections)}.")
        connections_ = connections
        current_x = r0
        current_maxiter = 500
        last_finite_result = None
        result = None
        for retry_number in range(max_retries + 1):
            result = minimize(_loss, current_x, constraints=cons,
                              options={'maxiter': current_maxiter, 'disp': False}, jac=True,
                              method='trust-constr')
            if finite_result(result):
                last_finite_result = result
            if result.success and finite_result(result):
                break
            if retry_number < max_retries:
                next_maxiter = current_maxiter * 2
                logger.warning(
                    f"Orientation optimization retry {retry_number + 1}/{max_retries}: not converged at maxiter={current_maxiter}; increasing maxiter to {next_maxiter}.")
                if np.all(np.isfinite(result.x)):
                    current_x = result.x
                current_maxiter = next_maxiter
        if not (result.success and finite_result(result)):
            if last_finite_result is None:
                logger.error("Orientation optimization failed: no finite solution; identity orientations are returned.")
                return identity_rot
            result = last_finite_result
            logger.warning(
                f"Orientation optimization did not converge after {max_retries} retries; the last finite result is retained.")
        rot_final = result.x.reshape(total_n_residue, 3, 3)
        n_optimized = total_n_residue
    else:
        cid_set, cid_meta, _ = optimization_by_chunk(
            chunk_per_d, connections, pos, local_frame_idx, trans, box, r0, expand_radius=expand_radius
        )
        n_chunks = len(cid_set)
        max_chunk_nodes = max(len(cid_meta[cid]['chunk_res_idx']) for cid in cid_set)
        max_expanded_nodes = max(len(cid_meta[cid]['expanded_res_idx']) for cid in cid_set)
        max_chunk_bonds = max(len(cid_meta[cid]['connections']) for cid in cid_set)
        logger.info(
            f"Orientation optimization by chunks: residues={total_n_residue}, bonds={len(connections)}, chunks={n_chunks}.")
        logger.info(
            f"Chunk decomposition: max nodes/chunk={max_chunk_nodes}, max expanded nodes/chunk={max_expanded_nodes}, max bonds/chunk={max_chunk_bonds}.")
        rot_final = identity_rot.copy()
        n_optimized = 0
        for chunk_number, cid in enumerate(cid_set, start=1):
            chunk_meta = cid_meta[cid]
            n_chunk_nodes = len(chunk_meta['chunk_res_idx'])
            n_expanded_nodes = len(chunk_meta['expanded_res_idx'])
            n_chunk_bonds = len(chunk_meta['connections'])
            logger.info(
                f"Chunk {chunk_number}/{n_chunks} started: nodes={n_chunk_nodes}, expanded nodes={n_expanded_nodes}, bonds={n_chunk_bonds}.")
            if n_chunk_bonds == 0:
                continue
            current_x = chunk_meta['rot']
            connections = chunk_meta['connections']
            connections_ = chunk_meta['connections_']
            local_frame_idx = chunk_meta['local_frame_idx']
            local_to_global = chunk_meta['local_to_global']
            n_residue = chunk_meta['n_residue']
            current_maxiter = 100
            last_finite_result = None
            result = None
            for retry_number in range(max_retries + 1):
                result = minimize(_loss, current_x, constraints=cons,
                                  options={'maxiter': current_maxiter, 'disp': False}, jac=True,
                                  method='trust-constr')
                if finite_result(result):
                    last_finite_result = result
                if result.success and finite_result(result):
                    break
                if retry_number < max_retries:
                    next_maxiter = current_maxiter * 2
                    logger.warning(
                        f"Chunk {chunk_number}/{n_chunks} retry {retry_number + 1}/{max_retries}: not converged at maxiter={current_maxiter}; increasing maxiter to {next_maxiter}.")
                    if np.all(np.isfinite(result.x)):
                        current_x = result.x
                    current_maxiter = next_maxiter
            if not (result.success and finite_result(result)):
                if last_finite_result is None:
                    logger.error(
                        f"Chunk {chunk_number}/{n_chunks} failed: no finite solution; node orientations are unchanged.")
                    continue
                result = last_finite_result
                logger.warning(
                    f"Chunk {chunk_number}/{n_chunks} did not converge after {max_retries} retries; the last finite result is retained.")
            local_rot = result.x.reshape(n_residue, 3, 3)
            for local_id in chunk_meta['chunk_local_idx']:
                rot_final[local_to_global[local_id]] = local_rot[local_id]
            n_optimized += n_chunk_nodes

    n_unchanged = total_n_residue - n_optimized
    gram = np.einsum('nij,nkj->nik', rot_final, rot_final)
    orthogonality_error = np.max(np.linalg.norm(gram - np.eye(3), axis=(1, 2)))
    determinant_error = np.max(np.abs(np.linalg.det(rot_final) - 1.0))
    constraint_tolerance = 1.0e-3
    total_elapsed = time.perf_counter() - total_start
    if orthogonality_error <= constraint_tolerance and determinant_error <= constraint_tolerance:
        logger.info(
            f"Orientation optimization completed: optimized={n_optimized}, unchanged={n_unchanged}, time={total_elapsed:.1f} s, orthogonality={orthogonality_error:.3e}, determinant={determinant_error:.3e}.")
    else:
        logger.warning(
            f"Orientation optimization failed constraint check: optimized={n_optimized}, unchanged={n_unchanged}, time={total_elapsed:.1f} s, orthogonality={orthogonality_error:.3e}, determinant={determinant_error:.3e}, tolerance={constraint_tolerance:.1e}.")
    return rot_final
