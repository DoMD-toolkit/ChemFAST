import numpy as np
from numba import njit
from scipy.spatial.transform import Rotation as R


@njit
def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ])


@njit
def quat_rot(q, v):
    q_vec = q[1:4]
    q_w = q[0]
    tx = 2.0 * (q_vec[1] * v[2] - q_vec[2] * v[1])
    ty = 2.0 * (q_vec[2] * v[0] - q_vec[0] * v[2])
    tz = 2.0 * (q_vec[0] * v[1] - q_vec[1] * v[0])
    t = np.array([tx, ty, tz])
    cx = q_vec[1] * t[2] - q_vec[2] * t[1]
    cy = q_vec[2] * t[0] - q_vec[0] * t[2]
    cz = q_vec[0] * t[1] - q_vec[1] * t[0]
    return np.array([
        v[0] + q_w * tx + cx,
        v[1] + q_w * ty + cy,
        v[2] + q_w * tz + cz
    ])


@njit
def md_step(local_coords, com_coords, mol_idx, bonds, is_fixed, q, omega, box, dt, damping, k):
    M_bonds = bonds.shape[0]
    K_mols = q.shape[0]

    torques = np.zeros((K_mols, 3), dtype=np.float64)
    energy = 0.0

    for b in range(M_bonds):
        idx_A = bonds[b, 0]
        idx_B = bonds[b, 1]

        m_A = mol_idx[idx_A]
        m_B = mol_idx[idx_B]

        r_A = quat_rot(q[m_A], local_coords[idx_A])
        r_B = quat_rot(q[m_B], local_coords[idx_B])

        posA_x = r_A[0] + com_coords[m_A, 0]
        posA_y = r_A[1] + com_coords[m_A, 1]
        posA_z = r_A[2] + com_coords[m_A, 2]

        posB_x = r_B[0] + com_coords[m_B, 0]
        posB_y = r_B[1] + com_coords[m_B, 1]
        posB_z = r_B[2] + com_coords[m_B, 2]

        dx = posA_x - posB_x
        dy = posA_y - posB_y
        dz = posA_z - posB_z

        dx -= box[0] * np.round(dx / box[0])
        dy -= box[1] * np.round(dy / box[1])
        dz -= box[2] * np.round(dz / box[2])

        energy += 0.5 * k * (dx * dx + dy * dy + dz * dz)
        fx = -k * dx
        fy = -k * dy
        fz = -k * dz

        if not is_fixed[m_A]:
            torques[m_A, 0] += r_A[1] * fz - r_A[2] * fy
            torques[m_A, 1] += r_A[2] * fx - r_A[0] * fz
            torques[m_A, 2] += r_A[0] * fy - r_A[1] * fx

        if not is_fixed[m_B]:
            torques[m_B, 0] -= r_B[1] * fz - r_B[2] * fy
            torques[m_B, 1] -= r_B[2] * fx - r_B[0] * fz
            torques[m_B, 2] -= r_B[0] * fy - r_B[1] * fx

    for m in range(K_mols):
        if is_fixed[m]: continue

        omega[m, 0] = omega[m, 0] * (1.0 - damping) + dt * torques[m, 0]
        omega[m, 1] = omega[m, 1] * (1.0 - damping) + dt * torques[m, 1]
        omega[m, 2] = omega[m, 2] * (1.0 - damping) + dt * torques[m, 2]

        w_quat = np.array([0.0, omega[m, 0], omega[m, 1], omega[m, 2]])
        q_dot = 0.5 * quat_mult(w_quat, q[m])

        q[m, 0] += dt * q_dot[0]
        q[m, 1] += dt * q_dot[1]
        q[m, 2] += dt * q_dot[2]
        q[m, 3] += dt * q_dot[3]

        norm = np.sqrt(q[m, 0] ** 2 + q[m, 1] ** 2 + q[m, 2] ** 2 + q[m, 3] ** 2)
        q[m] /= norm

    return energy


def optimize_orient(local_coords: np.ndarray, com: np.ndarray, box: np.ndarray,
                    bonds: np.ndarray, mol_idx: np.ndarray, is_fixed: np.ndarray, steps: int = 3000):
    r"""
    Args:
        local_coords: (N, 3) array, total AA corrdinates
        com: (K, 3) array, total K residues
        box: (3, ) box lengths
        bonds: (M, 2), M bonds to be optimized
        mol_idx: (N, ) molecular index of atom i
        is_fixed: (K, ) rotate or not, for residue k

    Returns: ret = (K, 3, 3), K rotation matrices, ret[k] = I, if is_fixed[k] == 1

    """
    # initialize q as I
    q = np.zeros((is_fixed.shape[0], 4), dtype=np.float64)  # K residues
    q[:, 0] = 1.0

    omega = np.zeros((is_fixed.shape[0], 3), dtype=np.float64)
    dt = 0.01
    damping = 0.15  # damping factor 0.15-0.3
    k = 15  # spring factor of bonds, too large k/damping will cause instability, 10-30
    # estimated based on dt=0.01, spring range from 2A to 20A
    md_step(local_coords, com, mol_idx, bonds, is_fixed, q, omega, box, dt, damping, k)
    # s = time.time()
    for _step in range(steps):
        energy = md_step(local_coords, com, mol_idx, bonds, is_fixed, q, omega, box, dt, damping, k)
        # if _step % 500 == 0:
        #     print(f"Step: {_step:3d} | Energy: {energy:.4f}")
    # e = time.time()
    # print(f"{e-s:.4f}s TPS {steps/(e-s):.4f}/s")

    return R.from_quat(q, scalar_first=True).as_matrix()


if __name__ == "__main__":
    local_coords = np.random.random((1_000_000, 3)) * 10
    com = local_coords.reshape(-1, 50, 3).mean(axis=1)
    box = np.ones(3) * 10
    mol_idx = np.zeros(local_coords.shape[0], dtype=np.int32)
    for i in range(com.shape[0]):
        mol_idx[i * 50: (i + 1) * 50] = i
    is_fixed = np.zeros(com.shape[0], dtype=np.bool)
    is_fixed[1] = True
    bonds = np.asarray([(i * 50 + 49, (i + 1) * 50) for i in range(com.shape[0] - 1)])

    ret = optimize_orient(local_coords, com, box, bonds, mol_idx, is_fixed)
    print(ret[0], ret[1])
