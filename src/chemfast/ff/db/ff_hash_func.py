from __future__ import annotations

import hashlib
import struct

import numpy as np
from numba import njit, prange
from rdkit import Chem

# ============================================================
# 固定配置
# ============================================================

_DIGEST_SIZE = 16

# hashlib.blake2b 的 person 最长 16 bytes
_PERSON_ATOM_INIT = b"DOMD-AINIT-v1"
_PERSON_ATOM_WL   = b"DOMD-AWL-v1"
_PERSON_BOND      = b"DOMD-BOND-v1"
_PERSON_ANGLE     = b"DOMD-ANGLE-v1"
_PERSON_DIHEDRAL  = b"DOMD-DIH-v1"
_PERSON_IMPROPER  = b"DOMD-IMP-v1"

# 13 个 int32 原子属性
_ATOM_FEATURE_STRUCT = struct.Struct(">7i")
_UINT32_STRUCT = struct.Struct(">I")


# ============================================================
# 基础函数
# ============================================================


def _blake16(person: bytes, payload: bytes) -> bytes:
    return hashlib.blake2b(
        payload,
        digest_size=_DIGEST_SIZE,
        person=person,
    ).digest()


def _cip_code(atom: Chem.Atom) -> int:
    """
    使用已经存在的 CIP 信息，不主动调用立体化学感知。
    """
    if not atom.HasProp("_CIPCode"):
        return 0

    value = atom.GetProp("_CIPCode")

    return {
        "R": 1,
        "S": 2,
        "r": 3,
        "s": 4,
    }.get(value, 0)


def _bond_code(bond: Chem.Bond) -> int:
    bond_type = int(bond.GetBondType()) & 0xFF

    order_x2 = int(round(bond.GetBondTypeAsDouble() * 2.0)) & 0xFF

    aromatic = int(bond.GetIsAromatic()) & 0x01
    conjugated = int(bond.GetIsConjugated()) & 0x01

    return bond_type | (order_x2 << 8) | (aromatic << 16) | (conjugated << 17)


# ============================================================
# 紧凑的 atom hash 容器
# ============================================================


class AtomHashes:
    """
    每个原子只存 16 bytes。

    同时保留紧凑 CSR 邻接数据，供 bond/angle/dihedral/improper
    现场取得相关键类型，不需要建立 Python edge dictionary。
    """

    __slots__ = (
        "_hashes",
        "_indptr",
        "_neighbors",
        "_bond_codes",
        "num_atoms",
    )

    def __init__(
        self,
        hashes    : np.ndarray,
        indptr    : np.ndarray,
        neighbors : np.ndarray,
        bond_codes: np.ndarray,
    ):
        self._hashes = np.ascontiguousarray(hashes, dtype=np.uint8)
        self._indptr = np.ascontiguousarray(indptr, dtype=np.int64)
        self._neighbors = np.ascontiguousarray(neighbors, dtype=np.int32)
        self._bond_codes = np.ascontiguousarray(
            bond_codes,
            dtype=np.uint32,
        )
        self.num_atoms = self._hashes.shape[0]

    def __len__(self) -> int:
        return self.num_atoms

    def __getitem__(self, idx: int) -> bytes:
        """
        返回真正的 Python bytes，可直接作为 dict/SQLite 主键。
        """
        return self._hashes[int(idx)].tobytes()

    def bond_code(self, i: int, j: int) -> int:
        """
        从 CSR 中查找 i-j 的键属性。

        化学图的原子度通常很小，因此这里直接扫描邻居，
        避免构造巨大的 Python dict。
        """
        i = int(i)
        j = int(j)

        start = int(self._indptr[i])
        end = int(self._indptr[i + 1])

        for pos in range(start, end):
            if int(self._neighbors[pos]) == j:
                return int(self._bond_codes[pos])

        raise ValueError(f"atoms {i} and {j} are not bonded")


# ============================================================
# 从 RDKit Mol 提取原子初始标签
# ============================================================


def _initial_atom_hashes(mol: Chem.Mol) -> np.ndarray:
    """
    初始标签只包含稳定的化学/图属性。

    不包含：
    - implicit H 数量
    - explicit H 计数
    - total H
    - total degree
    - CIP property

    显式氢已经是图中的节点，radius >= 1 时会通过 WL
    邻居传播自然进入原子环境，不应该再用 RDKit 的 H bookkeeping。
    """
    num_atoms = mol.GetNumAtoms()

    labels = np.empty(
        (num_atoms, _DIGEST_SIZE),
        dtype=np.uint8,
    )

    cache = {}

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()

        payload = _ATOM_FEATURE_STRUCT.pack(
            atom.GetAtomicNum(),
            atom.GetIsotope(),
            atom.GetFormalCharge(),
            atom.GetDegree(),
            atom.GetNumRadicalElectrons(),
            int(atom.GetHybridization()),
            int(atom.GetIsAromatic()),
        )

        digest_row = cache.get(payload)

        if digest_row is None:
            digest_row = np.frombuffer(
                _blake16(_PERSON_ATOM_INIT, payload),
                dtype=np.uint8,
            ).copy()

            cache[payload] = digest_row

        labels[idx] = digest_row

    return labels


# ============================================================
# 构建紧凑 CSR
# ============================================================


def _build_csr(
    mol: Chem.Mol,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    通过每个原子的局部 bond iterator 构建 CSR。

    不使用 mol.GetBonds()，避免超大分子中全局 BondSeq
    可能出现的近似 O(E^2) 遍历。

    每条键只处理一次，并同时写入两个方向。
    """
    num_atoms = mol.GetNumAtoms()

    # --------------------------------------------------------
    # 第一遍：获取每个原子的局部度数
    # --------------------------------------------------------

    degree = np.empty(num_atoms, dtype=np.int64)

    for atom in mol.GetAtoms():
        degree[atom.GetIdx()] = atom.GetDegree()

    # --------------------------------------------------------
    # CSR offset
    # --------------------------------------------------------

    indptr = np.empty(num_atoms + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(degree, out=indptr[1:])

    num_directed_edges = int(indptr[-1])

    neighbors = np.empty(
        num_directed_edges,
        dtype=np.int32,
    )

    bond_codes = np.empty(
        num_directed_edges,
        dtype=np.uint32,
    )

    cursor = indptr[:-1].copy()

    # --------------------------------------------------------
    # 第二遍：走每个原子的局部邻接键
    #
    # atom.GetBonds() 最多只遍历该原子的几个局部键。
    # j > i 保证每条键只处理一次。
    # --------------------------------------------------------

    for atom in mol.GetAtoms():
        i = atom.GetIdx()

        for bond in atom.GetBonds():
            j = bond.GetOtherAtomIdx(i)

            if j <= i:
                continue

            code = _bond_code(bond)

            # i -> j
            pos = cursor[i]
            neighbors[pos] = j
            bond_codes[pos] = code
            cursor[i] += 1

            # j -> i
            pos = cursor[j]
            neighbors[pos] = i
            bond_codes[pos] = code
            cursor[j] += 1

    return indptr, neighbors, bond_codes


# ============================================================
# Numba：构造并排序邻居 token
#
# 每个邻居 token：
#     4-byte bond code + 16-byte neighbor hash
#
# 每个原子的邻居 token 排序后，WL 结果与原子编号顺序无关。
# ============================================================


@njit(cache=True, nogil=True, parallel=True)
def _make_sorted_neighbor_tokens(
    indptr: np.ndarray,
    neighbors: np.ndarray,
    bond_codes: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    num_atoms = indptr.shape[0] - 1
    num_directed_edges = neighbors.shape[0]

    tokens = np.empty(
        (num_directed_edges, 4 + _DIGEST_SIZE),
        dtype=np.uint8,
    )

    for atom_idx in prange(num_atoms):
        start = indptr[atom_idx]
        end = indptr[atom_idx + 1]

        # 构造 token
        for pos in range(start, end):
            code = bond_codes[pos]
            neighbor_idx = neighbors[pos]

            tokens[pos, 0] = np.uint8((code >> 24) & 0xFF)
            tokens[pos, 1] = np.uint8((code >> 16) & 0xFF)
            tokens[pos, 2] = np.uint8((code >> 8) & 0xFF)
            tokens[pos, 3] = np.uint8(code & 0xFF)

            for byte_idx in range(_DIGEST_SIZE):
                tokens[pos, 4 + byte_idx] = labels[
                    neighbor_idx,
                    byte_idx,
                ]

        # 原子度通常很小，插入排序比通用排序更合适
        for pos in range(start + 1, end):
            current = pos

            while current > start:
                is_less = False

                for byte_idx in range(4 + _DIGEST_SIZE):
                    left = tokens[current, byte_idx]
                    right = tokens[current - 1, byte_idx]

                    if left < right:
                        is_less = True
                        break

                    if left > right:
                        break

                if not is_less:
                    break

                for byte_idx in range(4 + _DIGEST_SIZE):
                    tmp = tokens[current, byte_idx]
                    tokens[current, byte_idx] = tokens[
                        current - 1,
                        byte_idx,
                    ]
                    tokens[current - 1, byte_idx] = tmp

                current -= 1

    return tokens


# ============================================================
# Atom WL hash
# ============================================================


def atom_hash_func(
    mol: Chem.Mol,
    radius: int = 6,
    cache_limit: int = 100_000,
) -> AtomHashes:
    """
    对整个分子执行 radius 轮带边标签的 node-wise WL。

    radius=0:
        只使用原子本身属性。

    radius=1:
        加入一跳邻居及相关键属性。

    radius=N:
        加入 N 跳范围内的 WL 化学环境。

    不会：
    - 提取子分子；
    - 调用 SanitizeMol；
    - 调用 FastFindRings；
    - 调用 NetworkX；
    - 为每个原子单独遍历一次 radius 子图。
    """
    if radius < 0:
        raise ValueError("radius must be >= 0")

    indptr, neighbors, bond_codes = _build_csr(mol)
    labels = _initial_atom_hashes(mol)

    num_atoms = mol.GetNumAtoms()

    for iteration in range(radius):
        tokens = _make_sorted_neighbor_tokens(
            indptr,
            neighbors,
            bond_codes,
            labels,
        )

        next_labels = np.empty_like(labels)

        # 对重复环境只计算一次 BLAKE2b；
        # cache_limit 防止极端情况下字典无限增长。
        cache: dict[bytes, bytes] = {}
        iteration_bytes = _UINT32_STRUCT.pack(iteration + 1)

        for atom_idx in range(num_atoms):
            start = int(indptr[atom_idx])
            end = int(indptr[atom_idx + 1])

            # 布局是固定长度的，因此不存在字符串拼接歧义：
            # 16-byte self hash
            # + N * (4-byte bond code + 16-byte neighbor hash)
            signature = labels[atom_idx].tobytes() + tokens[start:end].tobytes()

            digest = cache.get(signature)

            if digest is None:
                digest = _blake16(
                    _PERSON_ATOM_WL,
                    iteration_bytes + signature,
                )

                if len(cache) < cache_limit:
                    cache[signature] = digest

            next_labels[atom_idx] = np.frombuffer(
                digest,
                dtype=np.uint8,
            )

        labels = next_labels

    return AtomHashes(
        hashes=labels,
        indptr=indptr,
        neighbors=neighbors,
        bond_codes=bond_codes,
    )


# ============================================================
# Bonded hashes
#
# 不提前存储 bonded hashes。
# 使用时直接根据 atom hash + 实际键属性现场计算。
# ============================================================


def bond_hash(
    atom_hashes: AtomHashes,
    i: int,
    j: int,
) -> bytes:
    """
    i-j == j-i
    """
    hi = atom_hashes[i]
    hj = atom_hashes[j]
    edge = _UINT32_STRUCT.pack(atom_hashes.bond_code(i, j))

    if hj < hi:
        hi, hj = hj, hi

    return _blake16(
        _PERSON_BOND,
        hi + edge + hj,
    )


def angle_hash(
    atom_hashes: AtomHashes,
    i: int,
    j: int,
    k: int,
) -> bytes:
    """
    j 是中心原子。

    i-j-k == k-j-i
    """
    hi = atom_hashes[i]
    hj = atom_hashes[j]
    hk = atom_hashes[k]

    edge_ij = _UINT32_STRUCT.pack(atom_hashes.bond_code(i, j))
    edge_jk = _UINT32_STRUCT.pack(atom_hashes.bond_code(j, k))

    forward = hi + edge_ij + hj + edge_jk + hk
    reverse = hk + edge_jk + hj + edge_ij + hi

    canonical = forward if forward <= reverse else reverse

    return _blake16(_PERSON_ANGLE, canonical)


def dihedral_hash(
    atom_hashes: AtomHashes,
    i: int,
    j: int,
    k: int,
    l: int,
) -> bytes:
    """
    i-j-k-l == l-k-j-i
    """
    hi = atom_hashes[i]
    hj = atom_hashes[j]
    hk = atom_hashes[k]
    hl = atom_hashes[l]

    edge_ij = _UINT32_STRUCT.pack(atom_hashes.bond_code(i, j))
    edge_jk = _UINT32_STRUCT.pack(atom_hashes.bond_code(j, k))
    edge_kl = _UINT32_STRUCT.pack(atom_hashes.bond_code(k, l))

    forward = hi + edge_ij + hj + edge_jk + hk + edge_kl + hl

    reverse = hl + edge_kl + hk + edge_jk + hj + edge_ij + hi

    canonical = forward if forward <= reverse else reverse

    return _blake16(_PERSON_DIHEDRAL, canonical)


def improper_hash(
    atom_hashes: AtomHashes,
    center: int,
    i: int,
    j: int,
    k: int,
) -> bytes:
    """
    center 是中心原子。

    三个外围原子的任意排列得到相同结果。
    """
    center_hash = atom_hashes[center]

    arms = [
        _UINT32_STRUCT.pack(atom_hashes.bond_code(center, i)) + atom_hashes[i],
        _UINT32_STRUCT.pack(atom_hashes.bond_code(center, j)) + atom_hashes[j],
        _UINT32_STRUCT.pack(atom_hashes.bond_code(center, k)) + atom_hashes[k],
    ]

    arms.sort()

    return _blake16(
        _PERSON_IMPROPER,
        center_hash + arms[0] + arms[1] + arms[2],
    )


from time import perf_counter


def profile_atom_hash_func(
    mol: Chem.Mol,
    radius: int = 6,
    cache_limit: int = 100_000,
):
    total_start = perf_counter()

    t0 = perf_counter()
    indptr, neighbors, bond_codes = _build_csr(mol)
    csr_time = perf_counter() - t0

    degree = np.diff(indptr)

    t0 = perf_counter()
    labels = _initial_atom_hashes(mol)
    initial_time = perf_counter() - t0

    print(
        f"atoms={mol.GetNumAtoms()}, "
        f"directed_edges={len(neighbors)}, "
        f"max_degree={degree.max(initial=0)}, "
        f"mean_degree={degree.mean():.3f}"
    )
    print(f"CSR:          {csr_time:.6f} s")
    print(f"atom init:    {initial_time:.6f} s")

    num_atoms = mol.GetNumAtoms()

    for iteration in range(radius):
        t0 = perf_counter()

        tokens = _make_sorted_neighbor_tokens(
            indptr,
            neighbors,
            bond_codes,
            labels,
        )

        token_time = perf_counter() - t0

        t0 = perf_counter()

        next_labels = np.empty_like(labels)
        cache = {}

        cache_hits = 0
        digest_calculations = 0
        iteration_bytes = _UINT32_STRUCT.pack(iteration + 1)

        for atom_idx in range(num_atoms):
            start = int(indptr[atom_idx])
            end = int(indptr[atom_idx + 1])

            signature = labels[atom_idx].tobytes() + tokens[start:end].tobytes()

            digest = cache.get(signature)

            if digest is None:
                digest_calculations += 1

                digest = _blake16(
                    _PERSON_ATOM_WL,
                    iteration_bytes + signature,
                )

                if len(cache) < cache_limit:
                    cache[signature] = digest
            else:
                cache_hits += 1

            next_labels[atom_idx] = np.frombuffer(
                digest,
                dtype=np.uint8,
            )

        digest_time = perf_counter() - t0
        labels = next_labels

        print(
            f"round {iteration + 1}: "
            f"tokens={token_time:.6f} s, "
            f"digest={digest_time:.6f} s, "
            f"cache_size={len(cache)}, "
            f"hits={cache_hits}, "
            f"digests={digest_calculations}"
        )

    print(f"total:        {perf_counter() - total_start:.6f} s")

    return AtomHashes(
        hashes=labels,
        indptr=indptr,
        neighbors=neighbors,
        bond_codes=bond_codes,
    )


if __name__ == "__main__":
    from chemfast.conf.fast_sanitize import fast_sanitize

    mol = Chem.RWMol()
    for _ in range(6):
        mol.AddAtom(Chem.Atom(6))
    mol.AddBond(0, 1, Chem.BondType.DOUBLE)
    mol.AddBond(1, 2, Chem.BondType.SINGLE)
    mol.AddBond(2, 3, Chem.BondType.DOUBLE)
    mol.AddBond(3, 4, Chem.BondType.SINGLE)
    mol.AddBond(4, 5, Chem.BondType.DOUBLE)
    mol.AddBond(5, 0, Chem.BondType.SINGLE)
    m = mol.GetMol()

    import time

    # m0 = Chem.MolFromSmiles("Cc1ccccc1C"*1000 + '.C')
    m1 = Chem.MolFromSmiles("Cc1ccccc1C" * 10000 + ".C", sanitize=False)
    print(m1.GetNumAtoms())
    s = time.time()
    fast_sanitize(m1)
    # Chem.SanitizeMol(m)
    print(time.time() - s)
    s = time.time()
    atom_hash_func(m1)
    # Chem.SanitizeMol(m)
    print(time.time() - s)
    # for a in m1.GetAtoms():
    #     print(a.GetIsAromatic())
    # print(Chem.MolToSmiles(m))
    # fp_bit0 = AllChem.GetMorganFingerprintAsBitVect(m0, radius=2, nBits=2048)
    # fp_bit1 = AllChem.GetMorganFingerprintAsBitVect(m1, radius=2, nBits=2048)
    # print(np.allclose(fp_bit0, fp_bit1))
