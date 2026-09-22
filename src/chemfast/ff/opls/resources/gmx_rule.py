import os

__this_dir__ = os.path.abspath(os.path.dirname(__file__))

from collections import deque

from rdkit import Chem
import pickle

from chemfast.ff.opls._misc import GMXRule, GMXBond, GMXAngle, GMXDihedral

opls_nb_dic = {}

for line in open(os.path.join(__this_dir__, 'gmx/ffnonbonded.itp')):
    if line.startswith(';'):
        continue
    line = line.strip().split()
    while '' in line:
        line.remove('')
    if len(line) < 8:
        continue
    opls_nb_dic[line[0].strip()] = (line[1].strip(), float(line[3].strip()), float(line[4].strip()),
                                    float(line[6].strip()), float(line[7].strip()), line[5].strip())


def get_pattern_diameter(pat: Chem.Mol) -> int:
    """
    计算模式分子 pat 的图直径；如果直径算出来 <= 0，
    就退而求其次用 pat.GetNumAtoms()-1（至少取 1）。
    """
    n = pat.GetNumAtoms()
    # 最简单的 fallback：如果 pat 只有 1 个 Atom， diameter 应该至少取 1
    if n <= 1:
        return 1

    # 正常计算直径
    dist = [[999999] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
        q = deque([i])
        seen = {i}
        while q:
            u = q.popleft()
            for b in pat.GetAtomWithIdx(u).GetBonds():
                v = b.GetOtherAtomIdx(u)
                if v not in seen:
                    seen.add(v)
                    dist[i][v] = dist[i][u] + 1
                    q.append(v)
    maxd = max(dist[i][j]
               for i in range(n)
               for j in range(n)
               if dist[i][j] < 999999)

    # 如果算出来的 maxd 小于 1，就用 n-1 作为直径
    if maxd < 1:
        maxd = n - 1
    return maxd


rules = []
for line in open(os.path.join(__this_dir__, 'gmx/STaGE_opls_tomoltemplate_opls.txt')):
    if line.startswith('*'):
        continue
    line = line.strip().split("|")
    while '' in line:
        line.remove('')
    if len(line) < 1:
        continue

    # print(line)
    element = line[0].strip()
    bond_type = line[1].strip()
    opls_num = line[2].strip()
    patt = Chem.MolFromSmarts(line[3].strip())
    smts = line[3].strip()
    charge = float(line[5].strip())
    desc = line[6].strip()
    r = opls_nb_dic.get(opls_num)

    # print((element, bond_type, patt, charge, desc))
    # ele, bond_type, pattern, mass, sigma, epsilon, charge, SMARTS, description
    if r is None:
        continue
    ptype = r[-1]

    rule = GMXRule(bond_type=bond_type, mass=r[1], sigma=r[3], epsilon=r[4],
                   charge=charge, smarts=smts, desc=desc, patt=patt, ptype=ptype, opls_num=opls_num)
    rules.append(rule)
    # print(rules[-1])
    # print(patt.GetNumAtoms())
    # print(get_pattern_diameter(patt))

pickle.dump(rules, open(os.path.join(__this_dir__, 'gmx/non_bond_rules.dat'), 'wb'))

rules = {}
for line in open(os.path.join(__this_dir__, 'gmx/ffbonded.itp')):
    if line.startswith('*'):
        continue
    if line.startswith(';'):
        continue
    if line.startswith('#'):
        continue
    line = line.strip().split()
    while '' in line:
        line.remove('')
    if len(line) < 2:
        continue

    if 'bondtypes' in line:
        flag = 0
        continue

    if 'angletypes' in line:
        flag = 1
        continue

    if 'dihedraltypes' in line:
        flag = 2
        continue

    if 'constrainttypes' in line:
        flag = 3
        continue

    if flag == 3:
        continue

    if flag == 0:
        name = f"{line[0].strip()}-{line[1].strip()}||{line[1].strip()}-{line[0].strip()}"
        rules[name] = GMXBond(atom_types=f"{line[0].strip()}-{line[1].strip()}", r0=float(line[3]),
                              k=float(line[4]), ftype=int(line[2]))
    if flag == 1:
        name = f"{line[0].strip()}-{line[1].strip()}-{line[2].strip()}"
        rules[name] = GMXAngle(atom_types=f"{line[0].strip()}-{line[1].strip()}-{line[2].strip()}",
                               t0=float(line[4]), k=float(line[5]), ftype=int(line[3]))
    if flag == 2:
        name = f"{line[0].strip()}-{line[1].strip()}-{line[2].strip()}-{line[3].strip()}"
        rules[name] = GMXDihedral(atom_types=f"{line[0].strip()}-{line[1].strip()}-{line[2].strip()}-{line[3].strip()}",
                                  ftype=int(line[4]), c0=float(line[5]), c1=float(line[6]), c2=float(line[7]),
                                  c3=float(line[8]), c4=float(line[9]), c5=float(line[10]))

for line in open(os.path.join(__this_dir__, 'gmx/boss_bonded.sb')):
    line = line.strip().split()
    while '' in line:
        line.remove('')
    name = line[0]
    name1 = name.split("-")
    while '' in name1:
        name1.remove("")
    if len(name1) == 2:
        rules[name] = GMXBond(atom_types=f"{name1[0]}-{name1[1]}", r0=float(line[2]), k=float(line[3]),
                              ftype=int(line[1]))
    if len(name1) == 3:
        rules[name] = GMXAngle(atom_types=f"{name1[0]}-{name1[1]}-{name1[2]}",
                               t0=float(line[2]), k=float(line[3]), ftype=int(line[1]))
    if len(name1) == 4:
        if int(line[1]) == 4:
            continue
        if '?' in name:
            name = name.replace("?", "")
        rules[name] = GMXDihedral(atom_types=f"{name1[0]}-{name1[1]}-{name1[2]}-{name1[3]}", ftype=int(line[1]),
                                  c0=float(line[2]), c1=float(line[3]), c2=float(line[4]), c3=float(line[5]),
                                  c4=float(line[6]), c5=float(line[7]))

pickle.dump(rules, open(os.path.join(__this_dir__, 'gmx/bonded_rules.dat'), 'wb'))
