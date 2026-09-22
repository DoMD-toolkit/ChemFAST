# Copyright (c) 2024 Rui Shi, Mingyang Li, Hujun Qian
# Licensed under the PolyForm Noncommercial License v1.0.0
#  https://polyformproject.org/licenses/noncommercial/1.0.0/
# Commercial Licensing
#  For commercial use or to enter into a commercial license agreement, please contact: lmy23@mails.jlu.edu.cn

from collections import namedtuple

from rdkit import Chem
from rdkit.Chem import rdChemReactions

atom_info = namedtuple('atom_info',
                       ('reactant_id', 'reactant_atom_id', 'product_id', 'product_atom_id', 'atomic_number'))
bond_info = namedtuple('bond_info',
                       ('product_id', 'product_atoms_id', 'reactants_id', 'reactant_atoms_id', 'bond_type',
                        'bond_stereo', 'bond_dir', 'stereo_atoms', 'status'))


def process_reactants(reactants):
    for r_idx, m0 in enumerate(reactants):
        m = Chem.Mol(m0)
        for atom in m.GetAtoms():
            atom.SetIntProp('reactant_idx', r_idx)
    return reactants


def map_reacting_atoms(reaction):
    reacting_map = {}
    for r_idx in range(reaction.GetNumReactantTemplates()):
        # print(r_idx, reaction.GetNumReactantTemplates())
        rt = reaction.GetReactantTemplate(r_idx)
        for atom in rt.GetAtoms():
            if atom.GetAtomMapNum():
                reacting_map[atom.GetAtomMapNum()] = r_idx
    return reacting_map


def map_atoms(products, reacting_map):
    amap = []
    ra_dict = {}
    for ip, p in enumerate(products):
        p_idx = ip
        for a in p.GetAtoms():
            p_aidx = a.GetIdx()
            old_mapno = a.GetPropsAsDict().get('old_mapno')
            r_aidx = a.GetPropsAsDict().get("react_atom_idx")
            if old_mapno is not None:  # reacting atoms
                r_idx = reacting_map[old_mapno]
                if ra_dict.get(r_idx) is None:
                    ra_dict[r_idx] = []
                ra_dict[r_idx].append(r_aidx)
                amap.append(atom_info(r_idx, r_aidx, p_idx, p_aidx, a.GetAtomicNum()))
    return amap, ra_dict


def atom_map(products, reaction):
    reacting_map = map_reacting_atoms(reaction)
    amap, reacting_atoms = map_atoms(products, reacting_map)
    return amap, reacting_atoms


def bond_map(reactants: list, products: list, reaction: rdChemReactions.ChemicalReaction, smarts: str) -> list:
    amap, reacting_atoms = atom_map(products, reaction)
    # print(smarts)
    reactants_smarts, products_smarts = smarts.split('>>')
    reactants_smarts = reactants_smarts.split('.')
    prods_smarts = products_smarts.split('.')

    r2p = {(m.reactant_id, m.reactant_atom_id): (m.product_id, m.product_atom_id) for m in amap}
    p2r = {(m.product_id, m.product_atom_id): (m.reactant_id, m.reactant_atom_id) for m in amap}

    rpatts = {m.reactant_id: Chem.MolFromSmarts(reactants_smarts[m.reactant_id]) for m in amap}
    ppatts = {m.product_id: Chem.MolFromSmarts(prods_smarts[m.product_id]) for m in amap}
    rsubmatches = {m.reactant_id: reactants[m.reactant_id].GetSubstructMatch(rpatts[m.reactant_id]) for m in amap}
    psubmatches = {m.product_id: products[m.product_id].GetSubstructMatch(ppatts[m.product_id]) for m in amap}

    for m in amap:
        r_atom = reactants[m.reactant_id].GetAtomWithIdx(m.reactant_atom_id)
        p_atom = products[m.product_id].GetAtomWithIdx(m.product_atom_id)

        r_neighbors = [a for a in r_atom.GetNeighbors() if (m.reactant_id, a.GetIdx()) not in r2p]
        p_neighbors = [a for a in p_atom.GetNeighbors() if (m.product_id, a.GetIdx()) not in p2r]

        for rn in r_neighbors:
            for pn in p_neighbors:
                rn_idx = rn.GetIdx()
                pn_idx = pn.GetIdx()
                if rn_idx not in rsubmatches[m.reactant_id] or pn_idx not in psubmatches[m.product_id]:
                    continue
                # if p2r.get((m.product_id, pn.GetIdx())) is not None or p2r.get((m.product_id, pn.GetIdx()))[0] != m.reactant_id:
                #    continue
                if (m.product_id,
                    pn.GetIdx()) not in p2r and rn.GetAtomicNum() == pn.GetAtomicNum() and rn.GetIsAromatic() == pn.GetIsAromatic():
                    # if (m.product_id, pn.GetIdx()) not in p2r and :
                    r_key = (m.reactant_id, rn.GetIdx())
                    p_key = (m.product_id, pn.GetIdx())
                    # print(f"Mapping untagged neighbor: Reactant {r_key} -> Product {p_key} (Element: {rn.GetSymbol()})")
                    r2p[r_key] = p_key
                    p2r[p_key] = r_key
                    break

    res = []

    for ir, r in enumerate(reactants):
        for bond in r.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

            if (ir, a) not in r2p or (ir, b) not in r2p:
                continue

            pid_a, paid_a = r2p[(ir, a)]
            pid_b, paid_b = r2p[(ir, b)]

            t, s, bdir = bond.GetBondType(), bond.GetStereo(), bond.GetBondDir()
            satoms = list(bond.GetStereoAtoms())

            if pid_a != pid_b:
                res.append(bond_info(pid_a, (paid_a, paid_b), (ir, ir), (a, b), t, s, bdir, satoms, 'deleted'))
            else:
                p_bond = products[pid_a].GetBondBetweenAtoms(paid_a, paid_b)
                if p_bond is None:
                    res.append(bond_info(pid_a, (paid_a, paid_b), (ir, ir), (a, b), t, s, bdir, satoms, 'deleted'))

    for ip, p in enumerate(products):
        for bond in p.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

            if (ip, a) not in p2r or (ip, b) not in p2r:
                continue

            rid_a, raid_a = p2r[(ip, a)]
            rid_b, raid_b = p2r[(ip, b)]

            t, s, bdir = bond.GetBondType(), bond.GetStereo(), bond.GetBondDir()
            satoms = list(bond.GetStereoAtoms())

            if rid_a != rid_b:
                res.append(bond_info(ip, (a, b), (rid_a, rid_b), (raid_a, raid_b), t, s, bdir, satoms, 'new'))
            else:
                r_bond = reactants[rid_a].GetBondBetweenAtoms(raid_a, raid_b)
                if r_bond is None:
                    res.append(bond_info(ip, (a, b), (rid_a, rid_b), (raid_a, raid_b), t, s, bdir, satoms, 'new'))
                elif r_bond.GetBondType() != t:
                    res.append(bond_info(ip, (a, b), (rid_a, rid_b), (raid_a, raid_b), t, s, bdir, satoms, 'changed'))

    return res
