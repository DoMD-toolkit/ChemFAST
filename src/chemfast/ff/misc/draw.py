import numpy as np
from rdkit import Chem
from rdkit import Geometry
from rdkit.Chem import rdCoordGen
from rdkit.Chem.Draw import rdMolDraw2D


def print_mol_ascii(mol_raw, highlight=0):
    return ''
    # TODO: IF THE MOLECULE is TOO LARGE, IT SHOULD HAVE A CUTOFF
    # DO NOT RENDER WHOLE MOLECULE
    # 1. Draw mol with rdCoordGen
    highlight_h = False
    for i, a in enumerate(mol_raw.GetAtoms()):
        a.SetIntProp('idx', i)
    mol = Chem.RemoveAllHs(mol_raw, sanitize=False)
    if mol_raw.GetAtomWithIdx(highlight).GetSymbol() == 'H':
        highlight_h = True
    if highlight_h:
        highlight_idx = mol_raw.GetAtomWithIdx(highlight).GetNeighbors()[0].GetIntProp('idx')
    else:
        highlight_idx = mol_raw.GetAtomWithIdx(highlight).GetIntProp('idx')  # highlight itself or neighbor for H
    rdCoordGen.AddCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(300, 300)
    drawer.drawOptions().fixedBondLength = 40
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    # 2. Get atom x,y coordinates
    # adapted from: https://gist.github.com/greglandrum/0c8b80825826392a6519af7519862baf
    atom_index = []
    atom_symbols = []
    atom_xpos = []
    atom_ypos = []
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        atom_index.append(i)
        symb = mol.GetAtomWithIdx(i).GetSymbol()
        if mol.GetAtomWithIdx(i).GetIsAromatic():
            symb = symb.lower()
        if mol.GetAtomWithIdx(i).GetIntProp('idx') == highlight_idx:
            if highlight_h:
                symb = f"[H]{symb}"
            else:
                symb = f"[{symb}]"
        atom_symbols.append(symb)
        pos_A = conf.GetAtomPosition(i)
        pos_point = Geometry.Point2D(pos_A.x, pos_A.y)
        dpos = drawer.GetDrawCoords(pos_point)
        atom_xpos.append(dpos.x)
        atom_ypos.append(dpos.y)

    # 3. Resize and shift coordinates
    atom_coords = list(zip(atom_xpos, atom_ypos))
    # A scale between 1/12 and 1/20 seems to work okay without distorting printing aspect ratio
    scale_atom_coords = [(p[0] / 14, p[1] / 14) for p in atom_coords]
    round_atom_coords = [(round(p[0]), round(p[1])) for p in scale_atom_coords]
    # Get min and max for x,y
    xmin = min(c[0] for c in round_atom_coords)
    xmax = max(c[0] for c in round_atom_coords)
    ymin = min(c[1] for c in round_atom_coords)
    ymax = max(c[1] for c in round_atom_coords)
    # figure out what to substract to shift coordinates by a min
    if xmin < ymin:
        xymin = xmin
    else:
        xymin = ymin
    # shift by xymin to move mol closer to origin
    norm_atom_coords = [(p[0] - xymin, p[1] - xymin) for p in round_atom_coords]

    # 4. Get Bond connection information for drawing bonds
    atom_begin = []
    atom_end = []
    bond_type = []

    for bond in mol.GetBonds():
        atom_begin.append(bond.GetBeginAtomIdx())
        atom_end.append(bond.GetEndAtomIdx())
        bond_type.append(bond.GetBondTypeAsDouble())
        # this could be used to define single, double, arom with symbols

    # 5. Get coordinates of middle of bond for plotting

    begin_connection_xpos = []
    begin_connection_ypos = []
    for i in atom_begin:
        begin_connection_xpos.append(norm_atom_coords[i][0])
        begin_connection_ypos.append(norm_atom_coords[i][1])

    end_connection_xpos = []
    end_connection_ypos = []
    for i in atom_end:
        end_connection_xpos.append(norm_atom_coords[i][0])
        end_connection_ypos.append(norm_atom_coords[i][1])

    midx_bond = []
    for x1, x2 in zip(begin_connection_xpos, end_connection_xpos):
        middle = (x1 + x2) / 2
        midx_bond.append(middle)

    midy_bond = []
    for y1, y2 in zip(begin_connection_ypos, end_connection_ypos):
        middle = (y1 + y2) / 2
        midy_bond.append(middle)
    # xy coordinates of center of bonds (*) to plot
    bond_center_points = list(zip(midx_bond, midy_bond))
    bond_symbol_list = {}
    for bxy, bt in zip(bond_center_points, bond_type):
        if np.isclose(bt, 1):
            bond_symbol_list[bxy] = '-'
        elif np.isclose(bt, 1.5):
            bond_symbol_list[bxy] = ':'
        elif np.isclose(bt, 2):
            bond_symbol_list[bxy] = '='
        elif np.isclose(bt, 3):
            bond_symbol_list[bxy] = '#'
        else:
            bond_symbol_list[bxy] = '*'
    bond_symbol_list = bond_symbol_list
    # 6. Setup a max plotting print range based on coordinates

    if xmax > ymax:
        xymax = xmax
    else:
        xymax = ymax
    # this just gets the steps for x.y coordinates in increments of 0.5 for the print grid
    start = 0
    stop = round(xymax + 1)
    samples = round((stop * 2) + 1)
    print_range = np.linspace(start, stop, samples)

    # 7. Print the molecule!
    # Adapted from: https://stackoverflow.com/questions/16240358/ascii-graph-in-python
    # https://creativecommons.org/licenses/by-sa/3.0/
    # bond_symbol = '*'
    fill_symbol = ' '
    res = ''
    for y in print_range:
        chars = []
        for x in print_range:
            if (x, y) in norm_atom_coords:
                index_value = norm_atom_coords.index((x, y))
                chars.append(atom_symbols[index_value])
            elif (x, y) in bond_center_points:
                bond_symbol = bond_symbol_list[(x, y)]
                chars.append(bond_symbol)
            else:
                chars.append(fill_symbol)
        # print(' '.join(chars))
        res += ' '.join(chars) + '\n'
    res_l = res.split('\n')
    while '' in res_l:
        res_l.remove('')
    _l = res_l[-1]
    while _l.isspace():
        _l = res_l.pop()
    res_l.append(_l)
    res_l.insert(0, ' ')  # avoiding the following "while" loop errors
    _l = res_l[0]
    while _l.isspace():
        _l = res_l.pop(0)
    res_l.insert(0, _l)
    # res_l.insert(0, "Bond order: Symbol\n1: `-', 1.5: `:', 2: `=', 3: `#', other: `*'")
    res_l.insert(0, ">" + "-" * 90 + "<")
    res_l.append(">" + "-" * 90 + "<")
    return '\n'.join(res_l)


if __name__ == '__main__':
    mol = Chem.MolFromSmiles('c1ccccc1')
    print(print_mol_ascii(mol, 3))
