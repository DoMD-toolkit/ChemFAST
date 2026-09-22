import re
from io import StringIO
from pathlib import Path
from typing import Optional, Sequence
from xml.etree import cElementTree
from xml.sax.saxutils import escape

import numpy as np
from rdkit import Chem


def pbc(x, l):
    """Applies periodic boundary conditions to a coordinate.

    Args:
        x (float or np.ndarray): The input coordinate.
        l (float or np.ndarray): The box length in the corresponding dimension.

    Returns:
        float or np.ndarray: The coordinate wrapped within [-l/2, l/2].
    """
    return x - l * np.rint(x / l)


def control_in(control_file):
    """Parses a control input file.

    Args:
        control_file (str): The path to the control file.
    """
    pass


class Box(object):
    """Represents the simulation box dimensions and tilt factors.

    Attributes:
        xy (float): Tilt factor xy.
        xz (float): Tilt factor xz.
        yz (float): Tilt factor yz.
    """

    def __init__(self):
        """Initializes a Box object with zero tilt factors."""
        self.xy = 0
        self.xz = 0
        self.yz = 0
        return

    def update(self, dic):
        """Updates the box attributes from a dictionary.

        Args:
            dic (dict): Dictionary containing box attributes.
        """
        self.__dict__.update(dic)


class XmlParser(object):
    """Parses XML files containing simulation data.

    This class reads an XML file and extracts specific data elements, optionally filtering
    by a list of needed tags. It handles specific structures like simulation boxes,
    reaction lists, and templates, while parsing other data into numpy arrays.

    Attributes:
        box (Box): The simulation box object.
        data (dict): A dictionary storing parsed data arrays and structures (e.g., 'reaction', 'template').
    """

    def __init__(self, filename, needed=None):
        """Initializes the XmlParser.



        Parsing logic:
        1. Reads root attributes and sets them as instance attributes (e.g., step, time).
        2. Iterates through child elements.
        3. 'box' tags update the self.box object.
        4. Other tags are filtered against the `needed` list if provided.
        5. 'reaction' tags are parsed into lists of reaction steps.
        6. 'template' tags are evaluated as Python dictionaries.
        7. All other tags are parsed as numpy arrays from text content.

        Args:
            filename (str): Path to the XML file to parse.
            needed (list, optional): A list of XML tag names to extract. If provided,
                tags not in this list (except 'box') are skipped. Defaults to None.
        """
        tree = cElementTree.ElementTree(file=filename)
        root = tree.getroot()
        self.box = Box()
        self.data = {}
        needed = [] if needed is None else needed
        for key in root[0].attrib:
            self.__dict__[key] = int(root[0].attrib[key])
        for element in root[0]:
            if element.tag == 'box':
                self.box.update(element.attrib)
                continue
            if (len(needed) > 0) and (element.tag not in needed):
                continue
            if element.tag == 'reaction':
                self.data['reaction'] = []
                reaction_list = element.text.strip().split('\n')
                while '' in reaction_list:
                    reaction_list.remove('')
                for l in reaction_list:
                    r = re.split(r'\s+', l)
                    while '' in r:
                        r.remove('')
                    r[1:] = [int(_) for _ in r[1:]]
                    self.data['reaction'].append(r)
                continue
            if element.tag == 'template':
                self.data['template'] = eval('{%s}' % element.text)
                continue
            #if len(element.text.strip()) > 0:
            #    self.data[element.tag] = np.genfromtxt(StringIO(element.text), dtype=None, encoding=None)
            if len(element.text.strip()) > 0:
                data = np.genfromtxt(StringIO(element.text), dtype=None, encoding=None)
                self.data[element.tag] = np.atleast_1d(data)


def write_mols_to_xml(
        mols: Sequence[Chem.Mol],
        box: Sequence[float],
        filename: str = 'chemfast.xml',
        graphs: Optional[Sequence] = None,
        program: str = 'galamost',
        version: str = '1.3'
) -> Path:
    """Write AA RDKit molecules into one XML system.

    Coordinates are read from RDKit conformers. Atom and bond metadata are read
    from the corresponding graph when available; otherwise, element-based types
    are generated.

    Graph node indices must match RDKit atom indices.
    """
    if graphs is not None and len(graphs) != len(mols):
        raise ValueError("graphs and mols must contain the same number of molecules.")

    box = np.asarray(box, dtype=float) * 0.1
    if box.shape != (3,) or np.any(box <= 0):
        raise ValueError(f"box must contain three positive lengths, found {box}.")

    output_path = Path(filename)
    if output_path.suffix.lower() != '.xml':
        output_path = output_path.with_suffix('.xml')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    atom_offsets = []
    n_atoms = 0
    n_bonds = 0
    for mol in mols:
        if mol.GetNumConformers() == 0:
            raise ValueError("Every molecule must contain an RDKit conformer.")
        atom_offsets.append(n_atoms)
        n_atoms += mol.GetNumAtoms()
        n_bonds += mol.GetNumBonds()

    def atom_data(mol_idx, atom_idx):
        atom = mols[mol_idx].GetAtomWithIdx(atom_idx)
        data = graphs[mol_idx].nodes[atom_idx] if graphs is not None else {}
        atom_type = str(data.get('type', atom.GetSymbol()))
        return atom, data, atom_type

    def iter_atoms():
        for mol_idx, mol in enumerate(mols):
            conf = mol.GetConformer()
            offset = atom_offsets[mol_idx]
            for atom_idx in range(mol.GetNumAtoms()):
                atom, data, atom_type = atom_data(mol_idx, atom_idx)
                yield mol_idx, atom_idx, offset + atom_idx, atom, data, atom_type, conf

    def write_section(handle, name, count, lines):
        handle.write(f'<{name} num="{count}">\n')
        for line in lines:
            handle.write(f'{line}\n')
        handle.write(f'</{name}>\n')

    def position_lines():
        for _, atom_idx, _, _, _, _, conf in iter_atoms():
            position = np.asarray(conf.GetAtomPosition(atom_idx), dtype=float) * 0.1
            position -= box * np.floor(position / box + 0.5)
            yield f'{position[0]:.8f} {position[1]:.8f} {position[2]:.8f}'

    def type_lines():
        for _, _, _, _, _, atom_type, _ in iter_atoms():
            yield escape(atom_type)

    def image_lines():
        for _, _, _, _, data, _, _ in iter_atoms():
            image = np.asarray(data.get('image', (0, 0, 0)), dtype=int)
            if image.shape != (3,):
                raise ValueError(f"Invalid image shape: {image.shape}.")
            yield f'{image[0]} {image[1]} {image[2]}'

    def body_lines():
        for _, _, _, _, data, _, _ in iter_atoms():
            yield str(int(data.get('body_id', -1)))

    def opls_type_lines():
        for _, _, _, _, data, atom_type, _ in iter_atoms():
            yield escape(str(data.get('opls_type', atom_type)))

    def monomer_id_lines():
        for mol_idx, _, global_idx, _, data, _, _ in iter_atoms():
            monomer_id = data.get(
                'monomer_id',
                data.get('global_res_id', data.get('res_id', mol_idx))
            )
            yield str(monomer_id)

    def charge_lines():
        for _, _, _, atom, data, _, _ in iter_atoms():
            charge = data.get('charge', atom.GetFormalCharge())
            yield f'{float(charge):.8g}'

    def mass_lines():
        for _, _, _, atom, data, _, _ in iter_atoms():
            yield f'{float(data.get("mass", atom.GetMass())):.8g}'

    def bond_lines():
        for mol_idx, mol in enumerate(mols):
            graph = graphs[mol_idx] if graphs is not None else None
            offset = atom_offsets[mol_idx]
            for bond in mol.GetBonds():
                u = bond.GetBeginAtomIdx()
                v = bond.GetEndAtomIdx()
                _, _, type_u = atom_data(mol_idx, u)
                _, _, type_v = atom_data(mol_idx, v)
                default_type = '-'.join(sorted((type_u, type_v)))

                if graph is not None and graph.has_edge(u, v):
                    bond_type = graph.edges[u, v].get('bond_type', default_type)
                else:
                    bond_type = default_type

                yield f'{escape(str(bond_type))} {offset + u} {offset + v}'

    root = f'{program}_xml'
    with output_path.open('w', encoding='utf-8', newline='\n') as handle:
        handle.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        handle.write(f'<{root} version="{escape(str(version))}">\n')
        handle.write(
            f'<configuration time_step="0" dimensions="3" natoms="{n_atoms}">\n'
        )
        handle.write(
            f'<box lx="{box[0]:.8f}" ly="{box[1]:.8f}" lz="{box[2]:.8f}" '
            f'xy="0.00000000" xz="0.00000000" yz="0.00000000"/>\n'
        )

        write_section(handle, 'position', n_atoms, position_lines())
        write_section(handle, 'type', n_atoms, type_lines())
        write_section(handle, 'image', n_atoms, image_lines())
        write_section(handle, 'body', n_atoms, body_lines())
        write_section(handle, 'opls_type', n_atoms, opls_type_lines())
        write_section(handle, 'monomer_id', n_atoms, monomer_id_lines())
        write_section(handle, 'charge', n_atoms, charge_lines())
        write_section(handle, 'mass', n_atoms, mass_lines())
        write_section(handle, 'bond', n_bonds, bond_lines())
        write_section(handle, 'angle', 0, ())
        write_section(handle, 'dihedral', 0, ())
        write_section(handle, 'improper', 0, ())

        handle.write('</configuration>\n')
        handle.write(f'</{root}>\n')

    return output_path
