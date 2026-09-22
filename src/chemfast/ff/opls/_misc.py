from collections import namedtuple

GMXRule = namedtuple(
    "GMXRule",
    [
        "opls_num",
        "bond_type",
        "mass",
        "sigma",
        "epsilon",
        "charge",
        "smarts",
        "desc",
        "patt",
        "ptype",
    ],
)

GMXBond = namedtuple("GMXBond", ["atom_types", "r0", "k", "ftype"])
GMXAngle = namedtuple("GMXAngle", ["atom_types", "t0", "k", "ftype"])
GMXDihedral = namedtuple(
    "GMXDihedral", ["atom_types", "ftype", "c0", "c1", "c2", "c3", "c4", "c5"]
)
GMXImproper = namedtuple(
    "GMXImproper", ["atom_types", "ftype", "c0", "c1", "c2", "c3", "c4", "c5"]
)

OPLSAtom = namedtuple(
    "OPLSAtom",
    ["opls_num", "bond_type", "element", "mass", "sigma", "epsilon", "charge", "ptype"],
)
OPLSBond = namedtuple("OPLSBond", ["indices", "r0", "k", "ftype"])
OPLSAngle = namedtuple("OPLSAngle", ["indices", "t0", "k", "ftype"])
OPLSDihedral = namedtuple(
    "OPLSDihedral", ["indices", "ftype", "c0", "c1", "c2", "c3", "c4", "c5"]
)
OPLSImproper = namedtuple("OPLSImproper", ["indices", "ftype", "params"])
