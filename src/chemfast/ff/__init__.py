from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol, Tuple, runtime_checkable


class FF_Type(str, Enum):
    OPLS = "OPLS"
    AMBER = "AMBER"
    CG = "CG"
    CHARMM = "CHARMM"


class InteractionType(str, Enum):
    BOND = "BOND"
    PAIR = "PAIR"
    PAIR_NB = "PAIR_NB"
    ANGLE = "ANGLE"
    DIHEDRAL = "DIHEDRAL"
    IMPROPER = "IMPROPER"


# UNIFIED INTERFACE OF PARAMS

@runtime_checkable
class InteractionParams(Protocol):
    """Any bonded parameter class accepted by BondedInteraction."""

    @property
    def ftype(self) -> int:
        ...


# NONBOND PARAMS

@dataclass(frozen=True)
class LJParams:
    """Lennard-Jones parameters: epsilon (kJ/mol), sigma (nm)."""
    epsilon: float
    sigma: float

    @property
    def ftype(self) -> int:  # satisfy the protocal, to fixed to be 1
        return 1


@dataclass(frozen=True)
class CGPolyParams:
    """Coarse-grained polynomial parameters."""
    c0: float
    c1: float
    c2: float
    c3: float
    ftype: int = 2  # default to be 2, not fixed


# BONDED PARAMS

@dataclass(frozen=True)
class HarmonicParams:
    """Generalized harmonic parameters; r0 is the equilibrium coordinate."""
    k: float
    r0: float
    ftype: int = 1  # or 6 for gmx, not fixed


@dataclass(frozen=True)
class PeriodicParams:
    """Periodic proper or improper dihedral parameters."""
    k: float
    phi0: float
    multiplicity: int

    @property
    def ftype(self) -> int:  # satisfy the protocal, to fixed to be 4
        return 4


@dataclass(frozen=True)
class DihedralParams:
    """Six-coefficient Ryckaert-Bellemans parameters."""
    c0: float
    c1: float
    c2: float
    c3: float
    c4: float
    c5: float
    ftype: int = 3


# UNIFIED RECORDS

@dataclass(frozen=True)
class Atom:
    ff_type: FF_Type
    ff_atom_type: str
    bond_type: str = ""
    ptype: str = "A"
    params: Optional[InteractionParams] = None
    element: Optional[str] = None
    mass: Optional[float] = 1.0
    charge: Optional[float] = 0.0
    hsp: Optional[Tuple[float, float, float]] = None

    @property
    def type(self) -> Tuple[str, ...] | None:
        """Compatibility alias for the former CGNonbond.type attribute."""
        return (self.ff_atom_type,)  # VERY VERY VERY MORON


@dataclass(frozen=True)
class Bonded:
    ff_type: FF_Type
    itype: InteractionType
    indices: Tuple[int, ...]
    params: Optional[InteractionParams] = None
    name: Optional[str] = ""
    ff_atom_types: Optional[Tuple[str, ...]] = None

    @property
    def type(self) -> Tuple[str, ...] | None:
        """Compatibility alias for the former CGNonbond.type attribute."""
        return self.ff_atom_types  # VERY VERY VERY MORON
