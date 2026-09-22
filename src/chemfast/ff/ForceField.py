import logging
from typing import Literal

from rdkit import Chem

try:
    from chemfast.misc.logger import logger
except ImportError:
    logger = logging.getLogger(__name__)


class FF(object):
    def __init__(self, name: Literal["opls", "cg", "amber"]):
        self.name = name.lower()
        self.params = None
        self.rdmol = None
        self.obmol = None
        self.success = False
        self._missing = None
        self.charges = {}
        self._meta = None
        self._cg_parameter_library = None

    def setup(self, rdmol: Chem.Mol | None = None, obmol=None, cg_graph=None, config=None,
              charge_factor: tuple[float, float] = (1.0, 1.0), **kwargs):
        self.rdmol, self.obmol = rdmol, obmol
        if self.name == "cg":
            from chemfast.cg.cg_ff.interaction import assign_cg_parameters, build_cg_parameters
            if cg_graph is None:
                logger.error("No CG graph provided for CG force field.")
                raise ValueError(
                    "cg_graph is required for CG force field setup.")
            if config is None:
                logger.error(
                    "No config provided for CG force field parameterization.")
                raise ValueError("config is required for CG force field setup.")

            n_conformers = kwargs.pop("n_conformers", 10)
            include_dihedrals = kwargs.pop("include_dihedrals", False)
            random_seed = kwargs.pop("random_seed", 2026)
            bond_k = kwargs.pop("bond_k", 1100.0)
            angle_k = kwargs.pop("angle_k", 25.0)
            dihedral_k = kwargs.pop("dihedral_k", 5.0)

            if self._cg_parameter_library is None:
                self._cg_parameter_library = build_cg_parameters(
                    config,
                    n_conformers=n_conformers,
                    include_dihedrals=include_dihedrals,
                    random_seed=random_seed,
                    bond_k=bond_k,
                    angle_k=angle_k,
                    dihedral_k=dihedral_k
                )

            self.params = assign_cg_parameters(cg_graph,
                                               self._cg_parameter_library)
            self._missing = ([], [], [])
            self._meta = {
                "n_atom": len(self.params[0]), "t_atom": len(self.params[0]),
                "n_bond": len(self.params[1]), "t_bond": len(self.params[1]),
                "n_ang": len(self.params[2]), "t_ang": len(self.params[2]),
                "n_dih": len(self.params[3]), "t_dih": len(self.params[3]),
                "n_imp": 0, "t_imp": 0
            }
            self.success = True
            return self.params
        elif self.name == "opls":
            from chemfast.ff.opls.opls import opls_setup
            formal_charge = Chem.GetFormalCharge(rdmol)
            logger.debug(f"Formal charge of molecule {rdmol}: {formal_charge:.4f}")
            self.params, self._missing, self.success, self._meta = opls_setup(rdmol, obmol, **kwargs)
            if not self.success:
                return self.params

            ion_indices, non_ion_indices = [], []
            for atom in rdmol.GetAtoms():
                indices = ion_indices if atom.GetDegree() == 0 or atom.GetFormalCharge() != 0 else non_ion_indices
                indices.append(atom.GetIdx())
            atom_count = len(self.params[0])
            total_opls_charge = sum(float(self.params[0][idx].charge) for idx in self.params[0])
            logger.debug(f"OPLS raw total charge for {rdmol}: {total_opls_charge:.4f}")
            global_drift_per_atom = total_opls_charge / atom_count
            temp_charges = {
                idx: self.params[0][idx].charge - global_drift_per_atom + formal_charge / atom_count
                for idx in self.params[0]
            }
            need_ion_constraint = False
            for idx in ion_indices:
                atom_formal_charge = rdmol.GetAtomWithIdx(idx).GetFormalCharge()
                if abs(temp_charges[idx]) > abs(atom_formal_charge):
                    need_ion_constraint = True
                    logger.debug(
                        f"Ion constraint triggered by atom {idx} "
                        f"(Calculated: {temp_charges[idx]:.4f} > Formal: {atom_formal_charge})"
                    )
                    break
            if not need_ion_constraint:
                self.charges = {idx: charge * charge_factor[1] for idx, charge in temp_charges.items()}
                logger.info(
                    f"OPLS reset total charge to formal charge {formal_charge * charge_factor[1]:.4f} "
                    "(Global uniform distribution)."
                )
                return self.params

            ion_total_charge = 0.0
            for idx in ion_indices:
                atom_fc = float(rdmol.GetAtomWithIdx(idx).GetFormalCharge())
                self.charges[idx] = atom_fc * charge_factor[1]
                ion_total_charge += atom_fc
            if non_ion_indices:
                target_non_ion_charge = formal_charge - ion_total_charge
                non_ion_opls_charge = sum(float(self.params[0][idx].charge) for idx in non_ion_indices)
                non_ion_count = len(non_ion_indices)
                non_ion_drift_per_atom = non_ion_opls_charge / non_ion_count
                for idx in non_ion_indices:
                    self.charges[idx] = (
                        self.params[0][idx].charge - non_ion_drift_per_atom
                        + target_non_ion_charge / non_ion_count
                    ) * charge_factor[0]
            logger.info(
                f"OPLS reset total charge to formal charge {formal_charge * charge_factor[0]:.4f} "
                f"(Ion constrained, {len(non_ion_indices)} non-ions adjusted)."
            )
            return self.params
        elif self.name == 'amber':
            from chemfast.ff.amber.amber import amber_setup
            self.params, self._missing, self.success, self._meta = amber_setup(rdmol, obmol, **kwargs)
            return self.params

        else:
            raise ValueError(f"Unsupported force field: {self.name!r}")


    # TODO: add MD modules
    def energy(self):
        pass

    def forces(self):
        pass

    def hessian(self):
        pass

    def optimize(self, runs: int = 1000):
        pass
