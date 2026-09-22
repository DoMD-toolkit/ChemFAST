from pathlib import Path

from rdkit import Chem
from chemfast.ff.pipeline import run_adv_top_mode

work_dir = Path(__file__).resolve().parent
sdf_file = work_dir / "aa" / "atomistic.sdf"
mols = list(Chem.SDMolSupplier(str(sdf_file), removeHs=False))
if not mols or any(mol is None for mol in mols):
    raise ValueError("The input SDF contains no molecules or an unreadable record.")

run_adv_top_mode(
    mols,
    output_dir=str(work_dir / "gromacs"),
    base_name="system",
    useGMX=True,
    useBOSS=True,
    useML=True,
    overwrite=False,
)
