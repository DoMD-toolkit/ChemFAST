import json
from pathlib import Path

from chemfast.cg.pipeline import build_pygamd_protocol, get_cgff_parameters

work_dir = Path(__file__).resolve().parent
dsl_file = work_dir / "system.json"
cg_dir = work_dir / "cg"

protocol = build_pygamd_protocol(
    dsl_file,
    output_dir=cg_dir,
    use_builtin=True,
    mass_density=0.2,
    random_seed=2026,
)
dsl = json.loads(dsl_file.read_text(encoding="utf-8"))
get_cgff_parameters(dsl, output=cg_dir / "cg_parameters.json")
print(protocol.xml)
print(protocol.runner)
