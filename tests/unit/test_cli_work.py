"""CLI path handling and strict ChemFAST density SDF, no PyGAMD installation required."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from chemfast.cli import _absolute_filler_paths, _output_filename, build_parser, main
# Load this leaf module without importing chemfast.conf's heavy GPU/CG dependencies.
reader_path = Path(__file__).resolve().parents[2] / "src/chemfast/conf/misc/sdf_reader.py"
spec = importlib.util.spec_from_file_location("chemfast_cli_sdf_reader", reader_path)
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)
read_chemfast_sdf = reader.read_chemfast_sdf


def _make_sdf(path: Path, *, wrong_count=False, extra_box=False, missing_tag=False) -> None:
    mol = Chem.AddHs(Chem.MolFromSmiles("CC"))
    assert AllChem.EmbedMolecule(mol, randomSeed=17) == 0
    if not missing_tag:
        mol.SetProp("RES_NAMES", " ".join(["A"] * mol.GetNumAtoms()))
    mol.SetProp("RES_NUMS", "0" if wrong_count else " ".join(["0"] * mol.GetNumAtoms()))
    box = "20 20 20 1 0 0 0 0 0" if extra_box else "20 20 20 0 0 0 0 0 0"
    mol.SetProp("BOX_TENSOR", box)
    with Chem.SDWriter(str(path)) as writer:
        writer.SetForceV3000(True)
        writer.write(mol)


def test_cli_subcommands_and_help(capsys):
    parser = build_parser()
    for command in ("prepare_cg", "reconstruct_aa", "density_optimization"):
        with pytest.raises(SystemExit) as exc:
            parser.parse_args([command, "--help"])
        assert exc.value.code == 0
        assert "--" in capsys.readouterr().out


def test_filler_reference_resolves_against_json_not_cwd(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    pdb = source / "molecule.pdb"
    pdb.write_text("END\n", encoding="utf-8")
    raw = {"fillers": [{"name": "X", "file": "molecule.pdb"}]}
    configured = _absolute_filler_paths(raw, source)
    assert configured["fillers"][0]["file"] == str(pdb)
    assert raw["fillers"][0]["file"] == "molecule.pdb"
    with pytest.raises(ValueError, match="filename"):
        _output_filename("../outside.xml", "--xml")


def test_strict_sdf_reader_accepts_reconstruction_metadata(tmp_path):
    path = tmp_path / "in.sdf"
    _make_sdf(path)
    mols, graphs, box = read_chemfast_sdf(path)
    assert len(mols) == len(graphs) == 1
    assert box.tolist() == [20., 20., 20., 0., 0., 0., 0., 0., 0.]
    assert graphs[0].number_of_nodes() == mols[0].GetNumAtoms()
    assert graphs[0].number_of_edges() == mols[0].GetNumBonds()


@pytest.mark.parametrize("options,pattern", [
    ({"wrong_count": True}, "RES_NAMES and RES_NUMS"),
    ({"missing_tag": True}, "missing ChemFAST metadata"),
    ({"extra_box": True}, "orthorhombic"),
])
def test_strict_sdf_reader_rejects_bad_metadata(tmp_path, options, pattern):
    path = tmp_path / "in.sdf"
    _make_sdf(path, **options)
    with pytest.raises(ValueError, match=pattern):
        read_chemfast_sdf(path)


def test_prepare_cg_empty_directory_guard_before_expensive_import(tmp_path, monkeypatch):
    root = tmp_path / "destination"
    (root / "cg").mkdir(parents=True)
    (root / "cg" / "existing.xml").write_text("not empty")
    config = tmp_path / "config.json"
    config.write_text("{}")
    with pytest.raises(SystemExit) as exc:
        main(["prepare_cg", "--json", str(config), "--name", str(root)])
    assert exc.value.code == 2


def test_reconstruct_name_only_uses_exact_fixed_files(tmp_path):
    root = tmp_path / "case"
    (root / "cg").mkdir(parents=True)
    (root / "config.json").write_text("{}")
    with pytest.raises(SystemExit) as exc:
        main(["reconstruct_aa", "--name", str(root)])
    assert exc.value.code == 2


def test_density_cli_preserves_metadata_and_writes_chemfast_sdf(tmp_path, monkeypatch, capsys):
    # Inject only the PyGAMD simulator backend to test the REAL CLI + strict reader + writer
    # without loading the optional CUDA/CG modules into this lightweight test environment.
    conf_root = reader_path.parents[1]
    fake_conf = types.ModuleType("chemfast.conf")
    fake_conf.__path__ = [str(conf_root)]
    fake_misc = types.ModuleType("chemfast.conf.misc")
    fake_misc.__path__ = [str(conf_root / "misc")]
    density_module = types.ModuleType("chemfast.conf.misc._density_optim")
    monkeypatch.setitem(sys.modules, "chemfast.conf", fake_conf)
    monkeypatch.setitem(sys.modules, "chemfast.conf.misc", fake_misc)
    monkeypatch.setitem(sys.modules, "chemfast.conf.misc._density_optim", density_module)

    src, dest = tmp_path / "before.sdf", tmp_path / "after.sdf"
    _make_sdf(src)
    seen = {}

    def fake_run(mols, graphs, cfg, *, target_density, work_dir, gpu, python_bin, **kwargs):
        seen.update(target_density=target_density, work_dir=work_dir, gpu=gpu, python_bin=python_bin)
        coords = mols[0].GetConformer().GetPositions()
        mols[0].GetConformer().SetAtomPosition(0, tuple(coords[0] + [1, 0, 0]))
        return mols, np.array([15., 15., 15.])

    density_module.run_density_optimization = fake_run
    assert main(["density_optimization", "--sdf_in", str(src), "--sdf_out", str(dest),
                 "--density", "1.1", "--gpu_id", "2"]) == 0
    old, _, old_box = read_chemfast_sdf(src)
    new, _, new_box = read_chemfast_sdf(dest)
    assert seen["gpu"] == 2 and seen["target_density"] == 1.1
    assert old_box[0] == 20 and new_box[0] == 15
    assert new[0].GetProp("RES_NAMES") == old[0].GetProp("RES_NAMES")
    assert new[0].GetProp("RES_NUMS") == old[0].GetProp("RES_NUMS")
    assert "Ensure PyGAMD" in capsys.readouterr().out
