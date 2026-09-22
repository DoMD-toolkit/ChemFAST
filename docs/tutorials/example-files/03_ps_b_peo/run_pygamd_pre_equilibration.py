#!/usr/bin/env python3
"""Simple PyGAMD NPT equilibration for a predefined CG topology.

Inputs:
    initial.xml
    cg_parameters.json

Output:
    reaction_final.xml

No polymerization is performed and reaction_path.txt is not modified.
"""

import json
import math
from pathlib import Path

from poetry import cu_gala as gala
from poetry import _options


XML_FILE = Path("initial.xml")
PARAMETER_FILE = Path("cg_parameters.json")
FINAL_FILE = Path("reaction_final.xml")

TEMPERATURE = 1.0
PRESSURE = 1.0
TAU_T = 0.5
TAU_P = 1.0

RELAXATION_DT = 0.0001
NPT_DT = 0.001
NONBONDED_CUTOFF = 2.5
NEIGHBOR_BUFFER = 0.1

# Soft-start relaxation followed by NPT equilibration.
NVE_SIGMA_SCALES = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2)
NVE_STEPS_PER_STAGE = 10000
NPT_STEPS = 100000
DUMP_PERIOD = 10000
LOG_PERIOD = 1000
NVE_LIMIT = 0.01


with PARAMETER_FILE.open("r", encoding="utf-8") as handle:
    PARAMETERS = json.load(handle)


def set_lj_parameters(force, sigma_scale=1.0):
    terms = PARAMETERS["nonbonded"]
    types = sorted(terms)
    for i, left in enumerate(types):
        p_left = terms[left]["params"]
        for right in types[i:]:
            p_right = terms[right]["params"]
            epsilon = math.sqrt(float(p_left["epsilon"]) * float(p_right["epsilon"]))
            sigma = sigma_scale * 0.5 * (float(p_left["sigma"]) + float(p_right["sigma"]))
            force.setParams(left, right, epsilon, sigma, 1.0)
    force.setEnergy_shift()


def add_bonded_forces(app, all_info):
    all_info.addBondTypeByPairs()
    all_info.addAngleTypeByPairs()

    bond_terms = []
    angle_terms = []

    for name, term in PARAMETERS["bonded"].items():
        interaction = str(term.get("itype", "")).upper()
        atom_types = term.get("ff_atom_types") or term.get("types")
        aliases = [str(term.get("name", name))]

        if atom_types:
            atom_types = tuple(map(str, atom_types))
            aliases += ["-".join(atom_types), "-".join(reversed(atom_types))]
        aliases = list(dict.fromkeys(aliases))

        if interaction == "BOND":
            for alias in aliases:
                all_info.addBondType(alias)
            bond_terms.append((aliases, term["params"]))
        elif interaction == "ANGLE":
            for alias in aliases:
                all_info.addAngleType(alias)
            angle_terms.append((aliases, term["params"]))

    if bond_terms:
        bond_force = gala.BondForceHarmonic(all_info)
        for aliases, params in bond_terms:
            for alias in aliases:
                bond_force.setParams(alias, float(params["k"]), float(params["r0"]))
        app.add(bond_force)

    if angle_terms:
        angle_force = gala.AngleForceHarmonic(all_info)
        for aliases, params in angle_terms:
            for alias in aliases:
                angle_force.setParams(alias, float(params["k"]), float(params["r0"]))
        app.add(angle_force)


def configure_xml_dump(output):
    output.setOutputPosition(True)
    output.setOutputMass(True)
    output.setOutputType(True)
    output.setOutputImage(True)
    output.setOutputBond(True)
    output.setOutputAngle(True)
    output.setOutputVelocity(True)
    output.setOutputInit(True)
    output.setOutputCris(True)


def main():
    build_method = gala.XMLReader(str(XML_FILE))
    perform_config = gala.PerformConfig(_options.gpu)
    all_info = gala.AllInfo(build_method, perform_config)

    app = gala.Application(all_info, RELAXATION_DT)

    neighbor_list = gala.NeighborList(
        all_info,
        NONBONDED_CUTOFF,
        NEIGHBOR_BUFFER,
    )
    neighbor_list.addExclusionsFromBonds()
    neighbor_list.addExclusionsFromAngles()

    lj_force = gala.LJForce(all_info, neighbor_list, NONBONDED_CUTOFF)
    set_lj_parameters(lj_force)
    app.add(lj_force)
    add_bonded_forces(app, all_info)

    group_all = gala.ParticleSet(all_info, "all")
    compute_all = gala.ComputeInfo(all_info, group_all)

    sorter = gala.Sort(all_info)
    sorter.setPeriod(10000)
    app.add(sorter)

    zero_momentum = gala.ZeroMomentum(all_info)
    zero_momentum.setPeriod(10000)
    app.add(zero_momentum)

    # Soft-start NVE relaxation.
    nve = gala.NVE(all_info, group_all)
    nve.setLimit(NVE_LIMIT)
    app.add(nve)

    for sigma_scale in NVE_SIGMA_SCALES:
        set_lj_parameters(lj_force, sigma_scale)
        app.run(NVE_STEPS_PER_STAGE)

    app.remove(nve)
    set_lj_parameters(lj_force, 1.2)

    # NPT equilibration.
    app.setDt(NPT_DT)
    npt = gala.NPTMTK(
        all_info,
        group_all,
        compute_all,
        compute_all,
        TEMPERATURE,
        PRESSURE,
        TAU_T,
        TAU_P,
    )
    app.add(npt)

    trajectory = gala.XMLDump(all_info, "cg_equil")
    trajectory.setPeriod(DUMP_PERIOD)
    configure_xml_dump(trajectory)
    app.add(trajectory)

    log = gala.DumpInfo(all_info, compute_all, "cg_equil.log")
    log.dumpBoxSize()
    log.dumpPressTensor()
    log.setPeriod(LOG_PERIOD)
    app.add(log)

    final_prefix = Path("reaction_final_tmp")
    final_dump = gala.XMLDump(all_info, str(final_prefix))
    final_dump.setPeriod(NPT_STEPS)
    configure_xml_dump(final_dump)
    app.add(final_dump)

    app.run(NPT_STEPS + 1)

    final_xml = max(
        Path(".").glob(f"{final_prefix.name}*.xml"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    final_xml.replace(FINAL_FILE)


if __name__ == "__main__":
    main()
