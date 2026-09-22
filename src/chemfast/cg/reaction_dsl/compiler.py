from __future__ import annotations

import json
from numbers import Real
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdChemReactions

from chemfast import settings
from chemfast.misc._utils import _expand_reactions_configs
from chemfast.cg.reaction_dsl.model import (
    ActivationTransfer,
    CompiledModel,
    FillerArm,
    FillerType,
    ReactantType,
    ReactionPath,
    ReactionOperator,
    ReactionRule,
    SystemState,
    TypeChange,
)


class DSLValidationError(ValueError):
    pass


def _fail(path: str, message: str) -> None:
    raise DSLValidationError(f"{path}: {message}")


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(path, "expected an object")
    return value


def _array(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(path, "expected an array")
    return value


def _require_keys(value: dict[str, Any], *, required: set[str], path: str) -> None:
    missing = required - value.keys()
    if missing:
        _fail(path, f"missing key(s): {', '.join(sorted(missing))}")


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(path, "expected a non-empty string")
    return value


def _integer(value: Any, path: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(path, f"expected an integer >= {minimum}")
    return value


def _probability(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not 0.0 <= float(value) <= 1.0:
        _fail(path, "expected a number in [0, 1]")
    return float(value)


def _compile_reactants(raw: Any) -> dict[str, ReactantType]:
    configs = _array(raw, settings.REACTANT_CONF)
    if not configs:
        _fail(settings.REACTANT_CONF, "at least one reactant type is required")

    result: dict[str, ReactantType] = {}
    for index, value in enumerate(configs):
        path = f"{settings.REACTANT_CONF}[{index}]"
        config = _object(value, path)
        _require_keys(
            config,
            required={
                settings.NAME_CONF,
                settings.MAX_VALENCE_CONF,
            },
            path=path,
        )

        name = _string(config[settings.NAME_CONF], f"{path}.{settings.NAME_CONF}")
        if name in result:
            _fail(f"{path}.{settings.NAME_CONF}", f"duplicate name {name!r}")

        has_smiles = settings.SMILES_CONF in config
        has_smarts = settings.SMARTS_CONF in config
        if has_smiles == has_smarts:
            _fail(
                path,
                f"exactly one of {settings.SMILES_CONF!r} or " f"{settings.SMARTS_CONF!r} is required",
            )
        if has_smarts and settings.COUNT_CONF in config and config[settings.COUNT_CONF] != 0:
            _fail(
                f"{path}.{settings.COUNT_CONF}",
                f"is not allowed with {settings.SMARTS_CONF!r}",
            )

        structure_key = settings.SMILES_CONF if has_smiles else settings.SMARTS_CONF
        structure = _string(config[structure_key], f"{path}.{structure_key}")
        parser = Chem.MolFromSmiles if has_smiles else Chem.MolFromSmarts
        mol = parser(structure)
        if mol is None:
            _fail(
                f"{path}.{structure_key}",
                f"RDKit could not parse the {structure_key.upper()}",
            )

        count = _integer(
            config.get(settings.COUNT_CONF, 0),
            f"{path}.{settings.COUNT_CONF}",
        )
        max_valence = _integer(
            config[settings.MAX_VALENCE_CONF],
            f"{path}.{settings.MAX_VALENCE_CONF}",
            1,
        )
        activate = _integer(
            config.get(settings.ACTIVATE_CONF, 0),
            f"{path}.{settings.ACTIVATE_CONF}",
        )

        result[name] = ReactantType(
            name=name,
            smiles=structure if has_smiles else None,
            smarts=structure if has_smarts else None,
            N=count,
            max_valence=max_valence,
            activate=activate,
            mol=mol,
        )
    return result


def _compile_fillers(raw: Any, reactants: dict[str, ReactantType]) -> dict[str, FillerType]:
    configs = _array(raw, settings.FILLER_CONF)
    result: dict[str, FillerType] = {}

    for index, value in enumerate(configs):
        path = f"{settings.FILLER_CONF}[{index}]"
        config = _object(value, path)
        _require_keys(
            config,
            required={
                settings.NAME_CONF,
                settings.COUNT_CONF,
                settings.FILE_CONF,
                settings.MAPPING_CONF,
            },
            path=path,
        )

        name = _string(config[settings.NAME_CONF], f"{path}.{settings.NAME_CONF}")
        if name in result:
            _fail(f"{path}.{settings.NAME_CONF}", f"duplicate name {name!r}")

        count = _integer(
            config[settings.COUNT_CONF],
            f"{path}.{settings.COUNT_CONF}",
        )
        file = _string(
            config[settings.FILE_CONF],
            f"{path}.{settings.FILE_CONF}",
        )
        arm_configs = _array(
            config[settings.MAPPING_CONF],
            f"{path}.{settings.MAPPING_CONF}",
        )
        if not arm_configs:
            _fail(
                f"{path}.{settings.MAPPING_CONF}",
                "at least one reactive arm is required",
            )

        arms: list[FillerArm] = []
        cg_ids: set[int] = set()
        for arm_index, arm_value in enumerate(arm_configs):
            arm_path = f"{path}.{settings.MAPPING_CONF}[{arm_index}]"
            arm = _object(arm_value, arm_path)
            _require_keys(
                arm,
                required={
                    settings.CG_ID_CONF,
                    settings.TYPE_CONF,
                    settings.ATOM_IDX_CONF,
                },
                path=arm_path,
            )

            cg_id = _integer(
                arm[settings.CG_ID_CONF],
                f"{arm_path}.{settings.CG_ID_CONF}",
            )
            if cg_id in cg_ids:
                _fail(
                    f"{arm_path}.{settings.CG_ID_CONF}",
                    f"duplicate cg_id {cg_id!r}",
                )
            cg_ids.add(cg_id)

            type_name = _string(
                arm[settings.TYPE_CONF],
                f"{arm_path}.{settings.TYPE_CONF}",
            )
            if type_name not in reactants:
                _fail(
                    f"{arm_path}.{settings.TYPE_CONF}",
                    f"unknown reactant type {type_name!r}",
                )

            atom_idx_raw = arm[settings.ATOM_IDX_CONF]
            if not isinstance(atom_idx_raw, list) or not atom_idx_raw:
                _fail(
                    f"{arm_path}.{settings.ATOM_IDX_CONF}",
                    "expected a non-empty integer array",
                )
            atom_idx = tuple(_integer(i, f"{arm_path}.{settings.ATOM_IDX_CONF}") for i in atom_idx_raw)
            if len(set(atom_idx)) != len(atom_idx):
                _fail(
                    f"{arm_path}.{settings.ATOM_IDX_CONF}",
                    "atom indices must be unique",
                )

            arms.append(FillerArm(cg_id, type_name, atom_idx))

        result[name] = FillerType(name, count, file, tuple(arms))
    return result


def _validate_activation_counts(
    reactants: dict[str, ReactantType],
    fillers: dict[str, FillerType],
) -> None:
    totals = {name: reactant.N for name, reactant in reactants.items()}
    for filler in fillers.values():
        for arm in filler.arms:
            totals[arm.type_name] += filler.N

    for index, reactant in enumerate(reactants.values()):
        if reactant.activate > totals[reactant.name]:
            _fail(
                f"{settings.REACTANT_CONF}[{index}].{settings.ACTIVATE_CONF}",
                f"cannot exceed generated node count {totals[reactant.name]}",
            )

def _extract_multibody_operator(
    rxn: rdChemReactions.ChemicalReaction,
    arity: int,
    path: str,
    type_changes: tuple[TypeChange, ...] = (),
) -> ReactionOperator:
    if rxn.GetNumProductTemplates() == 0:
        _fail(
            f"{path}.{settings.SMARTS_CONF}",
            "reaction has no product",
        )

    # Collect unique nonzero reactant maps and their reaction slots.
    atom_origin: dict[int, int] = {}
    for slot in range(arity):
        template = rxn.GetReactantTemplate(slot)
        for atom in template.GetAtoms():
            map_num = atom.GetAtomMapNum()
            if map_num == 0:
                continue
            if map_num in atom_origin:
                _fail(
                    f"{path}.{settings.SMARTS_CONF}",
                    f"reactant atom map {map_num} is not unique",
                )
            atom_origin[map_num] = slot

    # Every nonzero map must occur exactly once across all products.
    product_maps: set[int] = set()
    for product_index in range(rxn.GetNumProductTemplates()):
        current_product = rxn.GetProductTemplate(product_index)
        for atom in current_product.GetAtoms():
            map_num = atom.GetAtomMapNum()
            if map_num == 0:
                continue
            if map_num in product_maps:
                _fail(
                    f"{path}.{settings.SMARTS_CONF}",
                    f"product atom map {map_num} is not unique across products",
                )
            product_maps.add(map_num)

    reactant_maps = set(atom_origin)
    if reactant_maps != product_maps:
        missing = sorted(reactant_maps - product_maps)
        unexpected = sorted(product_maps - reactant_maps)
        details = []
        if missing:
            details.append(f"missing from products: {missing}")
        if unexpected:
            details.append(f"absent from reactants: {unexpected}")
        _fail(
            f"{path}.{settings.SMARTS_CONF}",
            "reactant and product atom maps differ; " + "; ".join(details),
        )

    product = rxn.GetProductTemplate(0)

    # Assign mapped product atoms directly to their reactant slots.
    atom_slots: dict[int, int] = {}
    present_slots: set[int] = set()
    for atom in product.GetAtoms():
        map_num = atom.GetAtomMapNum()
        if map_num == 0:
            continue
        slot = atom_origin[map_num]
        atom_slots[atom.GetIdx()] = slot
        present_slots.add(slot)

    if present_slots != set(range(arity)):
        missing = sorted(set(range(arity)) - present_slots)
        _fail(
            f"{path}.{settings.SMARTS_CONF}",
            f"product 0 does not contain reactant slot(s) {missing}",
        )

    # Group connected unmapped atoms and inherit the unique neighboring slot.
    unmapped_nodes = {
        atom.GetIdx()
        for atom in product.GetAtoms()
        if atom.GetAtomMapNum() == 0
    }
    unmapped_graph = nx.Graph()
    unmapped_graph.add_nodes_from(unmapped_nodes)

    for bond in product.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if begin in unmapped_nodes and end in unmapped_nodes:
            unmapped_graph.add_edge(begin, end)

    for component in nx.connected_components(unmapped_graph):
        neighboring_slots: set[int] = set()
        for atom_index in component:
            atom = product.GetAtomWithIdx(atom_index)
            for neighbor in atom.GetNeighbors():
                neighbor_index = neighbor.GetIdx()
                if neighbor_index in atom_slots:
                    neighboring_slots.add(atom_slots[neighbor_index])

        if not neighboring_slots:
            _fail(
                f"{path}.{settings.SMARTS_CONF}",
                f"unmapped product atom region {sorted(component)} "
                "is not connected to any mapped reactant slot",
            )
        if len(neighboring_slots) > 1:
            _fail(
                f"{path}.{settings.SMARTS_CONF}",
                f"unmapped product atom region {sorted(component)} "
                f"is connected to multiple reactant slots {sorted(neighboring_slots)}",
            )

        slot = next(iter(neighboring_slots))
        for atom_index in component:
            atom_slots[atom_index] = slot

    # Project product bonds onto reaction slots.
    edges: set[tuple[int, int]] = set()
    for bond in product.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        i, j = atom_slots[begin], atom_slots[end]
        if i != j:
            edges.add((min(i, j), max(i, j)))

    slot_graph = nx.Graph()
    slot_graph.add_nodes_from(range(arity))
    slot_graph.add_edges_from(edges)
    if not edges or not nx.is_connected(slot_graph):
        _fail(
            f"{path}.{settings.SMARTS_CONF}",
            "product 0 must connect all reactant slots",
        )

    degree_delta = tuple(slot_graph.degree(i) for i in range(arity))
    return ReactionOperator(
        tuple(sorted(edges)),
        degree_delta,
        type_changes,
    )


def _extract_multibody_operator_(
    rxn: rdChemReactions.ChemicalReaction, arity: int, path: str, type_changes: tuple[TypeChange, ...] = ()
) -> ReactionOperator:
    if rxn.GetNumProductTemplates() == 0:
        _fail(f"{path}.{settings.SMARTS_CONF}", "reaction has no product")

    atom_origin: dict[int, int] = {}
    for slot in range(arity):
        template = rxn.GetReactantTemplate(slot)
        for atom in template.GetAtoms():
            map_num = atom.GetAtomMapNum()
            if map_num == 0:
                continue
            if map_num in atom_origin:
                _fail(
                    f"{path}.{settings.SMARTS_CONF}",
                    f"reactant atom map {map_num} is not unique",
                )
            atom_origin[map_num] = slot

    product = rxn.GetProductTemplate(0)
    product_maps: set[int] = set()
    present_slots: set[int] = set()
    for atom in product.GetAtoms():
        map_num = atom.GetAtomMapNum()
        if map_num == 0:
            _fail(
                f"{path}.{settings.SMARTS_CONF}",
                "every atom in product 0 must have an atom map",
            )
        if map_num not in atom_origin:
            _fail(
                f"{path}.{settings.SMARTS_CONF}",
                f"product atom map {map_num} has no reactant origin",
            )
        if map_num in product_maps:
            _fail(
                f"{path}.{settings.SMARTS_CONF}",
                f"product atom map {map_num} is not unique",
            )
        product_maps.add(map_num)
        present_slots.add(atom_origin[map_num])

    if present_slots != set(range(arity)):
        missing = sorted(set(range(arity)) - present_slots)
        _fail(
            f"{path}.{settings.SMARTS_CONF}",
            f"product 0 does not contain reactant slot(s) {missing}",
        )

    edges: set[tuple[int, int]] = set()
    for bond in product.GetBonds():
        begin_map = bond.GetBeginAtom().GetAtomMapNum()
        end_map = bond.GetEndAtom().GetAtomMapNum()
        i, j = atom_origin[begin_map], atom_origin[end_map]
        if i != j:
            edges.add((min(i, j), max(i, j)))

    slot_graph = nx.Graph()
    slot_graph.add_nodes_from(range(arity))
    slot_graph.add_edges_from(edges)
    if not edges or not nx.is_connected(slot_graph):
        _fail(
            f"{path}.{settings.SMARTS_CONF}",
            "product 0 must connect all reactant slots",
        )

    degree_delta = tuple(slot_graph.degree(i) for i in range(arity))
    return ReactionOperator(tuple(sorted(edges)), degree_delta, type_changes)


def _slot_symmetries(
    reactants: tuple[str, ...],
    template_smarts: tuple[str, ...],
    operator: ReactionOperator,
    activation: ActivationTransfer | None,
) -> tuple[tuple[int, ...], ...]:
    graph = nx.Graph()
    type_change_targets = {change.node: change.to for change in operator.type_changes}
    for slot, (type_name, template) in enumerate(zip(reactants, template_smarts)):
        active_role = "none"
        if activation is not None:
            if activation.requires_active(slot):
                active_role = "source"
            if slot == activation.target:
                active_role += "+target"
        graph.add_node(
            slot,
            type=type_name,
            template=template,
            active_role=active_role,
            type_change=type_change_targets.get(slot),
        )
    graph.add_edges_from(operator.edges)

    def same_role(left: dict[str, Any], right: dict[str, Any]) -> bool:
        return left == right

    matcher = nx.algorithms.isomorphism.GraphMatcher(graph, graph, node_match=same_role)
    permutations = {tuple(mapping[i] for i in range(len(reactants))) for mapping in matcher.isomorphisms_iter()}
    return tuple(sorted(permutations))


def _template_signature(template: Chem.Mol) -> str:
    """Canonical query signature with atom-map labels removed."""

    normalized = Chem.Mol(template)
    for atom in normalized.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmarts(normalized)


def _matches_template(reactant: ReactantType, template: Chem.Mol) -> bool:
    if reactant.smiles is not None:
        return Chem.AddHs(reactant.mol).HasSubstructMatch(template)
    return reactant.mol.HasSubstructMatch(template)


#def _compile_reactions(raw: Any, reactants: dict[str, ReactantType]) -> dict[str, ReactionRule]:
#    configs = _array(raw, settings.REACTION_CONF)
#    if not configs:
#        _fail(settings.REACTION_CONF, "at least one reaction is required")
#    result: dict[str, ReactionRule] = {}
def _compile_reactions(raw: Any, reactants: dict[str, ReactantType]) -> dict[str, ReactionRule]:
    configs = _array(raw, settings.REACTION_CONF)
    if not configs:
        _fail(settings.REACTION_CONF, "at least one reaction is required")

    configs = _expand_reactions_configs(configs)

    result: dict[str, ReactionRule] = {}
    for index, value in enumerate(configs):
        path = f"{settings.REACTION_CONF}[{index}]"
        config = _object(value, path)
        _require_keys(
            config,
            required={
                settings.NAME_CONF,
                settings.REACTANT_CONF,
                settings.INTRINSIC_PROBABILITY_CONF,
            },
            path=path,
        )
        name = _string(config[settings.NAME_CONF], f"{path}.{settings.NAME_CONF}")
        if name in result:
            _fail(f"{path}.{settings.NAME_CONF}", f"duplicate name {name!r}")

        kind = config.get(settings.KIND_CONF, settings.GENERAL_KIND)
        if kind not in {settings.GENERAL_KIND, settings.RADICAL_KIND}:
            _fail(
                f"{path}.{settings.KIND_CONF}",
                f"expected {settings.GENERAL_KIND!r} or {settings.RADICAL_KIND!r}",
            )
        if kind == settings.RADICAL_KIND:
            _require_keys(
                config,
                required={settings.ACTIVATION_CONF},
                path=path,
            )

        roles_raw = config[settings.REACTANT_CONF]
        if not isinstance(roles_raw, list) or len(roles_raw) < 1:
            _fail(
                f"{path}.{settings.REACTANT_CONF}",
                "expected a non-empty array",
            )
        roles = tuple(_string(item, f"{path}.{settings.REACTANT_CONF}") for item in roles_raw)
        for slot, type_name in enumerate(roles):
            if type_name not in reactants:
                _fail(
                    f"{path}.{settings.REACTANT_CONF}[{slot}]",
                    f"unknown reactant type {type_name!r}",
                )

        unary = len(roles) == 1
        if not unary:
            _require_keys(
                config,
                required={settings.SMARTS_CONF},
                path=path,
            )

        probability = _probability(
            config[settings.INTRINSIC_PROBABILITY_CONF],
            f"{path}.{settings.INTRINSIC_PROBABILITY_CONF}",
        )
        if not unary:
            smarts = _string(
                config[settings.SMARTS_CONF],
                f"{path}.{settings.SMARTS_CONF}",
            )
            try:
                rxn = rdChemReactions.ReactionFromSmarts(smarts)
            except Exception as exc:
                raise DSLValidationError(f"{path}.{settings.SMARTS_CONF}: " "RDKit could not parse the reaction SMARTS") from exc
            if rxn is None:
                _fail(
                    f"{path}.{settings.SMARTS_CONF}",
                    "RDKit could not parse the reaction SMARTS",
                )
            _, errors = rxn.Validate()
            if errors:
                _fail(
                    f"{path}.{settings.SMARTS_CONF}",
                    f"RDKit reported {errors} reaction error(s)",
                )
            if rxn.GetNumReactantTemplates() != len(roles):
                _fail(
                    f"{path}.{settings.SMARTS_CONF}",
                    f"contains {rxn.GetNumReactantTemplates()} reactant templates, " f"expected {len(roles)}",
                )

            template_smarts = tuple(_template_signature(rxn.GetReactantTemplate(slot)) for slot in range(len(roles)))
            for slot, type_name in enumerate(roles):
                if not _matches_template(
                    reactants[type_name],
                    rxn.GetReactantTemplate(slot),
                ):
                    _fail(
                        f"{path}.{settings.SMARTS_CONF}",
                        f"reactant template {slot} does not match type {type_name!r}",
                    )

        activation = None
        if kind == settings.RADICAL_KIND:
            activation_path = f"{path}.{settings.ACTIVATION_CONF}"
            activation_raw = _object(
                config[settings.ACTIVATION_CONF],
                activation_path,
            )
            _require_keys(
                activation_raw,
                required={settings.FROM_CONF, settings.TO_CONF},
                path=activation_path,
            )
            source_path = f"{activation_path}.{settings.FROM_CONF}"
            source_raw = activation_raw[settings.FROM_CONF]
            if source_raw is None:
                sources = ()
            elif type(source_raw) is int:
                sources = (_integer(source_raw, source_path),)
            elif isinstance(source_raw, list):
                sources = tuple(
                    _integer(source, f"{source_path}[{index}]")
                    for index, source in enumerate(source_raw)
                )
                if len(sources) != 2 or len(set(sources)) != 2:
                    _fail(
                        source_path,
                        "double radical termination requires two distinct source slots",
                    )
                sources = tuple(sorted(sources))
            else:
                _fail(
                    source_path,
                    "expected an integer, a two-slot array, or null",
                )
            target_raw = activation_raw[settings.TO_CONF]
            target = (
                None
                if target_raw is None
                else _integer(
                    target_raw,
                    f"{activation_path}.{settings.TO_CONF}",
                )
            )
            for source in sources:
                if source >= len(roles):
                    _fail(
                        source_path,
                        "slot index is out of range",
                    )
            if target is not None and target >= len(roles):
                _fail(
                    f"{activation_path}.{settings.TO_CONF}",
                    "slot index is out of range",
                )
            if not sources and target is None:
                _fail(
                    activation_path,
                    f"{settings.FROM_CONF} and {settings.TO_CONF} cannot both be null",
                )
            if len(sources) == 2 and (
                len(roles) != 2
                or sources != (0, 1)
                or target is not None
            ):
                _fail(
                    activation_path,
                    "double radical termination requires a binary rule with "
                    "from=[0, 1] and to=null",
                )
            if unary and (sources, target) not in {
                ((), 0),
                ((0,), 0),
                ((0,), None),
            }:
                _fail(
                    activation_path,
                    "a unary radical reaction requires (from, to) to be " "(null, 0), (0, 0), or (0, null)",
                )
            activation = ActivationTransfer(sources, target)

        type_changes_raw = config.get(settings.TYPE_CHANGES_CONF, [])
        if not isinstance(type_changes_raw, list):
            _fail(
                f"{path}.{settings.TYPE_CHANGES_CONF}",
                "expected an array",
            )
        if kind == settings.RADICAL_KIND and type_changes_raw:
            _fail(
                f"{path}.{settings.TYPE_CHANGES_CONF}",
                "type changes are only supported by general reactions",
            )
        type_changes_list: list[TypeChange] = []
        changed_nodes: set[int] = set()
        for change_index, change_value in enumerate(type_changes_raw):
            change_path = f"{path}.{settings.TYPE_CHANGES_CONF}[{change_index}]"
            change = _object(change_value, change_path)
            _require_keys(
                change,
                required={settings.NODE_CONF, settings.TO_CONF},
                path=change_path,
            )

            node = _integer(
                change[settings.NODE_CONF],
                f"{change_path}.{settings.NODE_CONF}",
            )
            if node >= len(roles):
                _fail(
                    f"{change_path}.{settings.NODE_CONF}",
                    "reaction slot index is out of range",
                )
            if node in changed_nodes:
                _fail(
                    f"{change_path}.{settings.NODE_CONF}",
                    "each reaction slot may change type only once",
                )
            changed_nodes.add(node)

            target_type = _string(
                change[settings.TO_CONF],
                f"{change_path}.{settings.TO_CONF}",
            )
            if target_type not in reactants:
                _fail(
                    f"{change_path}.{settings.TO_CONF}",
                    f"unknown reactant type {target_type!r}",
                )
            type_changes_list.append(TypeChange(node, target_type, reactants[target_type].max_valence))
        type_changes = tuple(type_changes_list)

        if unary:
            operator = ReactionOperator((), (0,), type_changes)
            symmetries = ((0,),)
        else:
            operator = _extract_multibody_operator(
                rxn,
                len(roles),
                path,
                type_changes,
            )
            symmetries = _slot_symmetries(
                roles,
                template_smarts,
                operator,
                activation,
            )

        for slot, delta in enumerate(operator.degree_delta):
            max_valence = reactants[roles[slot]].max_valence
            if delta > max_valence:
                _fail(
                    f"{path}.{settings.SMARTS_CONF}",
                    f"slot {slot} needs valence {delta}, but type {roles[slot]!r} allows {max_valence}",
                )
        for change in operator.type_changes:
            delta = operator.degree_delta[change.node]
            if delta > change.target_max_valence:
                _fail(
                    f"{path}.{settings.TYPE_CHANGES_CONF}",
                    f"slot {change.node} needs valence {delta}, but target type {change.to!r} allows {change.target_max_valence}",
                )

        result[name] = ReactionRule(
            name=name,
            kind=kind,
            reactants=roles,
            intrinsic_probability=probability,
            operator=operator,
            activation=activation,
            slot_symmetries=symmetries,
        )
    return result


def compile_dict(raw: dict[str, Any]) -> CompiledModel:
    root = _object(raw, "root")
    _require_keys(
        root,
        required={
            settings.DSL_CONF,
            settings.REACTANT_CONF,
            settings.REACTION_CONF,
        },
        path="root",
    )
    if root[settings.DSL_CONF] != settings.DSL_VERSION:
        _fail(
            settings.DSL_CONF,
            f"expected {settings.DSL_VERSION!r}",
        )

    reactants = _compile_reactants(root[settings.REACTANT_CONF])
    fillers = _compile_fillers(root.get(settings.FILLER_CONF, []), reactants)
    _validate_activation_counts(reactants, fillers)
    reactions = _compile_reactions(root[settings.REACTION_CONF], reactants)

    topology_file = root.get(settings.CG_TOPOLOGY_FILE_CONF)
    if topology_file is not None:
        topology_file = _string(
            topology_file,
            settings.CG_TOPOLOGY_FILE_CONF,
        )
    return CompiledModel(reactants, fillers, reactions, topology_file)


def compile_file(path: str | Path) -> CompiledModel:
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DSLValidationError(f"{path}:{exc.lineno}:{exc.colno}: invalid JSON: {exc.msg}") from exc
    return compile_dict(raw)


def initialize_dict(
    raw: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[SystemState, ReactionPath]:
    model = compile_dict(raw)
    return model.initial_state(rng), ReactionPath()


def initialize_file(
    path: str | Path,
    rng: np.random.Generator,
) -> tuple[SystemState, ReactionPath]:
    model = compile_file(path)
    return model.initial_state(rng), ReactionPath()
