from copy import deepcopy
from typing import Any


def _expand_reactions_configs(reactions_configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate reaction rules: each rule has one ordered flat reactant list.

    For multiple type pairs, declare separate named rules sharing the same SMARTS.
    The input names are preserved for CG bonds and ReactionPath records.
    """
    validated = []
    for reaction_index, source in enumerate(reactions_configs):
        if not isinstance(source, dict):
            raise ValueError(f"reactions[{reaction_index}]: expected an object")
        reaction = deepcopy(source)
        reactants = reaction.get("reactants")
        if (not isinstance(reactants, list) or not reactants
                or not all(isinstance(item, str) and item for item in reactants)):
            raise ValueError(
                f"reactions[{reaction_index}].reactants: "
                "expected a non-empty array of non-empty type names, "
                'for example ["A", "B1"]; nested arrays are not supported'
            )
        validated.append(reaction)
    return validated
