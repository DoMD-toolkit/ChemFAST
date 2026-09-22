from chemfast.conf.domd_topo.reactor import Reactor


def topology_builder(
    reactant_config,
    reaction_template,
    filler_config,
    cg_graph,
    reaction_list,
    fast_sanitize_p=False,
):
    """Builds an all-atom topology from a parsed coarse-grained molecular graph.

    Args:
        reactant_config (dict): Parsed reactant configuration containing RDKit molecules.
        reaction_template (dict): Reaction SMARTS patterns and topology rules.
        filler_config (dict): Parsed rigid-molecule data containing centered positions and RDKit molecules.
        cg_graph (networkx.Graph): Parsed coarse-grained molecular graph.
        reaction_list (list[tuple]): Ordered reactions assigned to this CG molecule by the parser.

    Returns:
        tuple:
            - Chem.Mol: Reconstructed all-atom molecule with explicit hydrogens.
            - networkx.Graph: All-atom molecular graph.
    """
    if cg_graph is None:
        raise ValueError("cg_graph must be provided for topology building.")
    reactor = Reactor(
        reactant_config,
        reaction_template,
        filler_config,
        fast_sanitize_p=fast_sanitize_p,
    )


    # reactor1 = Reactor(
    #     reactant_config,
    #     reaction_template,
    #     rigid_config,
    #     fast_sanitize_p=fast_sanitize_p,
    # )

    ret = reactor.process(cg_graph, reaction_list)
    # ret1 = reactor1.process_old(cg_graph, reaction_list)
    # print('_____________', reactor.check(ret[0], ret1[0], ret[1], ret1[1]))

    return ret
