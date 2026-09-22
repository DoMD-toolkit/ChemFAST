class Config(object):
    def __init__(self,
                 reactant_config,
                 reaction_template,
                 filler_config,
                 box_tensor,
                 cg_sys,
                 reaction_list,
                 cg_graphs=None,
                 components=None,
                 raw_config=None):
        self.reactant_config       = reactant_config
        self.reaction_template     = reaction_template
        self.filler_config         = filler_config
        self.cg_sys                = cg_sys
        self.cg_graphs             = cg_graphs
        self.reaction_list         = reaction_list
        self.box_tensor            = box_tensor
        self.components            = [] if components is None else components
        self.raw_config            = raw_config

    def __str__(self):
        return (f"Config(reactant_config={self.reactant_config}, "
                f"reaction_template={self.reaction_template}, "
                f"filler_config={self.filler_config}, "
                f"box_tensor={self.box_tensor}, "
                f"cg_sys={self.cg_sys}, "
                f"reaction_list={self.reaction_list}, "
                f"cg_graphs={self.cg_graphs}, "
                f"components={self.components})")