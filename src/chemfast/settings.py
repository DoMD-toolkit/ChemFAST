# The threshold for the number of atoms in a molecule to determine whether to use the fast sanitization method or not.
FAST_SANITZE_THRESHOLD       = 1500

# CHUNK_BOUNDARY_EXPAND_RADIUS is the number of graph connectivities to expand the chunk boundary when splitting box
# into chunks.
CHUNK_BOUNDARY_EXPAND_RADIUS = 2

# The threshold for the number of atoms in a molecule to determine whether to use the small molecule embedding method
# (embed as a whole) or not.
IS_SMALL_THRESHOLD           = 300

# The threshold for the number of atoms in a molecule to determine whether to use `GMX` template-matching method for
# force field assignment or not.
THRESHOLD_H                  = 1000

# JSON KEYWORDS
DSL_VERSION                  = "v1"
DSL_CONF                     = "domd_react_dsl"
REACTANT_CONF                = "reactants"
FILLER_CONF                  = "fillers"
FILLER_IDX_CONF              = "filler_idx"
REACTION_CONF                = "reactions"
CG_TOPOLOGY_FILE_CONF        = "cg_topology_file"
REACTION_PATH_FILE_CONF      = "reaction_path_file"
CG_REACTANTS_CONF            = "reactants"
NAME_CONF                    = "name"

COUNT_CONF                   = "N"
SMILES_CONF                  = "smiles"
MAX_VALENCE_CONF             = "max_valence"
ACTIVATE_CONF                = "activate"

FILE_CONF                    = "file"
MAPPING_CONF                 = "mappings"
TYPE_CONF                    = "type"
ATOM_IDX_CONF                = "atom_idx"
SMARTS_CONF                  = "smarts"
CG_ID_CONF                   = "cg_id"

KIND_CONF                    = "kind"
GENERAL_KIND                 = "general"
RADICAL_KIND                 = "radical"
INTRINSIC_PROBABILITY_CONF   = "intrinsic_probability"
ACTIVATION_CONF              = "activation"
FROM_CONF                    = "from"
TO_CONF                      = "to"
TYPE_CHANGES_CONF            = "type_changes"
NODE_CONF                    = "node"
