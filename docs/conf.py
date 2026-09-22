from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
project = "ChemFAST"
author = "ChemFAST developers"
release = "1.0.0"
extensions = ["myst_nb", "sphinx.ext.autodoc", "sphinx.ext.autosummary", "sphinx.ext.napoleon"]
source_suffix = {".rst": "restructuredtext", ".md": "myst-nb"}
master_doc = "index"
templates_path = ["_templates"]
exclude_patterns = ["_build", "build", "README.md"]
html_theme = "alabaster"
html_title = "ChemFAST · Documentation"
html_static_path = ["_static"]
html_css_files = ["chemfast.css"]
html_theme_options = {"description": "Chemical definitions. Simulation-ready models.", "fixed_sidebar": True,
                      "sidebar_width": "250px", "page_width": "1180px", "show_powered_by": False}
html_theme_options.update({
    "logo": "left_small_logo.png",
    "logo_name": False,
    "description": "",
})
html_sidebars = {
    "**": ["about.html", "searchbox.html", "navigation.html"]
}
html_show_sourcelink = False
html_copy_source = False
html_favicon = "_static/favicon.svg"
autodoc_typehints = "none"
autodoc_member_order = "bysource"
autosummary_generate = True
napoleon_numpy_docstring = True
napoleon_google_docstring = False
nb_execution_mode = "off"
myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

# The curated pages provide explanations; source supplies the exact signatures.
# Do not render obsolete internal prose into the public API reference.
def skip_legacy_prose(app, what, name, obj, options, lines):
    if name.startswith("chemfast."):
        lines.clear()

def setup(app):
    app.connect("autodoc-process-docstring", skip_legacy_prose)

html_show_copyright = False
