# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "korg.py"
copyright = "2025, korg.py developers"
author = "korg.py developers"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
]


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = []

# Extension Options
# =================

# sphinx.ext.extlinks
# -------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/extlinks.html#module-sphinx.ext.extlinks

# This config is a dictionary of external sites, where the key is used as a
# name of a role and the value is the name of a tuple of strings that serve as
# templates for an external url and a template for the text that gets used.
_GITHUB_BASE = "https://github.com/ajwheeler/grackle"
_SRC_BASE = f"{_GITHUB_BASE}/tree/main"
extlinks = {
    "gh-issue": (_GITHUB_BASE + "/issues/%s", "gh-issue#%s"),
    "gh-pr": (_GITHUB_BASE + "/pull/%s", "gh-pr#%s"),
}
# for example :gh-issue:`2` should render as a hyperlink that says gh-issue#2 and
# links to Issue #2
