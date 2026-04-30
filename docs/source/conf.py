# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Add the package source directory to the Python path.
import os
import sys
import tomllib

sys.path.insert(0, os.path.abspath('../../src'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Spinguin'
copyright = '2026, Joni Eronen, Perttu Hilla'
author = 'Joni Eronen, Perttu Hilla'

# Determine whether the documentation is being built in GitHub Actions.
gh_actions = os.environ.get("GITHUB_ACTIONS") == "true"

# Determine whether the current build originates from a branch.
if gh_actions and os.environ.get("GITHUB_REF", "").startswith("refs/heads/"):
    branch_build = True
else:
    branch_build = False

# Use the "latest" label for branch builds.
if branch_build:
    release = "latest"

# Otherwise, read the release version from pyproject.toml.
else:
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    toml_dir = os.path.join(root_dir, "pyproject.toml")
    with open(toml_dir, "rb") as toml_file:
        toml = tomllib.load(toml_file)
    release = str(toml["project"]["version"])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode'
]
exclude_patterns = []
add_module_names = False
autodoc_member_order = "groupwise"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_theme_options = {
    "switcher": {
        "json_url": "https://nmroulu.github.io/Spinguin/version_switcher.json",
        "version_match": release
    },
    "logo": {
        "text": "Spinguin documentation",
        "image_light": "_static/spinguin_logo.png",
        "image_dark": "_static/spinguin_logo.png",
    }
}
html_sidebars = {
    "**": [
        "navbar-logo.html",
        "version-switcher.html",
        "icon-links.html",
        "search-button-field.html",
        "sbt-sidebar-nav.html"
        ]
}
html_static_path = ['_static']