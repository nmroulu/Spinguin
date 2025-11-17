# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Add path to the package
import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Spinguin'
copyright = '2025, Joni Eronen, Perttu Hilla'
author = 'Joni Eronen, Perttu Hilla'
release = 'latest'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode'
]
exclude_patterns = []
add_module_names = False

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