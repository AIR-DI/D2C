# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


import os
import sys
sys.path.insert(0, os.path.abspath('../'))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'D2C'
copyright = '2023, AIR-DREAM'
author = 'AIR-DREAM'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
master_doc = 'index'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'sphinx.ext.viewcode',
	'sphinx.ext.inheritance_diagram',
	'sphinx.ext.napoleon',
]

# Generate autosummary pages. Output should be set with: `:toctree: pythonapi/`
autosummary_generate = ['Python-API.rst']


templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_last_updated_fmt = '%b %d, %Y'
html_domain_indices = True
html_logo = '_static/images/d2c-logo.png'  # 指定logo图片的路径
html_theme_options = {'logo_only': False}  # 设置只显示logo而不显示项目名称
html_css_files = ['css/style.css']  # 指定自定义样式表的路径


# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = False
