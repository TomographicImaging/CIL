# Copyright 2019 United Kingdom Research and Innovation
# Copyright 2019 The University of Manchester
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Authors:
# CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt

# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import re

sys.path.insert(0, os.path.abspath('../Wrappers/Python/'))

from cil import version

# Project information
project = 'CIL'
copyright = '2017-2024'
author = 'CCPi developers'
version = version.version
release = version

# min Sphinx version
# needs_sphinx = '1.0'

extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinxcontrib.bibtex'
]

templates_path = ['_templates']
source_suffix = ['.rst', '.md']
master_doc = 'index'
language = 'en'
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []
pygments_style = None  # syntax highlighting

# HTML config
html_theme = 'pydata_sphinx_theme'
# default: ['localtoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html']
html_sidebars = {
   '**': ['globaltoc.html', 'sourcelink.html', 'searchbox.html'],
   'using/windows': ['windowssidebar.html', 'searchbox.html'],
}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['docsstatic']
htmlhelp_basename = 'CILdoc'

# LaTeX config
latex_elements = {
    # 'papersize': 'a4paper',
    # 'pointsize': '10pt',
    # 'preamble': '',
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto, manual, or own class]).
latex_documents = [(master_doc, 'CIL.tex', 'CIL Documentation', author, 'manual')]

# man page config
# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, 'cil', 'CIL Documentation', [author], 1)]

# Texinfo config
# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author, dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'CIL', 'CIL Documentation', author, 'CIL',
     'One line description of project.', 'Miscellaneous'),
]

# Epub config
epub_title = project
# epub_identifier = ''  # unique identifier, ISBN, project homepage, etc
# epub_uid = ''  # unique identification
epub_exclude_files = ['search.html']

# Extension config
autodoc_member_order = 'bysource'
todo_include_todos = True  # iff true, `todo` and `todoList` produce output

# Add bibtex files and style
bibtex_bibfiles = ['refs.bib']
bibtex_encoding = 'latin'
bibtex_reference_style = 'label'
bibtex_default_style = 'plain'
