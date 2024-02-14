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

# sphinx config; vis:
# - http://www.sphinx-doc.org/en/master/config
# - https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/layout.html

import sys
from os import getenv
from pathlib import Path

try:
    from cil import version
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve() / 'Wrappers' / 'Python'))
    from cil import version

# Project information
project = 'CIL'
copyright = '2017-2024'
author = 'CCPi developers'
version = version.version
release = version
_baseurl = f'/{getenv("GITHUB_REPOSITORY", "").split("/", 1)[-1]}/'.replace("//", "/")

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
    'sphinxcontrib.bibtex',
    'nbsphinx',
    'sphinx_gallery.load_style',
    'sphinx_copybutton',
]

templates_path = ['_templates']
source_suffix = ['.rst', '.md']
master_doc = 'index'
language = 'en'
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# HTML config
html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    "switcher": {
        "json_url": f"{_baseurl}versions.json",
        "version_match": version
    },
    "favicons": [{"rel": "icon", "sizes": "32x32", "href": "https://ccpi.ac.uk/wp-content/uploads/2022/11/cropped-CCPi_Logo_Icon_Only-32x32.png"}],
    "logo": {
        "image_light": "https://ccpi.ac.uk/wp-content/uploads/2022/11/CIL-logo-RGB.svg",
        "image_dark": "https://ccpi.ac.uk/wp-content/uploads/2022/11/CIL-logo-RGB-reversed.svg",
        "link": "/",
        "alt_text": "CIL - Home",
    },
    "show_version_warning_banner": True,
    "header_links_before_dropdown": 9,
    "navbar_persistent": [], # no search icon in top right
    "footer_start": ["copyright"], "footer_end": [], # clean footers
    "use_edit_page_button": True, # edit on GitHub link
    "primary_sidebar_end": [],
    "secondary_sidebar_items": [], # remove right sidebar
}
html_sidebars = {
   '**': ['search-field', 'version-switcher', 'globaltoc', 'edit-this-page'],
   'using/windows': ['search-field', 'version-switcher', 'windowssidebar', 'edit-this-page'],
}
html_show_sourcelink = False
html_context = {
    "github_user": "TomographicImaging", "github_repo": "CIL",
    "github_version": "master",
    "doc_path": "docs/source",
}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
#html_static_path = ['docsstatic']
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
