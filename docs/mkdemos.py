#!/usr/bin/env python
"""
1. Downloads demo notebooks to `source/demos/*.ipynb`
2. Creates `source/demos.rst` from `demos-template.rst`
"""
from pathlib import Path
from string import Template
from urllib.parse import urlparse
from urllib.request import urlopen

from tqdm import tqdm

# URLS of the notebooks to render
NOTEBOOKS = (
    "https://github.com/TomographicImaging/CIL-Demos/raw/main/demos/1_Introduction/00_CIL_geometry.ipynb",
    "https://github.com/TomographicImaging/CIL-User-Showcase/raw/main/003_1D_integral_inverse_problem/deriv2_cgls.ipynb"
)
SOURCE = Path(__file__).parent / "source" # sphinx documentation dir
NBDIR = "demos" # notebook subdir to create
(SOURCE / NBDIR).mkdir(parents=True, exist_ok=True)

# download the notebooks
notebooks = []
with tqdm(NOTEBOOKS, unit="ipynb") as urls:
    for url in urls:
        notebook = Path(urlparse(url).path)
        urls.set_description(notebook.stem)
        with urlopen(url) as response:
            (SOURCE / NBDIR / notebook.name).write_bytes(response.read())
        notebooks.append(f"    {NBDIR}/{notebook.stem}")

# load template
tmp = Template((SOURCE / '..' / 'demos-template.rst').read_text())
# write to demos.rst
(SOURCE / 'demos.rst').write_text(tmp.safe_substitute(notebooks="\n".join(notebooks)))
