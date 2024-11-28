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
NOTEBOOKS_load = [
    "https://tomography.stfc.ac.uk/how-tos/ZeissDataReader.ipynb",
    "https://tomography.stfc.ac.uk/how-tos/NikonDataReader.ipynb"
]

NOTEBOOKS_geometry = [
    "https://github.com/TomographicImaging/CIL-Demos/raw/main/demos/1_Introduction/00_CIL_geometry.ipynb"
]

NOTEBOOKS_advanced = [
    "https://github.com/TomographicImaging/CIL-User-Showcase/raw/main/003_1D_integral_inverse_problem/deriv2_cgls.ipynb",
    "https://github.com/TomographicImaging/CIL-Demos/raw/main/misc/callback_demonstration.ipynb"
]

SOURCE = Path(__file__).parent / "source" # sphinx documentation dir
NBDIR = "demos" # notebook subdir to create
(SOURCE / NBDIR).mkdir(parents=True, exist_ok=True)

# download the notebooks
def download_notebooks(urls):
    notebooks = []
    with tqdm(urls, unit="ipynb") as nb_urls:
        for url in nb_urls :
            notebook = Path(urlparse(url).path)
            print(notebook)
            nb_urls.set_description(notebook.stem)
            with urlopen(url) as response:
                (SOURCE / NBDIR / notebook.name).write_bytes(response.read())
            notebooks.append(f"    {NBDIR}/{notebook.stem}")
    return "\n".join(notebooks)

notebooks_load = download_notebooks(NOTEBOOKS_load)
notebooks_geometry = download_notebooks(NOTEBOOKS_geometry)
notebooks_advanced = download_notebooks(NOTEBOOKS_advanced)

# load template
tmp = Template((SOURCE / '..' / 'demos-template.rst').read_text())
# write to demos.rst
(SOURCE / 'demos.rst').write_text(tmp.safe_substitute(
    notebooks_load=notebooks_load,
    notebooks_geometry=notebooks_geometry,
    notebooks_advanced=notebooks_advanced
))
