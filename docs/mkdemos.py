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
NOTEBOOKS_load_links = [
    "https://tomography.stfc.ac.uk/how-tos/ZeissDataReader.ipynb",
    "https://tomography.stfc.ac.uk/how-tos/NikonDataReader.ipynb"
]

NOTEBOOKS_geometry_links = [
    "https://github.com/TomographicImaging/CIL-Demos/raw/main/demos/1_Introduction/00_CIL_geometry.ipynb"
]

NOTEBOOKS_advanced_links = [
    "https://github.com/TomographicImaging/CIL-User-Showcase/raw/main/003_1D_integral_inverse_problem/deriv2_cgls.ipynb",
    "https://github.com/TomographicImaging/CIL-Demos/raw/main/misc/callback_demonstration.ipynb"
]

SOURCE = Path(__file__).parent / "source" # sphinx documentation dir
NBDIR = "demos" # notebook subdir to create
(SOURCE / NBDIR).mkdir(parents=True, exist_ok=True)

# download the notebooks
def download_notebooks(urls):
    '''
    Downloads Jupyter notebooks from the a list of URLs and saves them to the 
    'source/demos' directory, then generates a list formatted for the documentation. 
    '''
    notebooks = []
    with tqdm(urls, unit="ipynb") as nb_urls:
        for url in nb_urls :
            notebook = Path(urlparse(url).path)
            nb_urls.set_description(notebook.stem)
            dest_path = SOURCE / NBDIR / notebook.name
            try:
                with urlopen(url) as response:
                    dest_path.write_bytes(response.read())
            
            except Exception as e:
                print(f"Warning: Could not download {url}. Error: {e}")
                if not dest_path.exists():
                    print(f"  No existing file found for {notebook.name}, skipping.")
                    continue  # Skip adding to the list if no file exists

                notebooks.append(f"    {NBDIR}/{notebook.stem}")
    return "\n".join(notebooks)

notebooks_load = download_notebooks(NOTEBOOKS_load_links)
notebooks_geometry = download_notebooks(NOTEBOOKS_geometry_links)
notebooks_advanced = download_notebooks(NOTEBOOKS_advanced_links)

# load template
tmp = Template((SOURCE / '..' / 'demos-template.rst').read_text())
# write to demos.rst
(SOURCE / 'demos.rst').write_text(tmp.safe_substitute(
    notebooks_load=notebooks_load,
    notebooks_geometry=notebooks_geometry,
    notebooks_advanced=notebooks_advanced
))
