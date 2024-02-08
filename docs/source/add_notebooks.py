from urllib.request import urlopen
from urllib.parse import urlparse
from io import BytesIO
import os
from pathlib import Path

# this file directory
ddir =os.path.dirname(os.path.abspath(__file__))

# create the notebooks directory
Path(f"{ddir}/notebooks").mkdir(parents=True, exist_ok=True)

# URLS of the notebooks to display on the documentation
urls = [
    "https://github.com/TomographicImaging/CIL-Demos/raw/main/demos/1_Introduction/00_CIL_geometry.ipynb",
    "https://github.com/TomographicImaging/CIL-User-Showcase/raw/main/003_1D_integral_inverse_problem/deriv2_cgls.ipynb"
]

# Elements in the example.rst file
example_toc = """
.. toctree::
    :maxdepth: 1
    :caption: Contents:

{}

"""
example_gallery = """
Thumbnails gallery
==================

.. nbgallery::
{}

"""

# download the notebooks and add them to the example.rst file
notebooks = ""
for url in urls:
    # find the file name
    parsed = urlparse(url)
    file_name = parsed.path.split('/')[-1]
    notebook_name = file_name.split(".")[0]
    print (f"Downloading {file_name}")
    try:
        with urlopen(url) as response:
            with BytesIO(response.read()) as bytes, open(os.path.join(ddir, 'notebooks', file_name), 'w') as f:
                f.write(bytes.read().decode('utf-8'))
        notebooks += f"    notebooks/{notebook_name}\n"
        
    except Exception as e:
        print(f"Error downloading {url}: {e}")
    

# open the example.rst file and add the notebook
try:
    with open(os.path.join(ddir, 'example.rst'), 'a') as f:
        f.write(f"{example_toc.format(notebooks)}\n")
        f.write(f"{example_gallery.format(notebooks)}\n")
except Exception as e:
    print(f"Error updating the example.rst file: {e}")

        

