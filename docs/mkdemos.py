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
    "https://raw.githubusercontent.com/TomographicImaging/CIL-Demos/refs/heads/main/demos/4_Deep_Dives/01_callbacks.ipynb"
]

NOTEBOOKS_usershowcase_links = [
    "https://github.com/TomographicImaging/CIL-User-Showcase/raw/main/001_multibang_regularisation/Multibang_Hackathon2023.ipynb",
    "https://github.com/TomographicImaging/CIL-User-Showcase/raw/main/002_Deblurring/Showcase_of_the_algorithms_for_deblurring_and_denoising.ipynb",
    "https://github.com/TomographicImaging/CIL-User-Showcase/raw/main/003_1D_integral_inverse_problem/deriv2_cgls.ipynb",
    "https://github.com/TomographicImaging/CIL-User-Showcase/raw/main/004_STEMPO-DynamicCT/stempo2d_v2_1.ipynb",
    "https://github.com/TomographicImaging/CIL-User-Showcase/raw/main/005_dynamic_mr_recon/dynamic_mr_recon.ipynb",
    "https://github.com/TomographicImaging/CIL-User-Showcase/raw/main/006_gVXR2CIL/CBCT-acquisition-with-noise-and-reconstruction.ipynb",
    "https://github.com/TomographicImaging/CIL-User-Showcase/raw/main/007_Hyperspectral_regularisation/Hyperspectral_regularisation.ipynb",
    "https://github.com/TomographicImaging/CIL-User-Showcase/raw/main/008_KL_divergence_and_weighted_least_squares/LS_WLS_KL_TotalVariation.ipynb",
    "https://github.com/TomographicImaging/CIL-User-Showcase/raw/main/009_offset_CT_apple/Apple_offset_reconstruction.ipynb",
    "https://github.com/TomographicImaging/CIL-User-Showcase/raw/main/010_reconstruct_BrukerSkyscan_data/bruker2cil.ipynb",
    "https://github.com/TomographicImaging/CIL-User-Showcase/raw/main/011_phasecontrast_exciscope/recon_phasecontrast_exciscope.ipynb",
    "https://github.com/TomographicImaging/CIL-User-Showcase/raw/main/012_wavelet_sparsity_controlled_regularization/controlledWaveletSparsity_2d.ipynb",
    "https://github.com/TomographicImaging/CIL-User-Showcase/raw/main/012_wavelet_sparsity_controlled_regularization/controlledWaveletSparsity_3d.ipynb",
    "https://github.com/TomographicImaging/CIL-User-Showcase/raw/main/013_anisotropic_regularization_for_FILD_measurements/013_Anisotropic_regularisation_FILD_measurements.ipynb",
    "https://github.com/TomographicImaging/CIL-User-Showcase/raw/main/014_GVXR_simulation_and_CIL_CPU_reconstruction/Simulation_and_CPU_reconstruction.ipynb",
    "https://github.com/TomographicImaging/CIL-User-Showcase/raw/main/015_Memory_Profiling_LSQR_CGLS/memory_profiling_sandstone.ipynb",
    "https://github.com/TomographicImaging/CIL-User-Showcase/raw/main/016_cil_torch_fista_pnp/fista_with_denoiser.ipynb"
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
            with urlopen(url) as response:
                (SOURCE / NBDIR / notebook.name).write_bytes(response.read())
            notebooks.append(f"    {NBDIR}/{notebook.stem}")
    return "\n".join(notebooks)

notebooks_load = download_notebooks(NOTEBOOKS_load_links)
notebooks_geometry = download_notebooks(NOTEBOOKS_geometry_links)
notebooks_advanced = download_notebooks(NOTEBOOKS_advanced_links)
notebooks_usershowcase = download_notebooks(NOTEBOOKS_usershowcase_links)

# load template
tmp = Template((SOURCE / '..' / 'demos-template.rst').read_text())
# write to demos.rst
(SOURCE / 'demos.rst').write_text(tmp.safe_substitute(
    notebooks_load=notebooks_load,
    notebooks_geometry=notebooks_geometry,
    notebooks_advanced=notebooks_advanced,
    notebooks_usershowcase=notebooks_usershowcase
))

tmp = Template((SOURCE / '..' / 'usershowcase-template.rst').read_text())
# write to usershowcase.rst
(SOURCE / 'usershowcase.rst').write_text(tmp.safe_substitute(
    notebooks_usershowcase=notebooks_usershowcase
))
