---
# https://github.com/Phlow/feeling-responsive/blob/gh-pages/pages/pages-root-folder/index.md
layout: frontpage
header:
  image_fullwidth: https://ccpi.ac.uk/wp-content/uploads/2022/08/front-page-1125x800.png # ideally 600x80 pixels
  title: '<img loading="lazy" src="https://ccpi.ac.uk/wp-content/uploads/2022/11/CIL-logo-RGB.svg" alt="CIL - Core Imaging Library">'
widgets:
- title: Examples
  url: 'https://mybinder.org/v2/gh/TomographicImaging/CIL-Demos/HEAD?urlpath=lab/tree/binder%2Findex.ipynb'
  image: https://ccpi.ac.uk/wp-content/uploads/2022/04/walnut_training.png
  text: >
    We have a repository with a <a href="https://github.com/TomographicImaging/CIL-Demos">large collection</a> of Jupyter Notebooks which cover a wide range of topics, from basic usage to advanced reconstructions with iterative methods.
    <br/>
    Some examples <em>without any local installation</em> are provided in <a href="https://mybinder.org">Binder</a>.
    Please click the button below to try them immediately in your browser.
- title: Documentation
  url: '/nightly/'
  image: https://ccpi.ac.uk/wp-content/uploads/2022/11/CIL-logo-RGB.svg
  text: >
    CIL has a live documentation which gets updated regularly and built nightly.
    We suggest to download and read the open access articles below, which provide very detailed information about CIL structure and usage.
- title: Papers
  url: 'https://doi.org/10.1098/rsta.2020.0192'
  image: https://ccpi.ac.uk/wp-content/uploads/2022/10/RSTA_cover.png
  text: >
    Jørgensen JS et al. 2021 <a href="https://doi.org/10.1098/rsta.2020.0192">Core Imaging Library – Part I: a versatile python framework for tomographic imaging</a>. Phil. Trans. R. Soc. A 20200192.
    <br/>
    Code: <a href="https://github.com/TomographicImaging/Paper-2021-RSTA-CIL-Part-I">Paper-2021-RSTA-CIL-Part-I</a>.
    <br/>
    <br/>
    Papoutsellis E et al. 2021 <a href="https://doi.org/10.1098/rsta.2020.0193">Core Imaging Library – Part II: multichannel reconstruction for dynamic and spectral tomography</a>. Phil. Trans. R. Soc. A 20200193.
    <br/>
    Code: <a href="https://github.com/TomographicImaging/Paper-2021-RSTA-CIL-Part-II">Paper-2021-RSTA-CIL-Part-II</a>.
callforaction:
  url: https://ccpi.ac.uk/training
  text: Training
  style: alert
---

## A versatile python framework for tomographic imaging

CIL is an **open-source** mainly Python framework for tomographic imaging for *cone* and *parallel beam* geometries. It comes with tools for **loading**, **preprocessing**, **reconstructing** and **visualising** tomographic data.

CIL provides **optimised** standard methods such as *Filtered Back Projection* and *FDK* and an extensive **modular optimisation framework** for prototyping reconstruction methods including *sparsity* and *total variation* regularisation, useful when conventional filtered backprojection reconstruction do not lead to satisfactory results, as in highly noisy, incomplete, non-standard or multichannel data arising for example in *dynamic*, *spectral* and *in situ* tomography.

[![source code](https://img.shields.io/badge/source%20code-GitHub-green)](https://github.com/TomographicImaging/CIL) [![licence](https://img.shields.io/github/license/TomographicImaging/CIL?label=licence)](https://www.apache.org/licenses/LICENSE-2.0.html) [![zenodo](https://img.shields.io/badge/zenodo-10.5281%2Fzenodo.4746198-blue)](https://doi.org/10.5281/zenodo.4746198)
