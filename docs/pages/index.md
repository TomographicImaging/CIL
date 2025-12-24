---
# https://github.com/Phlow/feeling-responsive/blob/gh-pages/pages/pages-root-folder/index.md
layout: frontpage
header:
  image_fullwidth: https://ccpi.ac.uk/wp-content/uploads/2022/08/front-page-1125x800.png # ideally 600x80 pixels
  title: '<img loading="lazy" src="https://ccpi.ac.uk/wp-content/uploads/2022/11/CIL-logo-RGB.svg" alt="CIL - Core Imaging Library">'
widgets:
- title: CIL Documentation
  url: '/nightly/'
  image: https://ccpi.ac.uk/wp-content/uploads/2022/11/CIL-logo-RGB.svg
  text: >
    Complete API reference, user guides, and tutorials for CIL.
    <br/>
    <br/>
    The documentation is updated regularly and built nightly from the latest development code.
- title: Citing CIL
  url: '/publications/#citing-cil'
  image: https://ccpi.ac.uk/wp-content/uploads/2022/11/CIL-logo-RGB.svg
  text: >
    Please cite CIL if you use it in your research.
    <br/>
    <br/>
    We provide three main papers describing CIL's framework, multichannel capabilities, and algorithmic developments, along with BibTeX and RIS export options.
- title: Research Using CIL
  url: '/publications/#research-using-cil'
  image: https://ccpi.ac.uk/wp-content/uploads/2022/10/RSTA_cover.png
  text: >
    View publications that have used CIL in their research.
    <br/>
    <br/>
    Journal papers, conference proceedings, and PhD theses organized by year.
- title: Try CIL in Binder
  url: 'https://mybinder.org/v2/gh/TomographicImaging/CIL-Demos/HEAD?urlpath=lab/tree/binder%2Findex.ipynb'
  image: https://ccpi.ac.uk/wp-content/uploads/2022/04/walnut_training.png
  text: >
    Run CIL examples <em>without any local installation</em> using <a href="https://mybinder.org">Binder</a>.
    <br/>
    <br/>
    Access a <a href="https://github.com/TomographicImaging/CIL-Demos">large collection</a> of Jupyter Notebooks covering basic usage to advanced reconstructions with iterative methods.
callforaction:
  url: https://ccpi.ac.uk/training
  text: Training
  style: alert
---

## A versatile python framework for tomographic imaging

CIL is an **open-source** mainly Python framework for tomographic imaging for *cone* and *parallel beam* geometries. It comes with tools for **loading**, **preprocessing**, **reconstructing** and **visualising** tomographic data.

CIL provides **optimised** standard methods such as *Filtered Back Projection* and *FDK* and an extensive **modular optimisation framework** for prototyping reconstruction methods including *sparsity* and *total variation* regularisation, useful when conventional filtered backprojection reconstruction do not lead to satisfactory results, as in highly noisy, incomplete, non-standard or multichannel data arising for example in *dynamic*, *spectral* and *in situ* tomography.

CIL is open-source software released under the [Apache v2.0 licence](http://www.apache.org/licenses/LICENSE-2.0.html).
