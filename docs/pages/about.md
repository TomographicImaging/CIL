---
layout: page-fullwidth
header: false
title: About the Core Imaging Library (CIL)
teaser: A versatile python framework for tomographic imaging
---

CIL is an open-source mainly Python framework for tomographic imaging for cone and parallel beam geometries. It comes with tools for loading, preprocessing, reconstructing and visualising tomographic data.

CIL is developed and maintained by a user driven community. For the CT users,  CIL provides optimised standard methods such as Filtered Back Projection and FDK and an extensive modular optimisation framework for prototyping reconstruction methods.

CIL also works  with users with more complex CT needs including highly noisy, incomplete, non-standard or multichannel data arising for example in dynamic, spectral and in situ tomography. CIL provides optimisation methods for reconstructions including sparsity and total variation regularisation, useful when conventional filtered backprojection reconstruction do not lead to satisfactory results.

For mathematical optimisation or inverse problems researchers, CIL provides a range of deterministic and stochastic optimisation methods, as well as a range of operators and functions for building complex loss functions and regularisers.

## Software, license and source code

CIL is open-source software released under the [Apache v2.0 license](https://www.apache.org/licenses/LICENSE-2.0.html). The [source code is on github](https://github.com/TomographicImaging/CIL) and the latest version is tested regularly, including with each change in the master repository.

## History and funders

CIL brought together a variety of different imaging libraries and was initially funded as an outcome of the  [Collaborative Computational Project in Tomographic Imaging (CCPi)](https://ccpi.ac.uk). Collaborative Computational Projects (CCPs) are a UK networks of expertise in key computational research fields. THey foster exchange and enable large-scale scientific software development, maintenance and distribution. CCPs are supported by the [Computational Science Centre for Research Communities (CoSeC)](https://www.scd.stfc.ac.uk/Pages/CoSeC.aspx).

Most of the development team are computational scientists at the [Scientific Computing Department](https://www.scd.stfc.ac.uk/Pages/home.aspx) at the [Science and Technology Facilities Council (STFC)](https://www.ukri.org/councils/stfc). STFC supports research in astronomy, physics and space science, and operates world-class research facilities for the UK.   More recently, it's core developers are also part-funded by the [Ada Lovelace Centre](https://www.scd.stfc.ac.uk/Pages/Ada-Lovelace-Centre.aspx),an integrated, cross-disciplinary data intensive science centre, for better exploitation of research carried out at our large scale National Facilities including the Diamond Light Source (DLS), the ISIS Neutron and Muon Facility, the Central Laser Facility (CLF) and the Culham Centre for Fusion Energy (CCFE).

To find out more about the team and developers see [Contacts](./nightly/#contacts).

## References

[1] Jørgensen JS et al. 2021 [Core Imaging Library Part I: a versatile python framework for tomographic imaging](https://doi.org/10.1098/rsta.2020.0192). Phil. Trans. R. Soc. A 20200192. [**Code.**](https://github.com/TomographicImaging/Paper-2021-RSTA-CIL-Part-I) [Pre-print](https://arxiv.org/abs/2102.04560)

[2] Papoutsellis E et al. 2021 [Core Imaging Library - Part II: multichannel reconstruction for dynamic and spectral
tomography](https://doi.org/10.1098/rsta.2020.0193). Phil. Trans. R. Soc. A 20200193. [**Code.**](https://github.com/TomographicImaging/Paper-2021-RSTA-CIL-Part-II) [Pre-print](https://arxiv.org/abs/2102.06126)
