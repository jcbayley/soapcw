

[![PyPI](https://img.shields.io/pypi/v/soapcw)](https://pypi.org/project/soapcw/)
[![Documentation](https://github.com/jcbayley/soapcw/actions/workflows/docs.yml/badge.svg)](https://jcbayley.github.io/soapcw/)
![tests](https://github.com/jcbayley/soapcw/actions/workflows/test.yml/badge.svg)


<img src="https://raw.githubusercontent.com/jcbayley/soapcw/main/logo/drawing.png" alt="Logo" width="30%"/>

# SOAP

SOAP: Applying the Viterbi algorithm to search for sources
of continuous gravitational waves.

<img src="https://raw.githubusercontent.com/jcbayley/soapcw/main/src/soapcw_pipeline/images/vitmap_ex.png" alt="Vitmap"/>

SOAP is primarily developed to search for continuous sources of
gravitational waves, however, has a more general application to search
for and long duration weak signal.

This package also includes tools to load in standard short Fourier transforms (SFTs) and prepare them for usage with the core SOAP search.

# Installation

Install the latest stable release from PyPI:

```bash
uv pip install soapcw
```

Or, for development from a clone of the repository, use [uv](https://docs.astral.sh/uv/) to create an environment and install the project (with dependencies pinned from `uv.lock`):

```bash
git clone https://github.com/jcbayley/soapcw
cd soapcw
uv sync
```

Then run commands inside the environment with `uv run`, e.g. `uv run soapcw-run-soap-astro --help`.

See the [installation docs](https://jcbayley.github.io/soapcw/installation.html) for more detail.


* Free software: MIT license
* Documentation: https://jcbayley.github.io/soapcw/

* old LIGO hosted package (!!!!!no longer maintained!!!!): https://git.ligo.org/joseph.bayley/soapcw/

# Publications

* Methods paper: https://journals.aps.org/prd/abstract/10.1103/PhysRevD.100.023006
* CNN followup paper: https://journals.aps.org/prd/abstract/10.1103/PhysRevD.102.083024
* Parameter estimation paper: https://journals.aps.org/prd/abstract/10.1103/PhysRevD.106.083022


# Features
#


# TODO

* robustly include three detectors

# Credits

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
