# FeatherPy

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]

[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

[![GitHub Discussion][github-discussions-badge]][github-discussions-link]

<!-- SPHINX-START -->

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/AlecThomson/FeatherPy/workflows/CI/badge.svg
[actions-link]:             https://github.com/AlecThomson/FeatherPy/actions
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/AlecThomson/FeatherPy/discussions
[pypi-link]:                https://pypi.org/project/FeatherPy/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/FeatherPy
[pypi-version]:             https://img.shields.io/pypi/v/FeatherPy
[rtd-badge]:                https://readthedocs.org/projects/FeatherPy/badge/?version=latest
[rtd-link]:                 https://FeatherPy.readthedocs.io/en/latest/?badge=latest

<!-- prettier-ignore-end -->

Combine images in the Fourier domain.

## Installation

This package is written in pure Python, with all dependencies (including the
Python version) specified in the `pyproject.toml`.

PyPI (stable):

```
pip install featherpy
```

GitHub (latest):

```
pip install git+https://github.com/AlecThomson/FeatherPy
```

## Usage

Command line:

```
$ featherpy -h
usage: featherpy [-h] [-c FEATHER_CENTRE] [-s FEATHER_SIGMA] [-u OUTER_UV_CUT] [-lu LOW_RES_UNIT] [-hu HIGH_RES_UNIT] [-p] [-o] low_res_file high_res_file output_file frequency

Feather two FITS files

positional arguments:
  low_res_file          Low resolution FITS file
  high_res_file         High resolution FITS file
  output_file           Output feathered FITS file
  frequency             Frequency of the data in Hz

options:
  -h, --help            show this help message and exit
  -c FEATHER_CENTRE, --feather-centre FEATHER_CENTRE
                        UV centre of the feathering function in meters (default: 0)
  -s FEATHER_SIGMA, --feather-sigma FEATHER_SIGMA
                        UV width of the feathering function in meters (default: 1)
  -u OUTER_UV_CUT, --outer-uv-cut OUTER_UV_CUT
                        Outer UV cut in meters (default: None)
  -lu LOW_RES_UNIT, --low-res-unit LOW_RES_UNIT
                        Unit of the low resolution data. Will try to read from BUNIT if not provided (default: None)
  -hu HIGH_RES_UNIT, --high-res-unit HIGH_RES_UNIT
                        Unit of the high resolution data. Will try to read from BUNIT if not provided (default: None)
  -p, --do-feather-plot
                        Make a plot of the feathering (default: False)
  -o, --overwrite       Overwrite the output file if it exists (default: False)
```

Python API:

```python
from featherpy.core import feather_from_fits
```

See the [API docs][rtd-link] for further detail.

## Algorithm

Here I follow the technique outlined by
[Weiss et al. 2001](https://ui.adsabs.harvard.edu/abs/2001A%26A...365..571W/abstract).
For further reading the see the
[CASA Feather documentation and references](https://casadocs.readthedocs.io/en/stable/notebooks/image_combination.html#Feather-&-CASAfeather).

Two input images are required a 'low resolution' image and a 'high resolution'
image. I assume the 'low resolution' is sensitive to all spatial scales, as is
typical for a single-dish radio image, and that the 'high resolution' image is
missing short spacing scales.

Weighting between the two images is done in the $uv$-plane in unites of metres.
I assume that:

- There is sufficient $uv$ overlap between the two images
- The images are measured at the same frequency and with the same bandwidth
- The images can be converted to units of Jy/sr

I weight between in the images in the $uv$-plane using a logistic (sigmoid)
function. You must specify both the mid-point of the overlap in $uv$ space as
well as the 1$\sigma$ overlap (which I approximately convert to the right
parameter of the logictic function).

Steps:

- Regrid low resolution image to match high resolution
- Convert low and high resolution images to Jy/sr
- FFT the low-resolution image
- Optionally apply an outer $uv$ cut to the low resolution FFT
- Deconvolve the low-resolution FFT
- FFT the high-resolution image
- Calculate weights in the $uv$-plane
- Combine image FFTs as a weighted sum
- FFT back to the image domain
- Convert feathered image to Jy/beam
