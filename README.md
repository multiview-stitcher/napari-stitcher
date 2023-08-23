<!--
[![License BSD-3](https://img.shields.io/pypi/l/napari-stitcher.svg?color=green)](https://github.com/m-albert/napari-stitcher/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-stitcher.svg?color=green)](https://pypi.org/project/napari-stitcher)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-stitcher.svg?color=green)](https://python.org)
[![tests](https://github.com/m-albert/napari-stitcher/workflows/tests/badge.svg)](https://github.com/m-albert/napari-stitcher/actions)
[![codecov](https://codecov.io/gh/m-albert/napari-stitcher/branch/main/graph/badge.svg)](https://codecov.io/gh/m-albert/napari-stitcher)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-stitcher)](https://napari-hub.org/plugins/napari-stitcher)
-->

# napari-stitcher
A toolbox and napari plugin for registering / fusing / stitching multi-view / multi-positioning image datasets in 2-3D. Improves and replaces [MVRegFUS](https://github.com/m-albert/MVRegFus).

WORK IN PROGRESS.

----------------------------------
## Installation

TODO

You can install `napari-stitcher` via [pip]:

    `pip install https://github.com/m-albert/napari-stitcher`

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.


[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/

## Some concepts / background / info:

### Image data model / format

Classes used to represent images:

- spatial-image
  - https://github.com/spatial-image/spatial-image
  - subclasses `xarray.DataArray`, i.e. broadly speaking labeled numpy arrays
  - dask compatible for lazy loading and parallel processing

- multiscale-spatial-image
  - https://github.com/spatial-image/multiscale-spatial-image
  - multiscale representation of spatial-image above
  - compatible with NGFF (github.com/ome/ngff)
  - based on `xarray.datatree`
  - also used by github.com/scverse/spatialdata

Problem: the two classes above, as well as NGFF, do not (yet) support:
  1) affine transformations
  2) different transformations for different timepoints

See https://github.com/ome/ngff/issues/94#issuecomment-1656309977.

Therefore, currently these classes are used with slight modifications. Namely:
- affine transformations for each timepoint are represented as `xarray.DataArray` with dimensions (t, x_in, x_out), typically of shape (N_tps, ndim+1, ndim+1)
- multiscale-spatial-image contains one `xarray.Dataset` per scale. These are collections of `xarray.DataArray` which are called "data variables" and share coordinates. The affine transformation parameters are added to each scale as a new data variable
- this is (hackily) compatible with `multiscale_spatial_image.to_zarr()` and `datatree.open_zarr()`

### Registration

- skimage.registration.phase_correlation
- elastix (`itk-elastix`) will be used for higher transformations
- planned to add beads alignment

### Transformation

- relies on dask_image.ndinterp.affine_transform

### Fusion

- chunkwise
- idea is to have a modular API (partly implemented) to plug in different fusion functions including:
  - blending
  - content-based
  - multi-view deconvolution


## Ideas / things to check out

- https://github.com/carbonplan/ndpyramid
- https://www.napari-hub.org/plugins/affinder

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-stitcher" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

Old screenshot of napari plugin:

![](misc-data/20221223_screenshot.png)