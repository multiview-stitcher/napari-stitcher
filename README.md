<!--
[![License BSD-3](https://img.shields.io/pypi/l/napari-stitcher.svg?color=green)](https://github.com/m-albert/napari-stitcher/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-stitcher.svg?color=green)](https://pypi.org/project/napari-stitcher)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-stitcher.svg?color=green)](https://python.org)
[![tests](https://github.com/m-albert/napari-stitcher/workflows/tests/badge.svg)](https://github.com/m-albert/napari-stitcher/actions)
[![codecov](https://codecov.io/gh/m-albert/napari-stitcher/branch/main/graph/badge.svg)](https://codecov.io/gh/m-albert/napari-stitcher)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-stitcher)](https://napari-hub.org/plugins/napari-stitcher)
-->

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

WORK IN PROGRESS

# napari-stitcher
A toolbox and napari plugin for registering / fusing / stitching multi-view / multi-positioning image datasets in 2-3D. Improves and replaces [MVRegFUS](https://github.com/m-albert/MVRegFus).


----------------------------------
## Installation

You can install `napari-stitcher` via [pip]:

    `pip install https://github.com/m-albert/napari-stitcher`

## Features

### Registration

#### Pairwise registration
- [x] phase correlation
- [ ] elastix (`itk-elastix`) will be used for higher transformations
- [ ] bead alignment
- [ ] phase correlation for rotation + translation

#### Group registration

- [x] registration graph construction
- [x] automatic determination of suitable reference view
- [x] parameter concatenation along graph connectivity paths
- [ ] global optimization of registration parameters from (potentially overdetermined) pairwise transforms
- [ ] drift correction / temporal alignment

### Transformation

- [x] chunked `dask_image.ndinterp.affine_transform`
- [ ] cupy-based transform

### Fusion

- [x] chunkwise
- [ ] modular API to plug in different fusion functions including:
- Supported weights:
  - [x] blending
  - [ ] content-based
- Supported fusion methods:
  - [x] weighted average
  - [ ] multi-view deconvolution

### Data formats
- [x] zarr based intermediate file format for reading and writing, compatible with parallel dask workflows: multiscale-spatial-data
- [x] tif input
- [x] tif writing

### Visualization

#### Napari
- [x] 2D slice view: multiscale rendering
- 3D rendered view:
  - [x] lowest scale
  - [ ] chunked rendering
- [x] colormaps optimized for highlighting differences between overlapping views


### Dimensionality
- [x] 2d
- [x] 3d

### Supported usage modes
- [x] as a library to build custom reconstruction workflows
- [ ] predefined workflows/pipelines adapted to specific setups
- [(x)] napari plugin
- [ ] cluster-enabled workflow

Screenshot of napari plugin (old):

![](misc-data/20221223_screenshot.png)

## Implementation details

### (Image) data structures

#### Affine transformations

- affine transformations associated to an image / view are represented as `xarray.DataArray`s with dimensions (t, x_in, x_out), typically of shape (N_tps, ndim+1, ndim+1)
- one transform per timepoint

#### [spatial-image](https://github.com/spatial-image/spatial-image)
  - subclasses `xarray.DataArray`, i.e. broadly speaking these are numpy/dask arrays with axis labels, coordinates and attributes
  - dask compatible for lazy loading and parallel processing

#### [multiscale-spatial-image](https://github.com/spatial-image/multiscale-spatial-image)
  - `xarray.datatree` containing one `xarray.Dataset` per (hierarchical) spatial scale
  -  these are collections of `xarray.DataArray` which are called "data variables" and share coordinates.
  - each scale contains a `spatial-image`s as a data variable named 'image' 
  - compatible with NGFF (github.com/ome/ngff)
  - can be (de-)serialized to zarr
  - also used by github.com/scverse/spatialdata

#### Coordinate systems

The two image structures above, as well as NGFF (as of 0.4.1), [do not yet support](https://github.com/ome/ngff/issues/94#issuecomment-1656309977):
  1) affine transformations
  2) different transformations for different timepoints

However, affine transformations are important for positioning views relatively to each other. Therefore, `spatial-image` and `multiscale-spatial-image` are used with modifications. Specifically, affine transformation parameters which transform the image into a coordinate system of a given name are attached to both:
- `spatial-image`: as key(name)/value pairs under a 'transform' attribute
- `multiscale-spatial-image`: to each scale as data variables, sharing the 't' coordinate with the associated image data variable. This is compatible with reading and writing `multiscale_spatial_image.to_zarr()` and `datatree.open_zarr()`.

In the code, coordinate systems are referred to as *transform_key* (TODO: find better name, e.g. *coordinate_system*).


### Registration using graphs

#### Overlap graph
An *overlap graph* is computed from the input images (represented as a directed `networkx.DiGraph`) in which the
- nodes represent the views and
- edges represent geometrical overlap.

This graph can be used to conveniently color views for visualization (overlapping views should have different colors, but the total number of colors used shouldn't be too large, i.e. exceed 2-4).

#### Reference view

A suitable reference view can be obtained from the overlap graph by e.g. choosing the view with maximal overlap to other views.

#### Registration graph

A *registration graph* or list of registration pairs (TODO: clarify whether this should be a graph or a list of pairs) is obtained from the overlap graph by e.g. finding shortest overlap-weighted paths between the reference view and all other views.


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