"""
This module is an example of a barebones writer plugin for napari.

It implements the Writer specification.
see: https://napari.org/stable/plugins/guides.html?#writers

Replace code below according to your needs.
"""
from __future__ import annotations

from tifffile import imwrite, imread
import zarr

import numpy as np
import xarray as xr
import dask.array as da
from dask import config as dask_config

from typing import TYPE_CHECKING, Any, List, Sequence, Tuple, Union

from napari_stitcher import _spatial_image_utils

if TYPE_CHECKING:
    DataType = Union[Any, Sequence[Any]]
    FullLayerData = Tuple[DataType, dict, str]


def write_single_image(path: str, data: Any, meta: dict) -> List[str]:
    """Writes a single image layer"""

    # implement your writer logic here ...

    # return path to any file(s) that were successfully written
    return [path]


def write_multiple(path: str, data: List[FullLayerData]) -> List[str]:
    """
    Writes zarr backed dask arrays containing fused images.
    Ignores transform_keys.
    FullLayerData: 3-tuple with (data, meta, layer_type)
    """

    # implement your writer logic here ...

    if not path.endswith('.tif'):
        raise ValueError('Only .tif file saving is supported.')
    
    xims = [d[0][0] for d in data]

    spacings = [_spatial_image_utils.get_spacing_from_xim(xim, asarray=True) for xim in xims]
    origins = [_spatial_image_utils.get_origin_from_xim(xim, asarray=True) for xim in xims]
    shapes = [_spatial_image_utils.get_shape_from_xim(xim, asarray=True) for xim in xims]

    for ixim in range(len(data)):
        if not np.allclose(spacings[ixim], spacings[0]) or \
           not np.allclose(origins[ixim], origins[0]) or \
           not np.allclose(shapes[ixim], shapes[0]):
            raise ValueError('Image saving: Data of all layers must occupy the same space.')

    xim_to_write = xr.concat([xim for xim in xims], dim='c')

    save_xim_as_tif(path, xim_to_write)

    # return path to any file(s) that were successfully written
    return [path]


def save_xim_as_tif(path, xim):

    spatial_dims = _spatial_image_utils.get_spatial_dims_from_xim(xim)
    spacing = _spatial_image_utils.get_spacing_from_xim(xim, asarray=True)

    xim = xim.transpose(*tuple(['t', 'c'] + spatial_dims))

    channels = [ch if isinstance(ch, str) else str(ch)
                for ch in xim.coords['c'].values]

    xim = xim.squeeze(drop=True)

    # imagej needs Z to come before C
    if 'z' in xim.dims and 'c' in xim.dims:
        axes = list(xim.dims)
        zpos = axes.index('z')
        cpos = axes.index('c')
        axes[zpos] = 'c'
        axes[cpos] = 'z'
        xim = xim.transpose(*tuple([ax for ax in axes]))

    axes = ''.join(xim.dims).upper()

    imwrite(
        path,
        shape=xim.shape,
        dtype=xim.dtype,
        imagej=True,
        resolution=tuple([1. / s for s in spacing[-2:]]),
        metadata={
            'axes': axes,
            # 'unit': 'um',
            'Labels': channels,
            }
    )

    store = imread(path, mode='r+', aszarr=True)
    z = zarr.open(store, mode='r+')

    # writing with tifffile is not thread safe,
    # so we need to disable dask's multithreading
    with dask_config.set(scheduler='single-threaded'):
        da.store(xim.data, z)

    store.close()

    return


if __name__ == "__main__":

    from napari_stitcher import _reader

    layer_tuples = _reader.read_mosaic_czi("/Users/malbert/software/napari-stitcher/image-datasets/mosaic_test.czi")

    write_multiple('/Users/malbert/Desktop/test.tif', [layer_tuples[0]])
