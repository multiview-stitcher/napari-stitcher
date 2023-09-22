"""
This module is an example of a barebones writer plugin for napari.

It implements the Writer specification.
see: https://napari.org/stable/plugins/guides.html?#writers

Replace code below according to your needs.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from typing import TYPE_CHECKING, Any, List, Sequence, Tuple, Union

from ngff_stitcher import spatial_image_utils, io

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

    spacings = [spatial_image_utils.get_spacing_from_xim(xim, asarray=True) for xim in xims]
    origins = [spatial_image_utils.get_origin_from_xim(xim, asarray=True) for xim in xims]
    shapes = [spatial_image_utils.get_shape_from_xim(xim, asarray=True) for xim in xims]

    for ixim in range(len(data)):
        if not np.allclose(spacings[ixim], spacings[0]) or \
           not np.allclose(origins[ixim], origins[0]) or \
           not np.allclose(shapes[ixim], shapes[0]):
            raise ValueError('Image saving: Data of all layers must occupy the same space.')

    xim_to_write = xr.concat([xim for xim in xims], dim='c')

    io.save_xim_as_tif(path, xim_to_write)

    # return path to any file(s) that were successfully written
    return [path]
