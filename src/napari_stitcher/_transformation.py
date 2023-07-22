import numpy as np
import xarray as xr
import transformations as tfs

from napari_stitcher import _spatial_image_utils

import dask.array as da
from dask_image.ndinterp import affine_transform as dask_image_affine_transform

import spatial_image as si


def transform_xim(
        xim,
        p=None,
        output_shape=None,
        output_spacing=None,
        output_origin=None,
        output_chunksize=64,
        order=1,
        ):
    
    ndim = _spatial_image_utils.get_ndim_from_xim(xim)

    if p is None:
        p = np.eye(ndim + 1)

    if output_shape is None:
        output_shape = _spatial_image_utils.get_shape_from_xim(xim, asarray=True)

    if output_spacing is None:
        output_spacing = _spatial_image_utils.get_spacing_from_xim(xim, asarray=True)

    if output_origin is None:
        output_origin = _spatial_image_utils.get_origin_from_xim(xim, asarray=True)

    ndim = _spatial_image_utils.get_ndim_from_xim(xim)
    spatial_dims = _spatial_image_utils.get_spatial_dims_from_xim(xim)
    # p = np.array(params[ixim])
    matrix = p[:ndim, :ndim]
    offset = p[:ndim, ndim]

    # spacing matrices
    Sx = np.diag(output_spacing)
    Sy = np.diag(_spatial_image_utils.get_spacing_from_xim(xim, asarray=True))

    matrix_prime = np.dot(np.linalg.inv(Sy), np.dot(matrix, Sx))
    offset_prime = np.dot(np.linalg.inv(Sy),
        offset - _spatial_image_utils.get_origin_from_xim(xim, asarray=True) +
        np.dot(matrix, output_origin))

    out_da = dask_image_affine_transform(
        xim.data,
        matrix=matrix_prime,
        offset=offset_prime,
        order=order,
        output_shape=tuple(output_shape),
        output_chunks=tuple([output_chunksize for _ in output_shape]),
        mode='constant',
        cval=0.,
        )
    
    sim = si.to_spatial_image(
        out_da,
        dims=xim.dims,
        scale={dim: output_spacing[idim] for idim, dim in enumerate(spatial_dims)},
        translation={dim: output_origin[idim] for idim, dim in enumerate(spatial_dims)},
        )
    
    return sim