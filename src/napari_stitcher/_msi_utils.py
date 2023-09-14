import numpy as np
import xarray as xr

import datatree

from functools import wraps
from pathlib import Path

import spatial_image as si
import multiscale_spatial_image as msi

from napari_stitcher import _spatial_image_utils


def get_store_decorator(store_path, store_overwrite=False):
    """
    Generator of decorators meant for functions that read some file (non lazy?) into a msi.
    Decorators stores resulting msi in a zarr and returns a new msi loaded from the store. 
    """
    if store_path is None:
        return lambda func: func

    def store_decorator(func):
        """
        store_decorator takes care of caching msi on disk
        """

        @wraps(func)
        def wrapper_decorator(*args, **kwargs):

            if not store_path.exists() or store_overwrite:
                msi = func(*args, **kwargs)
                msi.to_zarr(Path(store_path))
            
            return multiscale_spatial_image_from_zarr(Path(store_path))
        
        return wrapper_decorator

    return store_decorator


def get_transform_from_msim(msim, transform_key=None):
    """
    Get transform from msim. If transform_key is None, get the transform from the first scale.
    """

    return msim['scale0'][transform_key]


def multiscale_sel_coords(mxim, sel_dict):

    out_mxim = mxim.copy(deep=True)
    for child in mxim.children.keys():
        try:
            out_mxim[child] = out_mxim[child].sel(sel_dict)
        except:
            print('failed to sel for %s' %child)

    return out_mxim


def get_sorted_scale_keys(mxim):

    sorted_scale_keys = ['scale%s' %scale
        for scale in sorted([int(scale_key.split('scale')[-1])
            for scale_key in list(mxim.keys())
            if 'scale' in scale_key])] # there could be transforms also
    
    return sorted_scale_keys


def get_first_scale_above_target_spacing(mxim, target_spacing, dim='y'):

    sorted_scale_keys = get_sorted_scale_keys(mxim)

    for scale in sorted_scale_keys:
        scale_spacing = _spatial_image_utils.get_spacing_from_xim(mxim[scale]['image'])[dim]
        if scale_spacing > target_spacing:
            break

    return scale        


def multiscale_spatial_image_from_zarr(path):

    ndim = _spatial_image_utils.get_ndim_from_xim(datatree.open_datatree(path, engine="zarr")['scale0/image'])

    if ndim == 2:
        chunks = {'y': 256, 'x': 256}
    elif ndim == 3:
        # chunks = {'z': 64, 'y': 64, 'x': 64}
        chunks = {'z': 256, 'y': 256, 'x': 256}

    multiscale = datatree.open_datatree(path, engine="zarr", chunks=chunks)

    # compute transforms

    return multiscale


def get_optimal_multi_scale_factors_from_xim(sim, min_size=512):
    """
    This is currently simply downscaling z and xy until a minimum size is reached.
    Probably it'd make more sense to downscale considering the dims spacing.
    """

    spatial_dims = _spatial_image_utils.get_spatial_dims_from_xim(sim)
    current_shape = {dim: len(sim.coords[dim]) for dim in spatial_dims}
    factors = []
    while 1:
        curr_factors = {dim: 2 if current_shape[dim] >= min_size else 1 for dim in current_shape}
        if max(curr_factors.values()) == 1: break
        current_shape = {dim: int(current_shape[dim] / curr_factors[dim]) for dim in current_shape}
        factors.append(curr_factors)
        
    return factors


def get_transforms_from_dataset_as_dict(dataset):
    transforms_dict = {}
    for data_var, transform in dataset.items():
        if data_var == 'image': continue
        transform_key = data_var
        transforms_dict[transform_key] = transform
    return transforms_dict


def get_xim_from_msim(msim, scale='scale0'):
    """
    highest scale sim from msim with affine transforms
    """
    xim = msim['%s/image' %scale].copy()
    xim.attrs['transforms'] = get_transforms_from_dataset_as_dict(msim['scale0'])

    return xim


def get_msim_from_xim(xim, scale_factors=None):
    """
    highest scale sim from msim with affine transforms
    """

    spacing = _spatial_image_utils.get_spacing_from_xim(xim)
    origin = _spatial_image_utils.get_origin_from_xim(xim)

    if 'c' in xim.dims and 't' in xim.dims:
        xim = xim.transpose(*tuple(['t', 'c'] + [dim for dim in xim.dims if dim not in ['c', 't']]))
        c_coords = xim.coords['c'].values
    else:
        c_coords=None

    # view_xim.name = str(view)
    sim = si.to_spatial_image(
        xim.data,
        dims=xim.dims,
        c_coords=c_coords,
        scale=spacing,
        translation=origin,
        t_coords=xim.coords['t'].values,
        )

    if scale_factors is None:
        scale_factors = get_optimal_multi_scale_factors_from_xim(sim)

    msim = msi.to_multiscale(
        sim,
        chunks=256,
        scale_factors=scale_factors,
        )
    
    scale_keys = get_sorted_scale_keys(msim)
    for sk in scale_keys:
        for transform_key, transform in xim.attrs['transforms'].items():
            msim[sk][transform_key] = transform
    
    return msim


def set_affine_transform(msim, xaffine=None, transform_key=None, base_transform_key=None):

    assert(transform_key is not None)

    ndim = _spatial_image_utils.get_ndim_from_xim(get_xim_from_msim(msim))
    if xaffine is None:
        xaffine = [np.eye(ndim + 1)]

    if not isinstance(xaffine, xr.DataArray):
        xaffine = xr.DataArray(
            xaffine,
            dims=['t', 'x_in', 'x_out'])
        
    if not base_transform_key is None:
        xaffine = _spatial_image_utils.matmul_xparams(
            xaffine,
            get_transform_from_msim(msim, transform_key=base_transform_key))

    scale_keys = get_sorted_scale_keys(msim)
    for sk in scale_keys:
        msim[sk][transform_key] = xaffine


def ensure_time_dim(msim):

    if 't' in msim['scale0/image'].dims:
        return msim
    
    scale_keys = get_sorted_scale_keys(msim)
    for sk in scale_keys:
        msim[sk] = _spatial_image_utils.ensure_time_dim(msim[sk])

    return msim


def get_first_scale_above_target_spacing(mxim, target_spacing, dim='y'):

    sorted_scale_keys = get_sorted_scale_keys(mxim)

    for scale in sorted_scale_keys:
        scale_spacing = _spatial_image_utils.get_spacing_from_xim(mxim[scale]['image'])[dim]
        if scale_spacing > target_spacing:
            break

    return scale      
