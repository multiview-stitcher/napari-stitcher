import numpy as np

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

            # store_path = Path(store_path)

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

    # return msim.attrs['transforms'][transform_key]


def multiscale_sel_coords(mxim, sel_dict):

    out_mxim = mxim.copy(deep=True)
    for child in mxim.children.keys():
        try:
            out_mxim[child] = out_mxim[child].sel(sel_dict)
        except:
            print('failed to sel for %s' %child)

    # # sel transforms which are xr.Datasets in the mxim attributes
    # for data_var in mxim.attrs['transforms']:
    #     for k, v in sel_dict.items():
    #         if k in out_mxim.attrs['transforms'][data_var].dims:
    #             out_mxim.attrs['transforms'][data_var] = out_mxim.attrs['transforms'][data_var].sel({k: v})

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

    return multiscale


def get_optimal_multi_scale_factors_from_xim(sim):

    spatial_dims = _spatial_image_utils.get_spatial_dims_from_xim(sim)
    current_shape = {dim: len(sim.coords[dim]) for dim in spatial_dims}
    factors = []
    while 1:
        current_shape = {k: s/2 for k, s in current_shape.items()}
        if max (current_shape.values()) < 256: break
        factors.append({dim: 2 if current_shape[dim] >=1 else 1 for dim in current_shape})
        
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

    # view_xim.name = str(view)
    sim = si.to_spatial_image(
        xim.data,
        dims=xim.dims,
        c_coords=xim.coords['c'].values,
        scale=spacing,
        translation=origin,
        t_coords=xim.coords['t'].values,
        )

    if scale_factors is None:
        scale_factors = get_optimal_multi_scale_factors_from_xim(sim)

    msim = msi.to_multiscale(
        sim,
        # scale_factors=_msi_utils.get_optimal_multi_scale_factors_from_xim(view_sim),
        chunks=256,
        scale_factors=scale_factors,
        )
    
    scale_keys = get_sorted_scale_keys(msim)
    for sk in scale_keys:
        for transform_key, transform in xim.attrs['transforms'].items():
            msim[sk][transform_key] = transform
    
    # msim.attrs['transforms'] = xim.attrs['transforms']

    return msim


import xarray as xr
def set_affine_transform(msim, affine, transform_key):

    if not isinstance(affine, xr.DataArray):
        affine = xr.DataArray(
            np.stack([affine] * len(msim['scale0/image'].coords['t'])),
            dims=['t', 'x_in', 'x_out'])

    scale_keys = get_sorted_scale_keys(msim)
    for sk in scale_keys:
        msim[sk][transform_key] = affine
