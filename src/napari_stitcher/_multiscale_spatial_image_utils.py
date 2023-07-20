"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/guides.html?#readers
"""
import os

import zarr
import numpy as np
from datatree import open_datatree
from natsort import natsorted

from functools import wraps
from pathlib import Path

from napari_stitcher._spatial_image_utils import (
    get_spatial_dims_from_xim,
    get_origin_from_xim,
    get_shape_from_xim,
    get_spacing_from_xim,
    get_ndim_from_xim,
    )


def get_optimal_multi_scale_factors_from_xim(sim):

    spatial_dims = get_spatial_dims_from_xim(sim)
    current_shape = {dim: len(sim.coords[dim]) for dim in spatial_dims}
    factors = []
    while 1:
        current_shape = {k: s/2 for k, s in current_shape.items()}
        if max (current_shape.values()) < 512: break
        factors.append({dim: 2 if current_shape[dim] >=1 else 1 for dim in current_shape})
        
    return factors


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


def multiscale_sel_coords(mxim, sel_dict):
    out_mxim = mxim.copy()
    for child in mxim.children.keys():
        try:
            out_mxim[out_mxim.path+child]['image'] = out_mxim[out_mxim.path+child]['image'].sel(sel_dict)
        except:
            print('failed to sel for %s' %child)
    return out_mxim


def get_sorted_scale_keys(mxim):

    sorted_scale_keys = ['scale%s' %scale
        for scale in sorted([int(scale_key.split('scale')[-1])
            for scale_key in list(mxim.keys())])]
    
    return sorted_scale_keys


def get_first_scale_above_target_spacing(mxim, target_spacing, dim='y'):

    sorted_scale_keys = get_sorted_scale_keys(mxim)

    for scale in sorted_scale_keys:
        scale_spacing = get_spacing_from_xim(mxim[scale]['image'])[dim]
        if scale_spacing > target_spacing:
            break

    return scale        


def multiscale_spatial_image_from_zarr(path):

    multiscale = open_datatree(path, engine="zarr")

    return multiscale


def spatialimage_to_napari_layerdata(sim):

    spatial_dims = get_spatial_dims_from_xim(sim)
    ndim = len(spatial_dims)

    spacing = get_spacing_from_xim(sim)
    origin = get_origin_from_xim(sim)

    # optional kwargs for the corresponding viewer.add_* method
    add_kwargs = {}

    add_kwargs['scale'] = [spacing[dim] for dim in spatial_dims]
    add_kwargs['translate'] = [origin[dim] for dim in spatial_dims]

    if 'c' in sim.dims:
        add_kwargs['channel_axis'] = list(sim.dims).index('c')

    add_kwargs['metadata'] = sim.attrs

    layer_type = "image"  # optional, default is "image"

    if 'phys2world' in sim.attrs:
        affine = np.array(sim.attrs['phys2world']).reshape((ndim+1, ndim+1))
        add_kwargs["affine"] = affine

    return (sim, add_kwargs, layer_type)


def multiscale_to_napari_layerdata(multiscale, transform="phys2world"):

    scales = [
        childname
        for childname in multiscale.children
        if childname.startswith("scale")
    ]
    scales = natsorted(scales)

    multiscale_data = []
    for scale in scales:
        keys = multiscale[scale].data_vars.keys()
        assert len(keys) == 1
        dataset_name = [key for key in keys][0]
        dataset = multiscale[scale].data_vars.get(dataset_name)
        multiscale_data.append(dataset)

    spatial_dims = get_spatial_dims_from_xim(multiscale.children['scale0'].data_vars['image'])
    ndim = len(spatial_dims)

    spacing = get_spacing_from_xim(multiscale.children['scale0'].data_vars['image'])
    origin = get_origin_from_xim(multiscale.children['scale0'].data_vars['image'])

    # optional kwargs for the corresponding viewer.add_* method
    add_kwargs = {}

    add_kwargs['scale'] = [spacing[dim] for dim in spatial_dims]
    add_kwargs['translate'] = [origin[dim] for dim in spatial_dims]

    if 'c' in multiscale['/scale0']['image'].dims:
        add_kwargs['channel_axis'] = list(multiscale['/scale0']['image'].dims).index('c')

    add_kwargs['metadata'] = multiscale.attrs

    # # Find the scale and translation information for the highest res dataset
    # scale = scales[
    #     0
    # ]  # sorted list, first element corresponds to highest resolution dataset
    # datasets = multiscale.attrs["multiscales"][0]["datasets"]
    # for dataset in datasets:
    #     if scale in dataset["path"]:
    #         for coord_transform in dataset["coordinateTransformations"]:
    #             if "scale" in coord_transform:
    #                 add_kwargs["scale"] = coord_transform["scale"]
    #             if "translation" in coord_transform:
    #                 add_kwargs["translate"] = coord_transform["translation"]

    layer_type = "image"  # optional, default is "image"

    if transform is not None:
        affine = np.array(multiscale.attrs[transform]).reshape((ndim+1, ndim+1))
        add_kwargs["affine"] = affine

    return (multiscale_data, add_kwargs, layer_type)


def get_center_of_mxim(mxim, transform=None):

    xim = mxim['scale0']['image']
    spatial_dims = get_spatial_dims_from_xim(xim)
    ndim = len(spatial_dims)

    center = np.array([np.mean(xim.coords[sd].values) for sd in spatial_dims])

    if not transform is None:
        affine = np.array(mxim.attrs[transform]).reshape((ndim+1, ndim+1))
        center = affine.dot(np.append(center, 1))[:ndim]

    return center


def get_xim_from_mxim(mxim):
    
    xim = mxim['scale0']['image']
    xim.attrs = mxim.attrs

    return xim
