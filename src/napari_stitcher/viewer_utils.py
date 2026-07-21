import numpy as np
import xarray as xr
import dask.array as da
from dask import compute
from functools import partial

from multiview_stitcher import (
    mv_graph,
    spatial_image_utils,
    msi_utils,
    param_utils,
)

from . import _utils

from napari.experimental import link_layers
from napari.utils import notifications


def get_layer_dims(l, viewer):
    """
    Get the dimensions of a napari layer.

    Parameters
    ----------
    l : napari.layers.Image
        l.data contains Union[array, xr.DataArray] for each scale
    viewer : napari.Viewer
        Napari viewer

    Returns
    -------
    dims : list
        List of dimensions of the layer
    """

    ldata = l.data[0] if l.multiscale else l.data

    if isinstance(ldata, xr.DataArray):
        dims = list(ldata.dims)

    # infer dimensions for images loaded with napari-aicsimageio
    elif 'aicsimage' in l.metadata:
        xim = l.metadata['aicsimage']
        xim = xim.squeeze()
        dims = [dim.lower() for dim in xim.dims]
        dims = [dim for dim in dims if dim not in ['c']] # remove channel dim

    else:
        ndim = len(ldata.shape)
        dims = ['t', 'z', 'y', 'x'][-ndim:]

    return dims


def set_msims_affine_transforms_from_viewer(viewer, msims, transform_key):
    viewer_affines = [l.affine.affine_matrix for l in viewer.layers]
    for viewer_affine, msim in zip(viewer_affines, msims):
        sim = msi_utils.get_sim_from_msim(msim)
        if 't' in sim.coords:
            t_coords = sim.coords['t'].values
        else:
            t_coords = None
        affine = param_utils.affine_to_xaffine(viewer_affine, t_coords=t_coords)
        msi_utils.set_affine_transform(
            msim, affine, transform_key=transform_key)


def image_layer_to_msim(l, viewer):
    """
    Convert a napari Image layer into a MultiscaleSpatialImage compatible
    with multiview-stitcher.

    Handles both single-scale and multiscale napari Image layers, and both
    xarray.DataArray-backed and raw array (numpy/dask) data.

    Parameters
    ----------
    l : napari.layers.Image
        Napari image layer. l.data is a list of arrays/DataArrays for
        multiscale layers, or a single array for single-scale layers.
    viewer : napari.Viewer
        Napari viewer instance.

    Returns
    -------
    MultiscaleSpatialImage
        MultiscaleSpatialImage compatible with multiview-stitcher.
    """
    dims = get_layer_dims(l, viewer)
    sdims = [dim for dim in dims if dim in ['x', 'y', 'z']]

    ch_name = _utils.get_layer_c_coord(l)

    # Loading the layer from OME-Zarr directly
    # if the layer was read using napari-ome-zarr.
    # The benefit of this is that msims are backed
    # by zarr arrays rather than dask arrays,
    # which is more efficient for large datasets.
    msim = _utils.try_load_msim_from_napari_ome_zarr(
        l, viewer, dims, sdims, ch_name)

    if msim is None and l.multiscale:
        sims = []
        # All levels cover the same physical region, so coarser levels have
        # proportionally larger pixel sizes. Compute the per-level scale from
        # the shape ratio between the finest (level 0) and each level.
        finest_spatial_shape = np.array(l.data[0].shape[-len(sdims):], dtype=float)
        for ldata in l.data:
            level_spatial_shape = np.array(ldata.shape[-len(sdims):], dtype=float)
            level_scale_factors = finest_spatial_shape / level_spatial_shape

            if isinstance(ldata, xr.DataArray):
                # xarray-backed multiscale (e.g. from napari-aicsimageio)
                xim_sdims = spatial_image_utils.get_spatial_dims_from_sim(ldata)
                c_coord = _utils.get_layer_c_coord(l, ldata)
                sim = spatial_image_utils.get_sim_from_array(
                    ldata,
                    dims=list(ldata.dims),
                    scale={dim: l.scale[-(len(xim_sdims) - j)] * level_scale_factors[j]
                           for j, dim in enumerate(xim_sdims)},
                    translation={dim: t for dim, t
                                 in zip(xim_sdims, l.translate[-len(xim_sdims):])},
                    c_coords=[c_coord],
                )
            else:
                # Raw array (numpy/dask) multiscale
                data = ldata if isinstance(ldata, da.Array) \
                    else da.from_array(ldata)
                sim = spatial_image_utils.get_sim_from_array(
                    data,
                    dims=dims,
                    scale={dim: l.scale[-(len(sdims) - j)] * level_scale_factors[j]
                           for j, dim in enumerate(sdims)},
                    translation={dim: t for dim, t
                                 in zip(sdims, l.translate[-len(sdims):])},
                    c_coords=[ch_name],
                )
            sims.append(sim)

        # Build multiscale image from the individual resolution levels, then
        # correct the per-level origins for the half-pixel offset introduced
        # by downsampling (OME-Zarr v0.6 intrinsic coordinate convention).
        msim = msi_utils.get_msim_from_sims(sims)
        msim = msi_utils.correct_multiscale_origins(msim)

    elif msim is None:
        # Single-scale layer
        ldata = l.data
        data = ldata.data if isinstance(ldata, xr.DataArray) else ldata
        if not isinstance(data, da.Array):
            data = da.from_array(data)

        sim = spatial_image_utils.get_sim_from_array(
            data,
            dims=dims,
            scale={dim: s for dim, s in zip(sdims, l.scale[-len(sdims):])},
            translation={dim: t for dim, t
                         in zip(sdims, l.translate[-len(sdims):])},
            c_coords=[ch_name],
        )
        msim = msi_utils.get_msim_from_sim(sim, scale_factors=[])

    # Store the napari layer's affine transform as the metadata transform
    sim = msi_utils.get_sim_from_msim(msim)
    ndim = spatial_image_utils.get_ndim_from_sim(sim)
    affine = np.array(l.affine.affine_matrix)[-(ndim+1):, -(ndim+1):]
    affine_xr = param_utils.affine_to_xaffine(affine, t_coords=sim.coords['t'])
    msi_utils.set_affine_transform(
        msim, affine_xr, transform_key='affine_metadata')

    return msim


def add_image_layer_tuples_to_viewer(
        viewer, lds,
        do_link_layers=False,
        manage_viewer_transformations=True,
        ):
    """
    """

    layers = [viewer.add_image(ld[0], **ld[1]) for ld in lds]

    if do_link_layers:
        # make sure not to link affine transformations
        link_layers(layers, attributes=['contrast_limits', 'visible'])

    # add callback to manage viewer transformations
    # (napari doesn't yet support different affine transforms for a single layer)
    if manage_viewer_transformations:

        for l in layers:
            l.metadata['napari_stitcher_manage_transformations'] = True

        if manage_viewer_transformations_callback not in viewer.dims.events.callbacks:
            viewer.dims.events.connect(
                partial(manage_viewer_transformations_callback,
                        viewer=viewer)
                        )

    return layers


def create_image_layer_tuples_from_msim(
    msim,
    colormap=None,
    name_prefix=None,
    transform_key=None,
    ch_name=None,
    contrast_limits=None,
    blending='additive',
    data_as_array=False,
    ):

    """
    Convert a MultiscaleSpatialImage into a list of napari layer data tuples.

    If the msim has a channel dimension, one layer tuple is returned per channel.

    Parameters
    ----------
    msim : MultiscaleSpatialImage
    colormap : str, optional
    name_prefix : str, optional
    transform_key : str, optional
        Which transform key to apply as the layer affine. None means identity.
    ch_name : str, optional
    contrast_limits : list of two floats, optional
    blending : str, optional
    data_as_array : bool, optional
        If True, return raw dask arrays instead of SpatialImages for each scale.

    Returns
    -------
    list of (data, kwargs, 'image') tuples
    """

    if 'c' in msi_utils.get_dims(msim):
        out_layers = []
        for ch_coord in msi_utils.get_sim_from_msim(msim).coords['c']:

            out_layers += create_image_layer_tuples_from_msim(
                msi_utils.multiscale_sel_coords(msim, {'c': ch_coord}),
                colormap=colormap,
                name_prefix=name_prefix,
                transform_key=transform_key,
                ch_name=str(ch_coord.values),
                contrast_limits=contrast_limits,
                blending=blending,
                data_as_array=data_as_array,
                )

        return out_layers

    scale_keys = msi_utils.get_sorted_scale_keys(msim)
    ndim = spatial_image_utils.get_ndim_from_sim(msi_utils.get_sim_from_msim(msim))

    sim = msi_utils.get_sim_from_msim(msim)

    if contrast_limits is None:
        sim_thumb = msim[scale_keys[-1]]['image'].sel(t=sim.coords['t'][0])
        contrast_limits = [
            compute(np.min(sim_thumb.data))[0],
            compute(np.max(sim_thumb.data))[0]]
        if contrast_limits[0] == contrast_limits[1]:
            contrast_limits[1] = contrast_limits[1] + 1

    if ch_name is None:
        try:
            ch_name = str(sim.coords['c'].values[0])
        except:
            ch_name = str(sim.coords['c'].data)

    if colormap is None:
        if 'GFP' in ch_name:
            colormap = 'green'
        elif 'RFP' in ch_name:
            colormap = 'red'
        else:
            colormap = 'gray',

    if name_prefix is None:
        name = ch_name
    else:
        name = ' :: '.join([name_prefix, ch_name])

    if transform_key is not None:
        affine_transform_xr = msi_utils.get_transform_from_msim(msim, transform_key=transform_key)
        if 't' in affine_transform_xr.dims:
            affine_transform = np.array(affine_transform_xr.sel(t=sim.coords['t'][0]).data)
        else:
            affine_transform = np.array(affine_transform_xr.data)
    else:
        ndim = spatial_image_utils.get_ndim_from_sim(sim)
        affine_transform = np.eye(ndim + 1)

    multiscale_data = []
    for scale_key in scale_keys:
        multiscale_sim = msi_utils.get_sim_from_msim(msim, scale=scale_key)
        if data_as_array:
            multiscale_sim = multiscale_sim.data
        multiscale_data.append(multiscale_sim)

    spatial_dims = spatial_image_utils.get_spatial_dims_from_sim(
        sim)

    spacing = spatial_image_utils.get_spacing_from_sim(sim)
    origin = spatial_image_utils.get_origin_from_sim(sim)

    kwargs = \
        {
        'contrast_limits': contrast_limits,
        'name': name,
        'colormap': colormap,
        'gamma': 0.6,

        'affine': affine_transform,
        'translate': np.array([origin[dim] for dim in spatial_dims]),
        'scale': np.array([spacing[dim] for dim in spatial_dims]),
        'cache': True,
        'blending': blending,
        'multiscale': True,
        'metadata': {'full_affine_transform': affine_transform_xr}
        if transform_key is not None else None,
        }

    return [(multiscale_data, kwargs, 'image')]


def create_image_layer_tuples_from_msims(
        msims,
        positional_cmaps=True,
        name_prefix="tile",
        n_colors=2,
        transform_key=None,
        contrast_limits=None,
        ch_coord=None,
        data_as_array=False,
):
    """
    Convert a list of MultiscaleSpatialImages into napari layer data tuples.

    Parameters
    ----------
    msims : list of MultiscaleSpatialImage
    positional_cmaps : bool, optional
        If True, assign greedy colors based on spatial overlap.
    name_prefix : str, optional
    n_colors : int, optional
    transform_key : str, optional
    contrast_limits : list, optional
    ch_coord : optional
        If given, select only this channel coordinate from each msim.
    data_as_array : bool, optional

    Returns
    -------
    list of (data, kwargs, 'image') tuples
    """

    if positional_cmaps and len(msims) > 1:
        sims = [spatial_image_utils.get_sim_field(
            msi_utils.get_sim_from_msim(msim)) for msim in msims]
        cmaps = ['red', 'green', 'blue', 'yellow']
        greedy_colors = mv_graph.get_greedy_colors(
            sims, n_colors=n_colors, transform_key=transform_key)
        cmaps = [cmaps[greedy_colors[iview] % len(cmaps)] for iview in range(len(msims))]
    else:
        cmaps = [None for _ in msims]

    out_layers = []
    for iview, msim in enumerate(msims):
        out_layers += create_image_layer_tuples_from_msim(
            msim if ch_coord is None
            else msi_utils.multiscale_sel_coords(msim, {'c': ch_coord}),
            cmaps[iview],
            name_prefix=name_prefix + '_%03d' %iview,
            transform_key=transform_key,
            contrast_limits=contrast_limits,
            data_as_array=data_as_array,
            )

    return out_layers


def set_layer_xaffine(l, xaffine, transform_key, base_transform_key=None):
    for sim in l.data:
        spatial_image_utils.set_sim_affine(
            sim,
            xaffine,
            transform_key=transform_key,
            base_transform_key=base_transform_key)
    return


def manage_viewer_transformations_callback(event, viewer):
    """
    set transformations
    - for current timepoint
    - for each (compatible) layer loaded in viewer
    """

    try:
        # events are constantly triggered by viewer.dims.events,
        # but we only want to update if current_step changes
        if hasattr(event, 'type') and\
        event.type != 'current_step': return
    except AttributeError:
        pass

    layers_to_manage = [l for l in viewer.layers
                        if 'napari_stitcher_manage_transformations' in l.metadata.keys()
                        and l.metadata['napari_stitcher_manage_transformations']]

    if not len(layers_to_manage): return

    # determine spatial dimensions from layers
    all_spatial_dims = [spatial_image_utils.get_spatial_dims_from_sim(
        l.data[0]) for l in layers_to_manage]

    highest_sdim = max([len(sdim) for sdim in all_spatial_dims])

    # get curr tp
    # handle possibility that there had been no T dimension
    # when collecting sims from layers
    if len(viewer.dims.current_step) > highest_sdim:
        curr_tp = viewer.dims.current_step[-highest_sdim-1]
    else:
        curr_tp = 0

    for _, l in enumerate(layers_to_manage):

        if not 'full_affine_transform' in l.metadata.keys(): continue

        layer_sim = l.data[0]

        params = l.metadata['full_affine_transform']

        if 't' in params.dims:
            try:
                p = np.array(params.sel(t=layer_sim.coords['t'][curr_tp])).squeeze()
            except:
                notifications.notification_manager.receive_info(
                    'Timepoint %s: no parameters available for tp' % curr_tp)
                # if curr_tp not available, use nearest available parameter
                p = np.array(params.sel(t=layer_sim.coords['t'][curr_tp], method='nearest')).squeeze()
                continue
        else:
            p = np.array(params).squeeze()

        ndim_layer_data = l.ndim

        # if stitcher sim has more dimensions than layer data (i.e. time)
        vis_p = p[-(ndim_layer_data + 1):, -(ndim_layer_data + 1):]

        # if layer data has more dimensions than stitcher sim
        full_vis_p = np.eye(ndim_layer_data + 1)
        full_vis_p[-len(vis_p):, -len(vis_p):] = vis_p

        l.affine = full_vis_p
