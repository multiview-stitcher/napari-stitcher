import numpy as np
import networkx as nx
import xarray as xr
import dask.array as da
from dask import compute
from functools import partial
import warnings

import multiscale_spatial_image as msi
from spatial_image import to_spatial_image

from multiview_stitcher import mv_graph, spatial_image_utils, msi_utils, param_utils

from napari.experimental import link_layers
from napari.utils import notifications


def get_layer_dims(l,viewer):
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

    ldata = l.data
    
    if isinstance(ldata, xr.DataArray):
        dims = ldata.dims

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


def image_layer_to_msim(l, viewer):

    """
    Convert a napari layer into a MultiscaleSpatialImage compatible with multiview-stitcher.

    Parameters
    ----------
    l : napari.layers.Image
        l.data contains Union[array, xr.DataArray] for each scale

    Returns
    -------
    MultiscaleSpatialImage
        MultiscaleSpatialImage compatible with multiview-stitcher
    """

    if l.multiscale:

        msim = msi.MultiscaleSpatialImage()
        for isim, ldata in enumerate(l.data):

            # convert to SpatialImage if necessary
            if not isinstance(ldata, xr.DataArray):
                # need to implement downsampling logic for this
                raise(NotImplementedError('Multiscale layers with non-xarray data not supported yet.'))
            else:
                sdims = spatial_image_utils.get_spatial_dims_from_sim(ldata)

                ldata = ldata.assign_coords({'c': str(ldata.coords['c'].values)})

                sim = to_spatial_image(
                    ldata,
                    scale={dim: s for dim, s in zip(sdims, l.scale[-len(sdims):])},
                    translation={dim: t for dim, t in zip(sdims, l.translate[-len(sdims):])},
                    dims=ldata.dims,
                )

            msi.MultiscaleSpatialImage(name='scale%s' %isim, data=sim, parent=msim)
        
    else:
        # use dimension labels from viewer if indicated
        # consider that labels are set if x and y are present
        ldata = l.data
        dims = get_layer_dims(l,viewer)

        sdims = [dim for dim in dims if dim in ['x', 'y', 'z']]

        if not 't' in dims:
            dims = ['t'] + dims
            ldata = ldata[np.newaxis]

        # make sure to work with dask array
        if isinstance(ldata, xr.DataArray):
            data = ldata.data
        else:
            data = ldata

        if not isinstance(data, da.Array):
            data = da.from_array(data)

        sim = to_spatial_image(
            data,
            scale={dim: s for dim, s in zip(sdims, l.scale[-len(sdims):])},
            translation={dim: t for dim, t in zip(sdims, l.translate[-len(sdims):])},
            dims=dims,
        )

        if len(l.name.split(' :: ')) > 1:
            sim = sim.assign_coords(c=l.name.split(' :: ')[-1])
        else:
            sim = sim.assign_coords(c='default_channel')
            
        msim = msi.MultiscaleSpatialImage()
        msi.MultiscaleSpatialImage(name='scale%s' %0, data=sim, parent=msim)
        
    ndim = spatial_image_utils.get_ndim_from_sim(msi_utils.get_sim_from_msim(msim))
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
        link_layers(layers)

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

    if ndim == 3 and len(scale_keys) > 1:
        warnings.warn('In 3D, theres a napari bug concerning scale/translate when using multiscale images. Using only a single resolution (the lowest).')
        msim = msi_utils.get_msim_from_sim(msi_utils.get_sim_from_msim(msim, scale=scale_keys[-1]), scale_factors=[])
        scale_keys = msi_utils.get_sorted_scale_keys(msim)

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
        # 'contrast_limits': [np.iinfo(sim.dtype).min,
        #                     np.iinfo(sim.dtype).max],
        # 'contrast_limits': [np.iinfo(sim.dtype).min,
        #                     30],
        'name': name,
        'colormap': colormap,
        'gamma': 0.6,

        'affine': affine_transform,
        'translate': np.array([origin[dim] for dim in spatial_dims]),
        'scale': np.array([spacing[dim] for dim in spatial_dims]),
        'cache': True,
        'blending': blending,
        # 'multiscale': False,
        'multiscale': True,
        'metadata': {'full_affine_transform': affine_transform_xr}
        if transform_key is not None else None,
        }

    return [(multiscale_data, kwargs, 'image')]
    # return [(multiscale_data[0], kwargs, 'image')]


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

    if positional_cmaps:
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

    # layers_to_manage = [l for l in viewer.layers if l.name in layer_names_to_manage]

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
                # notifications.notification_manager.receive_info(
                #     'Timepoint %s: no parameters available, taking nearest available one.' % curr_tp)
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
