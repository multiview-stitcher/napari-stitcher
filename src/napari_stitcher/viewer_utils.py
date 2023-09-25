import numpy as np
import networkx as nx
from dask import compute
from functools import partial

import multiscale_spatial_image as msi

from ngff_stitcher import mv_graph, spatial_image_utils, msi_utils

from napari.experimental import link_layers
from napari.utils import notifications


def image_layer_to_msim(l):

    if l.multiscale:
        msim = msi.MultiscaleSpatialImage()
        for isim, sim in enumerate(l.data):
            msi.MultiscaleSpatialImage(name='scale%s' %isim, data=sim, parent=msim)
        
        ndim = spatial_image_utils.get_ndim_from_sim(msi_utils.get_sim_from_msim(msim))
        affine = np.array(l.affine.affine_matrix)[-(ndim+1):, -(ndim+1):]
        affine_xr = spatial_image_utils.affine_to_xr(affine, t_coords=l.data[0].coords['t'])
        msi_utils.set_affine_transform(
            msim, affine_xr, transform_key='affine_metadata')
        
        return msim
    else:
        raise(Exception('Napari image layer not supported.'))


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
        viewer.dims.events.connect(
            partial(manage_viewer_transformations_callback, viewer=viewer))

    return layers


# def create_image_layer_tuple_from_msim(
#     msim,
#     colormap='gray',
#     name_prefix=None,
#     transform_key=None,
#     ch_name=None,
#     contrast_limits=None,
#     blending='additive',
#     ):

#     """
#     """

#     sim = msim['scale0/image']
#     scale_keys = msi_utils.get_sorted_scale_keys(msim)

#     if contrast_limits is None:
#         sim_thumb = msim[scale_keys[-1]]['image'].sel(t=sim.coords['t'][0])
#         contrast_limits = [v for v in [
#                     np.min(np.array(sim_thumb.data)),
#                     np.max(np.array(sim_thumb.data))]]

#     if ch_name is None:
#         try:
#             ch_name = str(sim.coords['c'].values[0])
#         except:
#             ch_name = str(sim.coords['c'].data)

#     if colormap is None:
#         if 'GFP' in ch_name:
#             colormap = 'green'
#         elif 'RFP' in ch_name:
#             colormap = 'red'
#         else:
#             colormap = 'gray',

#     if name_prefix is None:
#         name = ch_name
#     else:
#         name = ' :: '.join([name_prefix, ch_name])

#     if not transform_key is None:
#         affine_transform_xr = msi_utils.get_transform_from_msim(msim, transform_key=transform_key)
#         affine_transform = np.array(affine_transform_xr.sel(t=sim.coords['t'][0]).data)
#     else:
#         ndim = spatial_image_utils.get_ndim_from_sim(sim)
#         affine_transform = np.eye(ndim + 1)

#     multiscale_data = []
#     for scale_key in scale_keys:
#         multiscale_sim = msim[scale_key]['image']
#         multiscale_sim.attrs['transforms'] = msi_utils.get_transforms_from_dataset_as_dict(msim[scale_key])
#         multiscale_data.append(multiscale_sim)

#     spatial_dims = spatial_image_utils.get_spatial_dims_from_sim(
#         sim)
#     ndim = len(spatial_dims)

#     spacing = spatial_image_utils.get_spacing_from_sim(sim)
#     origin = spatial_image_utils.get_origin_from_sim(sim)

#     kwargs = \
#         {
#         'contrast_limits': contrast_limits,
#         # 'contrast_limits': [np.iinfo(sim.dtype).min,
#         #                     np.iinfo(sim.dtype).max],
#         # 'contrast_limits': [np.iinfo(sim.dtype).min,
#         #                     30],
#         'name': name,
#         'colormap': colormap,
#         'gamma': 0.6,

#         'affine': affine_transform,
#         'translate': np.array([origin[dim] for dim in spatial_dims]),
#         'scale': np.array([spacing[dim] for dim in spatial_dims]),
#         'cache': True,
#         'blending': blending,
#         'multiscale': True,
#         }

#     return (multiscale_data, kwargs, 'image')


# def create_image_layer_tuples_from_msims(
#         msims,
#         positional_cmaps=True,
#         name_prefix="tile",
#         n_colors=2,
#         transform_key=None,
# ):

#     sims = [msi_utils.get_sim_from_msim(msim) for msim in msims]

#     if positional_cmaps:
#         cmaps = get_cmaps_from_sims(
#             [spatial_image_utils.sim_sel_coords(sim, {'t':sim.coords['t'][0]}) for sim in sims],
#             n_colors=n_colors, transform_key=transform_key)
#     else:
#         cmaps = [None for _ in msims]

#     channels = []

#     out_layers = [
#         create_image_layer_tuple_from_msim(
#                     msi_utils.multiscale_sel_coords(msim, {'c': ch_coord}),
#                     cmaps[iview],
#                     name_prefix=name_prefix + '_%03d' %iview,
#                     transform_key=transform_key,
#                     )
#             for iview, msim in enumerate(msims)
#         for ch_coord in msim['scale0/image'].coords['c']
#         ]
    
#     return out_layers


def create_image_layer_tuples_from_msim(
    msim,
    colormap=None,
    name_prefix=None,
    transform_key=None,
    ch_name=None,
    contrast_limits=None,
    blending='additive',
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
                )
            
        return out_layers

    sim = msi_utils.get_sim_from_msim(msim)
    scale_keys = msi_utils.get_sorted_scale_keys(msim)

    if contrast_limits is None:
        sim_thumb = msim[scale_keys[-1]]['image'].sel(t=sim.coords['t'][0])
        contrast_limits = [v for v in [
                    compute(np.min(sim_thumb.data))[0],
                    compute(np.max(sim_thumb.data))[0]]]

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

    if not transform_key is None:
        affine_transform_xr = msi_utils.get_transform_from_msim(msim, transform_key=transform_key)
        affine_transform = np.array(affine_transform_xr.sel(t=sim.coords['t'][0]).data)
    else:
        ndim = spatial_image_utils.get_ndim_from_sim(sim)
        affine_transform = np.eye(ndim + 1)

    multiscale_data = []
    for scale_key in scale_keys:
        multiscale_sim = msim[scale_key]['image']
        multiscale_sim.attrs['transforms'] = msi_utils.get_transforms_from_dataset_as_dict(msim[scale_key])
        multiscale_data.append(multiscale_sim)

    spatial_dims = spatial_image_utils.get_spatial_dims_from_sim(
        sim)
    ndim = len(spatial_dims)

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
):

    sims = [msi_utils.get_sim_from_msim(msim) for msim in msims]

    if positional_cmaps:
        cmaps = get_cmaps_from_sims(
            [spatial_image_utils.sim_sel_coords(sim, {'t':sim.coords['t'][0]}) for sim in sims],
            n_colors=n_colors, transform_key=transform_key)
    else:
        cmaps = [None for _ in msims]

    out_layers = []
    for iview, msim in enumerate(msims):
        out_layers += create_image_layer_tuples_from_msim(
            # msi_utils.multiscale_sel_coords(msim, {'c': ch_coord}),
            msim,
            cmaps[iview],
            name_prefix=name_prefix + '_%03d' %iview,
            transform_key=transform_key,
            contrast_limits=contrast_limits,
            )
    
    return out_layers


def get_cmaps_from_sims(sims, n_colors=2, transform_key=None):
    """
    Get colors from view adjacency graph analysis

    Idea: use the same logic to determine relevant registration edges

    """

    view_adj_graph = mv_graph.build_view_adjacency_graph_from_msims(
        [msi_utils.get_msim_from_sim(sim, scale_factors=[]) for sim in sims],
        expand=True,
        transform_key=transform_key
        )

    # thresholds = threshold_multiotsu(overlaps)

    # strategy: remove edges with overlap values of increasing thresholds until
    # the graph division into n_colors is successful

    # modify overlap values
    # strategy: add a small amount to edge overlap depending on how many edges the nodes it connects have (betweenness?)

    edge_vals = nx.edge_betweenness_centrality(view_adj_graph)

    edges = [e for e in view_adj_graph.edges(data=True)]
    for e in edges:
        edge_vals[tuple(e[:2])] = edge_vals[tuple(e[:2])] + e[2]['overlap']

    sorted_unique_vals = sorted(np.unique([v for v in edge_vals.values()]))

    nx.set_edge_attributes(view_adj_graph, edge_vals, name='edge_val')

    thresh_ind = 0
    while 1:
        colors = nx.coloring.greedy_color(view_adj_graph)
        if len(set(colors.values())) <= n_colors:# and nx.coloring.equitable_coloring.is_equitable(view_adj_graph, colors):
            break
        view_adj_graph.remove_edges_from(
            [(a,b) for a, b, attrs in view_adj_graph.edges(data=True)
            if attrs["edge_val"] <= sorted_unique_vals[thresh_ind]])
        thresh_ind += 1

    cmaps = ['red', 'green', 'blue', 'yellow']
    cmaps = {iview: cmaps[color_index % len(cmaps)]
             for iview, color_index in colors.items()}
    
    return cmaps


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

    # compatible_layers = [l for l in self.viewer.layers
    #                         if si.is_spatial_image(l.data[0])]
    
    # consider all layers for now
    compatible_layers = viewer.layers
    
    if not len(compatible_layers): return
    
    # determine spatial dimensions from layers
    all_spatial_dims = [spatial_image_utils.get_spatial_dims_from_sim(
        l.data[0]) for l in compatible_layers]
    
    highest_sdim = max([len(sdim) for sdim in all_spatial_dims])

    # get curr tp
    # handle possibility that there had been no T dimension
    # when collecting sims from layers
    if len(viewer.dims.current_step) > highest_sdim:
        curr_tp = viewer.dims.current_step[-highest_sdim-1]
    else:
        curr_tp = 0

    for _, l in enumerate(compatible_layers):

        if not 'full_affine_transform' in l.metadata.keys(): continue

        layer_sim = l.data[0]

        params = l.metadata['full_affine_transform']

        try:
            p = np.array(params.sel(t=layer_sim.coords['t'][curr_tp])).squeeze()

        except:
            notifications.notification_manager.receive_info(
                'Timepoint %s: no parameters available for tp %s' % curr_tp)
            continue
            # if curr_tp not available, use nearest available parameter
            # notifications.notification_manager.receive_info(
            #     'Timepoint %s: no parameters available, taking nearest available one.' % curr_tp)
            # p = np.array(params.sel(t=layer_sim.coords['t'][curr_tp], method='nearest')).squeeze()

        ndim_layer_data = len(layer_sim.shape)

        # if stitcher sim has more dimensions than layer data (i.e. time)
        vis_p = p[-(ndim_layer_data + 1):, -(ndim_layer_data + 1):]

        # if layer data has more dimensions than stitcher sim
        full_vis_p = np.eye(ndim_layer_data + 1)
        full_vis_p[-len(vis_p):, -len(vis_p):] = vis_p

        l.affine.affine_matrix = full_vis_p

        # refreshing layers fails sometimes
        # this solution is suboptimal though
        try:
            l.refresh()
        except:
            pass