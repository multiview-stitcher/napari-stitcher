import numpy as np
import networkx as nx


import multiscale_spatial_image as msi

from ngff_stitcher import mv_graph, spatial_image_utils, msi_utils

from napari.experimental import link_layers


def image_layer_to_msim(l):

    if l.multiscale:
        msim = msi.MultiscaleSpatialImage()
        for ixim, xim in enumerate(l.data):
            msi.MultiscaleSpatialImage(name='scale%s' %ixim, data=xim, parent=msim)
        
        ndim = spatial_image_utils.get_ndim_from_xim(msi_utils.get_xim_from_msim(msim))
        affine = np.array(l.affine.affine_matrix)[-(ndim+1):, -(ndim+1):]
        affine_xr = spatial_image_utils.affine_to_xr(affine, t_coords=l.data[0].coords['t'])
        msi_utils.set_affine_transform(
            msim, affine_xr, transform_key='affine_metadata')
        
        return msim
    else:
        raise(Exception('Napari image layer not supported.'))


def add_image_layer_tuples_to_viewer(viewer, lds, do_link_layers=False):
    """
    """

    layers = [viewer.add_image(ld[0], **ld[1]) for ld in lds]

    if do_link_layers:
        link_layers(layers, ('contrast_limits', 'visible'))

    return layers


def create_image_layer_tuple_from_msim(
    msim,
    colormap='gray',
    name_prefix=None,
    transform_key=None,
    ch_name=None,
    contrast_limits=None,
    blending='additive',
    ):

    """
    """

    xim = msim['scale0/image']
    scale_keys = msi_utils.get_sorted_scale_keys(msim)

    if contrast_limits is None:
        xim_thumb = msim[scale_keys[-1]]['image'].sel(t=xim.coords['t'][0])
        contrast_limits = [v for v in [
                    np.min(np.array(xim_thumb.data)),
                    np.max(np.array(xim_thumb.data))]]

    if ch_name is None:
        try:
            ch_name = str(xim.coords['c'].values[0])
        except:
            ch_name = str(xim.coords['c'].data)

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
        affine_transform = np.array(affine_transform_xr.sel(t=xim.coords['t'][0]).data)
    else:
        ndim = spatial_image_utils.get_ndim_from_xim(xim)
        affine_transform = np.eye(ndim + 1)

    multiscale_data = []
    for scale_key in scale_keys:
        multiscale_xim = msim[scale_key]['image']
        multiscale_xim.attrs['transforms'] = msi_utils.get_transforms_from_dataset_as_dict(msim[scale_key])
        multiscale_data.append(multiscale_xim)

    spatial_dims = spatial_image_utils.get_spatial_dims_from_xim(
        xim)
    ndim = len(spatial_dims)

    spacing = spatial_image_utils.get_spacing_from_xim(xim)
    origin = spatial_image_utils.get_origin_from_xim(xim)

    kwargs = \
        {
        'contrast_limits': contrast_limits,
        # 'contrast_limits': [np.iinfo(xim.dtype).min,
        #                     np.iinfo(xim.dtype).max],
        # 'contrast_limits': [np.iinfo(xim.dtype).min,
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
        }

    return (multiscale_data, kwargs, 'image')


def create_image_layer_tuples_from_msims(
        msims,
        positional_cmaps=True,
        name_prefix="tile",
        n_colors=2,
        transform_key=None,
):

    xims = [msi_utils.get_xim_from_msim(msim) for msim in msims]

    if positional_cmaps:
        cmaps = get_cmaps_from_xims(
            [spatial_image_utils.xim_sel_coords(xim, {'t':xim.coords['t'][0]}) for xim in xims],
            n_colors=n_colors, transform_key=transform_key)
    else:
        cmaps = [None for _ in msims]

    out_layers = [
        create_image_layer_tuple_from_msim(
                    msi_utils.multiscale_sel_coords(msim, {'c': ch_coord}),
                    cmaps[iview],
                    name_prefix=name_prefix + '_%03d' %iview,
                    transform_key=transform_key,
                    )
            for iview, msim in enumerate(msims)
        for ch_coord in msim['scale0/image'].coords['c']
        ]
    
    return out_layers


def get_cmaps_from_xims(xims, n_colors=2, transform_key=None):
    """
    Get colors from view adjacency graph analysis

    Idea: use the same logic to determine relevant registration edges

    """

    view_adj_graph = mv_graph.build_view_adjacency_graph_from_xims(
        xims, expand=True, transform_key=transform_key)

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
        spatial_image_utils.set_xim_affine(
            sim,
            xaffine,
            transform_key=transform_key, 
            base_transform_key=base_transform_key)
    return
