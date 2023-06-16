import numpy as np
import networkx as nx

from napari_stitcher import _mv_graph, _spatial_image_utils

def create_image_layer_tuple_from_spatial_xim(xim,
                                              colormap='gray_r',
                                              name_prefix=None,
                                              ):

    """
    Note:
    - xarray.DataArray can have coordinates for dimensions that are not listed in xim.dims (?)
    - useful for channel names
    """

    ch_name = str(xim.coords['C'].data)

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

    metadata = \
        {
        # 'napari_stitcher_reader_function': 'read_mosaic_czi',
        # 'channel_name': ch_name,
        }
    
    # metadata['xr_attrs'] = xim.attrs.copy()

    spatial_dims = _spatial_image_utils.get_spatial_dims_from_xim(xim)
    origin = _spatial_image_utils.get_origin_from_xim(xim)
    spacing = _spatial_image_utils.get_spacing_from_xim(xim)

    contrast_limit_im = xim.sel(T=xim.coords['T'][0])
    if 'Z' in xim.dims:
        contrast_limit_im = contrast_limit_im.sel(Z=xim.coords['Z'][len(xim.coords['Z'])//2])

    kwargs = \
        {
        'contrast_limits': [v for v in [
            np.min(np.array(contrast_limit_im.data)),
            np.max(np.array(contrast_limit_im.data))]],
        # 'contrast_limits': [np.iinfo(xim.dtype).min,
        #                     np.iinfo(xim.dtype).max],
        # 'contrast_limits': [np.iinfo(xim.dtype).min,
        #                     30],
        'name': name,
        'colormap': colormap,
        'gamma': 0.6,
        # 'scale': [
        #     # xim.attrs['spacing'].loc[dim]
        #     xim.coords[dim][1] - xim.coords[dim][0]
        #           for dim in xim.attrs['spatial_dims']],
        # 'translate': [
        #     # xim.attrs['origin'].loc[dim]
        #     xim.coords[dim][0]
        #             for dim in xim.attrs['spatial_dims']],
        # 'affine': _spatial_image_utils.get_data_to_world_matrix_from_spatial_image(xim),
        'translate': np.array([origin[dim] for dim in spatial_dims]),
        'scale': np.array([spacing[dim] for dim in spatial_dims]),
        'cache': True,
        'blending': 'additive',
        'metadata': metadata,
        }

    return (xim, kwargs, 'image')


def create_image_layer_tuples_from_xims(
        xims,
        positional_cmaps=True,
        name_prefix="tile",
        n_colors=2,
):
    
    if positional_cmaps:
        cmaps = get_cmaps_from_xims(xims, n_colors=n_colors)
    else:
        cmaps = [None for xim in xims]

    out_layers = [
        create_image_layer_tuple_from_spatial_xim(
                    view_xim.sel(C=ch_coord),
                    cmaps[iview],
                    name_prefix=name_prefix + '_%03d' %iview)
            for iview, view_xim in enumerate(xims)
        for ch_coord in view_xim.coords['C']
        ]
    
    return out_layers


def get_cmaps_from_xims(xims, n_colors=2):
    """
    Get colors from view adjacency graph analysis

    Idea: use the same logic to determine relevant registration edges

    """

    mv_graph = _mv_graph.build_view_adjacency_graph_from_xims(xims, expand=True)

    # thresholds = threshold_multiotsu(overlaps)

    # strategy: remove edges with overlap values of increasing thresholds until
    # the graph division into n_colors is successful

    # modify overlap values
    # strategy: add a small amount to edge overlap depending on how many edges the nodes it connects have (betweenness?)

    edge_vals = nx.edge_betweenness_centrality(mv_graph)

    edges = [e for e in mv_graph.edges(data=True)]
    for e in edges:
        edge_vals[tuple(e[:2])] = edge_vals[tuple(e[:2])] + e[2]['overlap']

    sorted_unique_vals = sorted(np.unique([v for v in edge_vals.values()]))

    nx.set_edge_attributes(mv_graph, edge_vals, name='edge_val')

    thresh_ind = 0
    # while max([d for n, d in mv_graph.degree()]) >= n_colors:
    while 1:
        colors = nx.coloring.greedy_color(mv_graph)
        if len(set(colors.values())) <= n_colors:# and nx.coloring.equitable_coloring.is_equitable(mv_graph, colors):
            break
        mv_graph.remove_edges_from(
            [(a,b) for a, b, attrs in mv_graph.edges(data=True)
            if attrs["edge_val"] <= sorted_unique_vals[thresh_ind]])
        thresh_ind += 1

    cmaps = ['red', 'green', 'blue', 'gray']
    cmaps = {iview: cmaps[color_index % len(cmaps)]
             for iview, color_index in colors.items()}
    
    return cmaps
