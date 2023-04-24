import numpy as np
import networkx as nx
import xarray as xr
from dask import compute


def build_view_adjacency_graph_from_xims(xims, expand=False):
    """
    Build graph representing view overlap relationships from list of xarrays.
    Will be used for
      - groupwise registration
      - determining visualization colors
    """

    g = nx.Graph()
    for iview, xim in enumerate(xims):
        g.add_node(iview,
                   xim=xim)
        
    for iview1, xim1 in enumerate(xims):
        for iview2, xim2 in enumerate(xims):
            if iview1 >= iview2: continue
            
            # if iview1 == 0 and iview2 == 8:
            #     import pdb; pdb.set_trace()
            overlap = get_overlap_between_pair_of_xims(xim1, xim2, expand=expand)
            
            if overlap > 0:
                g.add_edge(iview1, iview2, overlap=overlap)

    return g


def get_overlap_between_pair_of_xims(xim1, xim2, expand=False):

    """
    How to handle T?
    """

    spatial_dims = xim1.attrs['spatial_dims']

    x1_i, x1_f = np.array([[xim1.coords[dim][index].data
                            for dim in spatial_dims]
                            for index in [0, -1]])
    
    x2_i, x2_f = np.array([[xim2.coords[dim][index].data
                            for dim in spatial_dims]
                            for index in [0, -1]])
    
    # # expand limits so that in case of no overlap the neighbours are shown
    # a = 10
    # if expand:
    #     x1_i = x1_i - a * xim1.attrs['spacing'].data
    #     x2_i = x2_i - a * xim2.attrs['spacing'].data

    #     x1_f = x1_f + a * xim1.attrs['spacing'].data
    #     x2_f = x2_f + a * xim2.attrs['spacing'].data

    # expand limits so that in case of no overlap the neighbours are shown
    a = 10
    if expand:
        x1_i = x1_i - a * np.array([xim1.coords[dim][1] - xim1.coords[dim][1] for dim in spatial_dims])
        x2_i = x2_i - a * np.array([xim2.coords[dim][1] - xim2.coords[dim][1] for dim in spatial_dims])

        x1_f = x1_f + a * np.array([xim1.coords[dim][1] - xim1.coords[dim][1] for dim in spatial_dims])
        x2_f = x2_f + a * np.array([xim2.coords[dim][1] - xim2.coords[dim][1] for dim in spatial_dims])

    dim_overlap_opt1 = (x1_f >= x2_i) * (x1_f <= x2_f)
    dim_overlap_opt2 = (x2_f >= x1_i) * (x2_f <= x1_f)

    dim_overlap = dim_overlap_opt1 + dim_overlap_opt2

    if np.all(dim_overlap):
        overlap = np.min([x2_f, x1_f], 0) - np.max([x2_i, x1_i], 0)
    else:
        overlap = np.array([0] * len(spatial_dims))
        
    overlap_area = np.product(overlap)

    return overlap_area


def get_registration_pairs_from_view_dict(view_dict, min_percentile=49):
    """
    Automatically determine list of pairwise views to be registered using
    'origin' and 'shape' information in view_dict.
    """
    ndim = len(view_dict[list(view_dict.keys())[0]]['spacing'])

    all_pairs, overlap_areas = [], []
    for iview1, v1 in view_dict.items():
        for iview2, v2 in view_dict.items():
            if iview1 >= iview2: continue

            x1_i, x1_f = np.array([[v1['origin'][dim], v1['origin'][dim] + v1['shape'][dim] * v1['spacing'][dim]] for dim in range(ndim)]).T
            x2_i, x2_f = np.array([[v2['origin'][dim], v2['origin'][dim] + v2['shape'][dim] * v2['spacing'][dim]] for dim in range(ndim)]).T

            dim_overlap_opt1 = (x1_f >= x2_i) * (x1_f <= x2_f)
            dim_overlap_opt2 = (x2_f >= x1_i) * (x2_f <= x1_f)

            dim_overlap = dim_overlap_opt1 + dim_overlap_opt2

            # print(iview1, iview2, x1_i, x1_f, x2_i, x2_f, dim_overlap_opt1, dim_overlap_opt2, dim_overlap)

            if np.all(dim_overlap):
                overlap = np.min([x2_f, x1_f], 0) - np.max([x2_i, x1_i], 0)
                # print(iview1, iview2, x1_i, x1_f, x2_i, x2_f, dim_overlap_opt1, dim_overlap_opt2, dim_overlap, overlap)
                all_pairs.append((iview1, iview2))
                overlap_areas.append(np.product(overlap))

    all_pairs, overlap_areas = np.array(all_pairs), np.array(overlap_areas)
    all_pairs = all_pairs[overlap_areas >= np.percentile(overlap_areas, min_percentile), :]

    return all_pairs


def get_node_with_maximal_overlap_from_graph(g):
    """
    g: graph containing edges with 'overlap' weight
    """
    total_node_overlaps = {node: np.sum([g.edges[e]['overlap']
            for e in g.edges if node in e])
        for node in g.nodes}
    ref_node = max(total_node_overlaps, key=total_node_overlaps.get)
    return ref_node


def get_registration_pairs_from_overlap_graph(g,
                                              method='shortest_paths_considering_overlap',
                                              min_percentile=49,
                                              ref_node=None,
                                              ):

    all_pairs = np.array([e for e in g.edges])

    if not len(all_pairs):
        raise(ValueError('No overlap between views found.'))

    if method == 'percentile':

        overlap_areas = [g.get_edge_data(e[0], e[1])['overlap'] for e in all_pairs]

        reg_pairs = all_pairs[overlap_areas >= np.percentile(overlap_areas, min_percentile), :]
    
    elif method == 'shortest_paths_considering_overlap':

        if ref_node is None:
            ref_node = get_node_with_maximal_overlap_from_graph(g)

        # invert overlap to use as weight in shortest path
        for e in g.edges:
            g.edges[e]['overlap_inv'] = 1 / g.edges[e]['overlap']

        # get shortest paths to ref_node
        paths = nx.shortest_path(g, source=ref_node, weight='overlap_inv')

        # get all pairs of views that are connected by a shortest path
        reg_pairs = []
        for _, sp in paths.items():
            if len(sp) < 2: continue
            for i in range(len(sp) - 1):
                pair = (sp[i], sp[i + 1])
                if pair not in reg_pairs:
                    reg_pairs.append(pair)

    else:
        raise NotImplementedError
    
    return reg_pairs


def compute_graph_edges(input_g, weight_name='transform', scheduler='threads'):

    """
    Perform simultaneous compute on all edge attributes with given name
    """

    g = input_g.copy()

    edge_weight_dict = {e: g.edges[e][weight_name]
                        for e in g.edges if weight_name in g.edges[e]}

    edge_weight_dict = compute(edge_weight_dict, scheduler=scheduler)[0]

    for e, w in edge_weight_dict.items():
        g.edges[e][weight_name] = w

    return g


if __name__ == "__main__":

    from napari_stitcher import _reader

    xims=_reader.read_mosaic_czi_into_list_of_spatial_xarrays(
        '../napari-stitcher/image-datasets/arthur_20220621_premovie_dish2-max.czi', scene_index=0)
    
    g = build_view_adjacency_graph_from_xims(xims)