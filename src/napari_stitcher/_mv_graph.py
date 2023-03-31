import numpy as np
import networkx as nx
import xarray as xr


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
    
    # expand limits so that in case of no overlap the neighbours are shown
    a = 10
    if expand:
        x1_i = x1_i - a * xim1.attrs['spacing'].data
        x2_i = x2_i - a * xim2.attrs['spacing'].data

        x1_f = x1_f + a * xim1.attrs['spacing'].data
        x2_f = x2_f + a * xim2.attrs['spacing'].data

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


if __name__ == "__main__":

    from napari_stitcher import _reader