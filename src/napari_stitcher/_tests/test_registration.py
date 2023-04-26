import numpy as np
import dask.array as da
import xarray as xr

from napari_stitcher import _registration, _sample_data, _spatial_image_utils, _reader, _mv_graph


def test_pairwise():

    layers = _sample_data.make_sample_data()

    xims = [l[0] for l in layers]

    spatial_dims = _spatial_image_utils.get_spatial_dims_from_xim(xims[0])

    pd = _registration.register_pair_of_spatial_images(xims,
            registration_binning={dim: 4 for dim in spatial_dims})

    # p = pd.compute(scheduler='single-threaded')
    p = pd

    assert np.allclose(p,
        np.array([[1.        , 0.        , 1.73333333],
                  [0.        , 1.        , 7.36666667],
                  [0.        , 0.        , 1.        ]]))


def test_register_graph():

    view_xims = _reader.read_mosaic_czi_into_list_of_spatial_xarrays(
        _sample_data.get_sample_data_path())
    
    g = _mv_graph.build_view_adjacency_graph_from_xims(view_xims)

    # g_pairs = _registration.get_registration_pair_graph(g)
    g_reg = _registration.get_registration_graph_from_overlap_graph(g)

    assert(max(['transform' in g_reg.edges[e].keys()
        for e in g_reg.edges]))
    
    assert([type(g_reg.edges[e]['transform'].data) == da.core.Array
        for e in g_reg.edges if 'transform' in g_reg.edges[e].keys()])
    
    # g_reg = _mv_graph.select_coords_from_graph(g_reg, {'C': [view_xims[0].coords['C'][0]]}, ['transform'])
    g_reg = _mv_graph.select_coords_from_graph(g_reg,
                                               {'C': [view_xims[0].coords['C'][0].data]},
                                               edge_attributes=['transform'])

    g_reg_computed = _mv_graph.compute_graph_edges(g_reg)
    
    assert([type(g_reg_computed.edges[e]['transform'].data) == np.ndarray
        for e in g_reg_computed.edges if 'transform' in g_reg_computed.edges[e].keys()])

    # get node parameters
    g_reg_nodes = _registration.get_node_params_from_reg_graph(g_reg_computed)

    assert(['transforms' in g_reg_nodes.nodes[n].keys()
            for n in g_reg_nodes.nodes])
    
    # import pdb; pdb.set_trace()
