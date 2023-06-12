from napari_stitcher import _mv_graph, _sample_data, _reader, _spatial_image_utils

def test_overlap():

    view_xims = _reader.read_mosaic_czi_into_list_of_spatial_xarrays(
        _sample_data.get_sample_data_path())
    
    overlap_nonempty = _mv_graph.get_overlap_between_pair_of_xims(view_xims[0],
                                               view_xims[1])
    
    assert(overlap_nonempty > 0)

    # make sure second xim doesn't overlap with first one
    spatial_dims = _spatial_image_utils.get_spatial_dims_from_xim(view_xims[0])
    for dim in spatial_dims:
        view_xims[1] = view_xims[1].assign_coords(
            {dim: view_xims[1].coords[dim] + view_xims[0].coords[dim][-1] + 0.1})

    overlap_empty = _mv_graph.get_overlap_between_pair_of_xims(view_xims[0],
                                                               view_xims[1])
    
    assert(overlap_empty == 0)
    
    return


def test_mv_graph_creation():

    view_xims = _reader.read_mosaic_czi_into_list_of_spatial_xarrays(
        _sample_data.get_sample_data_path())
    
    mv_graph = _mv_graph.build_view_adjacency_graph_from_xims(view_xims)
    
    assert(len(mv_graph.nodes) == len(view_xims))
    assert(len(mv_graph.edges) == 1)

    # import pdb; pdb.set_trace()
    
    return


# def test_get_pairs():

#     view_xims = _reader.read_mosaic_czi_into_list_of_spatial_xarrays(
#         _sample_data.get_sample_data_path())
    
#     mv_graph = _mv_graph.build_view_adjacency_graph_from_xims(view_xims)

#     pairs = _mv_graph.get_registration_pairs_from_overlap_graph(mv_graph,
#                     method='shortest_paths_considering_overlap')
    
#     assert(len(pairs) == 1)

#     pairs = _mv_graph.get_registration_pairs_from_overlap_graph(mv_graph,
#                     method='percentile')
    
#     assert(len(pairs) == 1)
