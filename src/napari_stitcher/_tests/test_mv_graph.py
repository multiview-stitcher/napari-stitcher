import numpy as np

from napari_stitcher import _mv_graph, _sample_data, _reader, _spatial_image_utils

import pytest


@pytest.mark.parametrize(
    "ndim, overlap", [
        (ndim, overlap) for ndim in [2, 3] for overlap in [0, 1, 3]
    ]
)
def test_overlap(ndim, overlap):

    spacing_x=0.5
    spacing_y=0.5
    spacing_z=2
    xims = _sample_data.generate_tiled_dataset(
        ndim=ndim, overlap=overlap, N_c=2, N_t=2,
        tile_size=15, tiles_x=3, tiles_y=3, tiles_z=3,
        spacing_x=spacing_x, spacing_y=spacing_y, spacing_z=spacing_z)
    
    overlap_areas = []
    for ixim1, xim1 in enumerate(xims):
        for ixim2, xim2 in enumerate(xims):
            overlap_area, overlap_coords = _mv_graph.get_overlap_between_pair_of_xims(xim1, xim2)
            overlap_areas.append(overlap_area)

    overlap_areas = np.array(overlap_areas).reshape((len(xims), len(xims)))

    if overlap == 0:

        assert(len(np.unique(overlap_areas)) == 1+1)
        assert(overlap_areas[0][1] == -1)

    else:
        if ndim == 2:
            if overlap == 1:
                assert(len(np.unique(overlap_areas)) == 3)
            else:
                assert(len(np.unique(overlap_areas)) == 4)
        elif ndim == 3:
            if overlap == 1:
                assert(len(np.unique(overlap_areas)) == 3)
            else:
                assert(len(np.unique(overlap_areas)) == 5)

        assert(np.min(overlap_areas) == -1)
        assert(np.max(overlap_areas) > 0)
        if overlap == 1:
            assert(0 in overlap_areas)

    return


def test_mv_graph_creation():

    view_xims = _reader.read_mosaic_czi_into_list_of_spatial_xarrays(
        _sample_data.get_sample_data_path())
    
    mv_graph = _mv_graph.build_view_adjacency_graph_from_xims(view_xims)
    
    assert(len(mv_graph.nodes) == len(view_xims))
    assert(len(mv_graph.edges) == 1)
    
    return
