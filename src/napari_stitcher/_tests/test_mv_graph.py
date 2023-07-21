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
        ndim=ndim, overlap=overlap, N_c=2, N_t=1,
        tile_size=15, tiles_x=3, tiles_y=3, tiles_z=3,
        spacing_x=spacing_x, spacing_y=spacing_y, spacing_z=spacing_z)
    
    xims = [xim.sel(t=0) for xim in xims]
    
    overlap_areas = []
    for ixim1, xim1 in enumerate(xims):
        for ixim2, xim2 in enumerate(xims):
            overlap_area, _ = _mv_graph.get_overlap_between_pair_of_xims(xim1, xim2, transform_key='affine_metadata')
            overlap_areas.append(overlap_area)

    overlap_areas = np.array(overlap_areas).reshape((len(xims), len(xims)))

    unique_overlap_areas = np.unique(overlap_areas)

    # remove duplicate values (because of float comparison)
    unique_overlap_areas_filtered = list(unique_overlap_areas.copy())

    for uoa in unique_overlap_areas:
        if len(np.where(np.abs(uoa - np.array(unique_overlap_areas_filtered)) < spacing_x / 10.)[0]) > 1:
            unique_overlap_areas_filtered.remove(uoa)

    unique_overlap_areas_filtered = np.array(unique_overlap_areas_filtered)

    if overlap == 0:

        assert(len(unique_overlap_areas) == 1+1)
        assert(overlap_areas[0][1] == -1)

    else:
        if ndim == 2:
            if overlap == 1:
                assert(len(unique_overlap_areas_filtered) == 3)
            else:
                assert(len(unique_overlap_areas_filtered) == 4)
        elif ndim == 3:
            if overlap == 1:
                assert(len(unique_overlap_areas_filtered) == 3)
            else:
                assert(len(unique_overlap_areas_filtered) == 5)

        assert(np.min(overlap_areas) == -1)
        assert(np.max(overlap_areas) > 0)
        if overlap == 1:
            assert(0 in overlap_areas)

    return


def test_mv_graph_creation():

    view_xims = _reader.read_mosaic_image_into_list_of_spatial_xarrays(
        _sample_data.get_sample_data_path())
    
    mv_graph = _mv_graph.build_view_adjacency_graph_from_xims(view_xims)
    
    assert(len(mv_graph.nodes) == len(view_xims))
    assert(len(mv_graph.edges) == 1)
    
    return


def test_get_intersection_polygon_from_pair_of_xims_2D():

    view_xims = _reader.read_mosaic_image_into_list_of_spatial_xarrays(
        _sample_data.get_sample_data_path())
    
    intersection_polygon = _mv_graph.get_intersection_polygon_from_pair_of_xims_2D(
        view_xims[0], view_xims[1],
        transform_key='affine_metadata')
    
    assert(intersection_polygon.area() > 0)

    
                                                         
    