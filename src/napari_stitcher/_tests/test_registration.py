import numpy as np

from napari_stitcher import _registration, _sample_data, _spatial_image_utils, _reader, _mv_graph


def test_pairwise():

    layers = _sample_data.make_sample_data()

    xims = [l[0] for l in layers]

    spatial_dims = _spatial_image_utils.get_spatial_dims_from_xim(xims[0])

    pd = _registration.register_pair_of_spatial_images(xims,
            registration_binning={dim: 4 for dim in spatial_dims})

    p = pd.compute(scheduler='single-threaded')

    assert np.allclose(p,
        np.array([[1.        , 0.        , 1.73333333],
                  [0.        , 1.        , 7.36666667],
                  [0.        , 0.        , 1.        ]]))


def test_register_graph():

    view_xims = _reader.read_mosaic_czi_into_list_of_spatial_xarrays(
        _sample_data.get_sample_data_path())
    
    g = _mv_graph.build_view_adjacency_graph_from_xims(view_xims)

    gd = _registration.register_graph(g)
    