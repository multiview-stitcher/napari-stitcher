import numpy as np
import dask.array as da
import xarray as xr

from scipy import ndimage

from napari_stitcher._reader import READER_METADATA_TRANSFORM_KEY

from napari_stitcher import _registration, _sample_data, _spatial_image_utils, _reader, _mv_graph

import pytest


def test_pairwise():

    # layers = _sample_data.make_sample_data()
    # xims = [l[0] for l in layers]

    example_data_path = _sample_data.get_sample_data_path()
    xims = _reader.read_mosaic_image_into_list_of_spatial_xarrays(example_data_path)

    xims = [_spatial_image_utils.xim_sel_coords(xim,
                {
                    # 't': xim.coords['t'][0],
                    'c': xim.coords['c'][0],
                    })

            for xim in xims]

    spatial_dims = _spatial_image_utils.get_spatial_dims_from_xim(xims[0])

    pd = _registration.register_pair_of_xims_over_time(xims[0], xims[1],
            registration_binning={dim: 1 for dim in spatial_dims},
            transform_key='affine_metadata',
    )

    # p = pd.compute(scheduler='single-threaded')
    p = pd.compute()

    assert np.allclose(p.sel(t=0),
        np.array([[ 1.        ,  0.        , -1.95      ],
                  [ 0.        ,  1.        , -7.58333333],
                  [ 0.        ,  0.        ,  1.        ]]))
    

@pytest.mark.parametrize(
    "ndim", [2, 3]
)
def test_register_with_single_pixel_overlap(ndim):

    xims = _sample_data.generate_tiled_dataset(
            ndim=ndim, overlap=1, N_c=2, N_t=2,
            tile_size=10, tiles_x=1, tiles_y=2, tiles_z=1,
            spacing_x=1, spacing_y=1, spacing_z=1)
    
    # _registration.register_pair_of_spatial_images(xims[0], xims[1])
    _registration.register_pair_of_xims_over_time(xims[0], xims[1])


def test_register_graph():

    view_xims = _reader.read_mosaic_image_into_list_of_spatial_xarrays(
        _sample_data.get_sample_data_path())
    
    view_xims = [xim.sel(c=xim.coords['c'][0]) for xim in view_xims]
    
    g = _mv_graph.build_view_adjacency_graph_from_xims(view_xims)

    # g_pairs = _registration.get_registration_pair_graph(g)
    g_reg = _registration.get_registration_graph_from_overlap_graph(g)

    assert(max(['transform' in g_reg.edges[e].keys()
        for e in g_reg.edges]))
    
    assert([type(g_reg.edges[e]['transform'].data) == da.core.Array
        for e in g_reg.edges if 'transform' in g_reg.edges[e].keys()])
    
    g_reg_computed = _mv_graph.compute_graph_edges(g_reg, scheduler='single-threaded')
    
    assert([type(g_reg_computed.edges[e]['transform'].data) == np.ndarray
        for e in g_reg_computed.edges if 'transform' in g_reg_computed.edges[e].keys()])

    # get node parameters
    g_reg_nodes = _registration.get_node_params_from_reg_graph(g_reg_computed)

    assert(['transforms' in g_reg_nodes.nodes[n].keys()
            for n in g_reg_nodes.nodes])
    

def test_get_stabilization_parameters():

    for ndim in [2, 3]:

        N_t = 10

        im = np.random.randint(0, 100, (5,) * ndim, dtype=np.uint16)
        im = ndimage.zoom(im, [10] * ndim, order=1)

        # simulate random stage drifts
        shifts = (np.random.random((N_t, ndim)) - 0.5) * 30

        # simulate drift
        drift = np.cumsum(np.array([[10] * ndim] * N_t), axis=0)

        tl = []
        for t in range(N_t):
            tl.append(ndimage.shift(im, shifts[t] + drift[t], order=1, mode='reflect'))
        tl = np.array(tl)

        params_da = _registration.get_stabilization_parameters(tl, sigma=1)
        params = params_da.compute()

        assert len(params) == N_t

        assert(np.all(np.abs(
            np.mean(np.diff(shifts, axis=0) - np.diff(params, axis=0), axis=0)\
            < np.std((shifts))) / 3))


def test_get_optimal_registration_binning():

    ndim = 3
    xims = [xr.DataArray(da.empty(([1000] * ndim)), dims=_spatial_image_utils.SPATIAL_DIMS)
            for _ in range(2)]
    reg_binning = _registration.get_optimal_registration_binning(*tuple(xims))

    assert(min(reg_binning.values()) > 1)
