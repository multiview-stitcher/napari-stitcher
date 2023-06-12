from napari_stitcher import _sample_data

# add your tests here...


def test_generate_tiled_dataset():

    N_c = 2
    for ndim in [2, 3]:

        xims = _sample_data.generate_tiled_dataset(ndim=ndim, N_c=2, N_t=4)
        computed = [xim.compute() for xim in xims]

        assert(xims[0].data.ndim == ndim + int(N_c > 1) + 1)
