import numpy as np
from napari_stitcher import _spatial_image_utils, _reader, _sample_data


def test_get_data_to_world_matrix_from_spatial_image():

    sample_fn = _sample_data.get_sample_data_path()

    xims = _reader.read_mosaic_czi_into_list_of_spatial_xarrays(sample_fn)

    M = _spatial_image_utils.get_data_to_world_matrix_from_spatial_image(xims[1])

    assert M.shape == (3, 3)
    assert np.allclose(M[0,0], 1.08333333)
    assert np.allclose(M[1,2], 901.3333333333333)
    assert np.allclose(M[2,1], 0)

    xim1_rebuilt = _spatial_image_utils.assign_si_coords_from_params(
                                            xims[1].copy(), M)

    assert xims[1].dims == xim1_rebuilt.dims
    assert np.allclose(xim1_rebuilt.attrs['direction'], np.eye(3))

    for dim in xims[1].dims:
        assert(min(xims[1].coords[dim] == xim1_rebuilt.coords[dim]))
    
    M_rebuilt = _spatial_image_utils.get_data_to_world_matrix_from_spatial_image(xims[1])

    assert np.allclose(M, M_rebuilt)
