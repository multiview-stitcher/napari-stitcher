from pathlib import Path

from napari_stitcher import napari_get_reader, _reader


def test_read_mosaic_image_into_list_of_spatial_xarrays():

    test_path = str(Path(__file__).parent.parent.parent.parent /\
                             "image-datasets" / "mosaic_test.czi")
    
    view_xims = _reader.read_mosaic_image_into_list_of_spatial_xarrays(test_path)

    assert(2, len(view_xims))
    assert(min([ax in view_xims[0].dims for ax in ['x', 'y']]))

    return


# tmp_path is a pytest fixture
def test_reader():
    """An example of how you might test your plugin."""

    test_path = str(Path(__file__).parent.parent.parent.parent /\
                             "image-datasets" / "mosaic_test.czi")

    # try to read it back in
    reader = napari_get_reader(test_path)
    assert callable(reader)

    # make sure we're delivering the right format
    layer_data_list = reader(test_path)

    assert isinstance(layer_data_list, list) and len(layer_data_list) > 0

    layer_data_tuple = layer_data_list[0]
    assert isinstance(layer_data_tuple, tuple) and len(layer_data_tuple) > 0

    # make sure it's the same as it started
    # np.testing.assert_allclose(original_data, layer_data_tuple[0])


# def test_get_reader_pass():
#     reader = napari_get_reader("fake.file")
#     assert reader is None
