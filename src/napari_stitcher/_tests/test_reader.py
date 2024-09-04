from pathlib import Path

from napari_stitcher import napari_get_reader, _reader

from multiview_stitcher.sample_data import get_mosaic_sample_data_path


def test_read_mosaic_image_into_list_of_spatial_xarrays():

    test_path = get_mosaic_sample_data_path()
    
    view_sims = _reader.read_mosaic_image_into_list_of_spatial_xarrays(test_path)

    assert 2 == len(view_sims)
    assert min([ax in view_sims[0].dims for ax in ['x', 'y']])

    return


# tmp_path is a pytest fixture
def test_reader():
    """An example of how you might test your plugin."""

    test_path = get_mosaic_sample_data_path()

    # try to read it back in
    reader = napari_get_reader(str(test_path))
    assert callable(reader)

    # make sure we're delivering the right format
    layer_data_list = reader(str(test_path))

    assert isinstance(layer_data_list, list) and len(layer_data_list) > 0

    layer_data_tuple = layer_data_list[0]
    assert isinstance(layer_data_tuple, tuple) and len(layer_data_tuple) > 0
