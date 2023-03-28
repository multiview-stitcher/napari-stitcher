import numpy as np
import tempfile
from pathlib import Path

import tifffile

from napari_stitcher import write_multiple


def create_full_layer_data_list(channels=[0],
                                times=[0],
                                field_ndim=2,
                                dtype=np.uint8,
                                spacing_xy=0.5):
    """
    Create test data for writer tests.
    Returns a List[FullLayerData].
    """

    view_dict = {}
    for view in range(1):
        view_dict[view] = {"physical_pixel_sizes": [spacing_xy] * field_ndim}

    test_im = np.random.randint(0, 100, [len(times)] + [7] * field_ndim).astype(dtype)

    full_layer_data_list = []

    for ch in channels:
        full_layer_data_list.append([test_im,
                                    {'metadata':
                                        {'view':-1,
                                        "view_dict": view_dict,
                                        'stack_props': {'spacing': [0.1] * field_ndim},
                                        'times': times}},
                                    'image1_ch%03d' %ch])
        
    return full_layer_data_list


def test_writer_general():

    spacing_xy = 0.5
    im_dtype = np.uint8

    for field_ndim in [2]:
        for times in [[0], [0, 1]]:
            for channels in [[0], [0, 1]]:
                full_layer_data_list = create_full_layer_data_list(channels=channels,
                                                                times=times,
                                                                field_ndim=field_ndim,
                                                                dtype=im_dtype,
                                                                spacing_xy=spacing_xy)
                
                with tempfile.TemporaryDirectory() as tmpdir:

                    filepath = str(Path(tmpdir) / "test.tif")

                    write_multiple(filepath, full_layer_data_list[:])

                    read_im = tifffile.imread(filepath)

                    # make sure dimensionality is right
                    assert(read_im.ndim == field_ndim + int(len(times) > 1) + int(len(channels) > 1))

                    # test metadata

                    # https://pypi.org/project/tifffile/#examples
                    tif = tifffile.TiffFile(filepath)
                    assert(tif.series[0].axes,
                           ['', 'T'][len(times) > 1] +\
                           ['', 'Z'][field_ndim > 2] +\
                           ['', 'C'][len(channels) > 1] +\
                           'YX')
                    
                    resolution_unit_checked = False
                    resolution_value_checked = False
                    bitspersample_checked = False

                    p = tif.pages[0]
                    for tag in p.tags:
                        print(tag.name, '/', tag.value)
                        if tag.name == 'ResolutionUnit':
                            assert(tag.value, 'um')
                            resolution_unit_checked = True
                        if tag.name == 'XResolution':
                            assert(np.isclose(spacing_xy, tag.value[1] / tag.value[0]))
                            resolution_value_checked = True
                        if tag.name == 'BitsPerSample':
                            assert(tag.value, np.iinfo(im_dtype).bits)
                            bitspersample_checked = True

                    assert(resolution_unit_checked)
                    assert(resolution_value_checked)
                    assert(bitspersample_checked)
                    
                    # import pdb; pdb.set_trace()
