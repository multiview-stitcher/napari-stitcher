import numpy as np
import xarray as xr
import dask.array as da

import tempfile
from pathlib import Path

import tifffile

from napari_stitcher import _sample_data, viewer_utils

from ngff_stitcher import msi_utils
from ngff_stitcher.io import METADATA_TRANSFORM_KEY


def test_writer_napari(make_napari_viewer):

    viewer = make_napari_viewer()

    spacing_xy = 0.5
    im_dtype = np.uint8

    for field_ndim in [2, 3]:
        for times in [[0], [0, 1]]:
            for channels in [
                [0],
                [0, 1]]:
                
                sims = _sample_data.generate_tiled_dataset(
                    ndim=field_ndim, N_t=len(times), N_c=len(channels),
                    tiles_x=1, tiles_y=1, tiles_z=1, dtype=im_dtype,
                    spacing_x=spacing_xy, spacing_y=spacing_xy,
                )

                msims = [msi_utils.get_msim_from_xim(sim) for sim in sims]

                full_layer_data_list = viewer_utils.create_image_layer_tuples_from_msims(
                    msims,
                    positional_cmaps=False,
                    transform_key=METADATA_TRANSFORM_KEY,
                )

                viewer.layers.clear()
                for l in full_layer_data_list:
                    viewer.add_image(l[0], **l[1])
                
                with tempfile.TemporaryDirectory() as tmpdir:

                    filepath = str(Path(tmpdir) / "test.tif")
                    viewer.layers.save(filepath, plugin='napari-stitcher')

                    read_im = tifffile.imread(filepath)

                    # make sure dimensionality is right
                    assert(read_im.ndim == field_ndim + int(len(times) > 1) + int(len(channels) > 1))

                    # test metadata

                    # https://pypi.org/project/tifffile/#examples
                    tif = tifffile.TiffFile(filepath)
                    assert(tif.series[0].axes,
                           ['', 't'][len(times) > 1] +\
                           ['', 'z'][field_ndim > 2] +\
                           ['', 'c'][len(channels) > 1] +\
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
