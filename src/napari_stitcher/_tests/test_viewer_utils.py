import numpy as np
import dask.array as da
import xarray as xr

from napari_stitcher._reader import READER_METADATA_TRANSFORM_KEY

from napari_stitcher import _utils, _sample_data, _msi_utils, _spatial_image_utils, _reader, _viewer_utils


def test_create_image_layer_tuples_from_msims():

    xims = _reader.read_mosaic_image_into_list_of_spatial_xarrays(
        _sample_data.get_sample_data_path())
    
    msims = [_msi_utils.get_msim_from_xim(xim) for xim in xims]

    lds = _viewer_utils.create_image_layer_tuples_from_msims(msims)
    