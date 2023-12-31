import numpy as np
import dask.array as da
import xarray as xr

from multiview_stitcher.io import METADATA_TRANSFORM_KEY
from multiview_stitcher import msi_utils, sample_data

from napari_stitcher import viewer_utils, _reader


def test_create_image_layer_tuples_from_msims():

    sims = _reader.read_mosaic_image_into_list_of_spatial_xarrays(
        sample_data.get_mosaic_sample_data_path())
    
    msims = [msi_utils.get_msim_from_sim(sim) for sim in sims]

    lds = viewer_utils.create_image_layer_tuples_from_msims(msims)
    