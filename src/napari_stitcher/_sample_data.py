"""
This module is an example of a barebones sample data provider for napari.

It implements the "sample data" specification.
see: https://napari.org/stable/plugins/guides.html?#sample-data

Replace code below according to your needs.
"""
from __future__ import annotations

import numpy as np

from pathlib import Path

from napari_stitcher import viewer_utils
from napari_stitcher._reader import read_mosaic

from ngff_stitcher.sample_data import get_mosaic_sample_data_path
from ngff_stitcher.io import METADATA_TRANSFORM_KEY
from ngff_stitcher.sample_data import generate_tiled_dataset
from ngff_stitcher.msi_utils import get_msim_from_xim


def make_sample_data():
    """Generates an image"""
    # Return list of tuples
    # [(data1, add_image_kwargs1), (data2, add_image_kwargs2)]
    # Check the documentation for more information about the
    # add_image_kwargs
    # https://napari.org/stable/api/napari.Viewer.html#napari.Viewer.add_image

    sample_path = get_mosaic_sample_data_path()

    return read_mosaic([sample_path])


def drifting_timelapse_with_stage_shifts_no_overlap_2d():

    xims = generate_tiled_dataset(
        ndim=2, N_t=20, N_c=1,
        tile_size=30, tiles_x=3, tiles_y=3, tiles_z=1,
        drift_scale=2., shift_scale=2.,
        overlap=0, zoom=8, dtype=np.uint8)
    
    msims = [get_msim_from_xim(xim) for xim in xims]

    layer_tuples = viewer_utils.create_image_layer_tuples_from_msims(
        msims, transform_key=METADATA_TRANSFORM_KEY)

    return layer_tuples


def timelapse_with_stage_shifts_with_overlap_3d():

    xims = generate_tiled_dataset(
        ndim=3, N_t=20, N_c=1,
        tile_size=30, tiles_x=3, tiles_y=3, tiles_z=1,
        drift_scale=0., shift_scale=2.,
        overlap=3, zoom=8, dtype=np.uint8)
    
    msims = [get_msim_from_xim(xim) for xim in xims]

    layer_tuples = viewer_utils.create_image_layer_tuples_from_msims(
        msims, transform_key=METADATA_TRANSFORM_KEY)

    return layer_tuples
