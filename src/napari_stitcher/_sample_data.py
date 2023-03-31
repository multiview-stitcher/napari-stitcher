"""
This module is an example of a barebones sample data provider for napari.

It implements the "sample data" specification.
see: https://napari.org/stable/plugins/guides.html?#sample-data

Replace code below according to your needs.
"""
from __future__ import annotations

import numpy
from pathlib import Path

from napari_stitcher._reader import read_mosaic_czi
import napari_stitcher


def get_sample_data_path():

    sample_path = Path(__file__).parent.parent.parent /\
                             "image-datasets" / "mosaic_test.czi"
    
    return sample_path


def make_sample_data():
    """Generates an image"""
    # Return list of tuples
    # [(data1, add_image_kwargs1), (data2, add_image_kwargs2)]
    # Check the documentation for more information about the
    # add_image_kwargs
    # https://napari.org/stable/api/napari.Viewer.html#napari.Viewer.add_image

    sample_path = get_sample_data_path()

    return read_mosaic_czi([sample_path])
