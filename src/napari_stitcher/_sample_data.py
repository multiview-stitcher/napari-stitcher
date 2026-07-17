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

from multiview_stitcher.sample_data import get_mosaic_sample_data_path
from multiview_stitcher.io import METADATA_TRANSFORM_KEY
from multiview_stitcher.sample_data import generate_tiled_dataset
from multiview_stitcher.msi_utils import get_msim_from_sim
from multiview_stitcher import msi_utils, spatial_image_utils

_DIGIT_SEGMENTS = {
    0: (
        "top",
        "upper_left",
        "upper_right",
        "lower_left",
        "lower_right",
        "bottom",
    ),
    1: ("upper_right", "lower_right"),
    2: ("top", "upper_right", "middle", "lower_left", "bottom"),
    3: ("top", "upper_right", "middle", "lower_right", "bottom"),
    4: ("upper_left", "upper_right", "middle", "lower_right"),
    5: ("top", "upper_left", "middle", "lower_right", "bottom"),
    6: ("top", "upper_left", "middle", "lower_left", "lower_right", "bottom"),
    7: ("top", "upper_right", "lower_right"),
    8: (
        "top",
        "upper_left",
        "upper_right",
        "middle",
        "lower_left",
        "lower_right",
        "bottom",
    ),
    9: ("top", "upper_left", "upper_right", "middle", "lower_right", "bottom"),
}


def _draw_segment(image, segment, value):
    """Draw one segment of a seven-segment digit into ``image``."""
    size = image.shape[0]
    margin = size // 6
    thickness = max(size // 12, 3)
    y_mid = size // 2
    y_bottom = size - margin - thickness
    x_right = size - margin - thickness

    segments = {
        "top": (
            slice(margin, margin + thickness),
            slice(margin, size - margin),
        ),
        "middle": (
            slice(y_mid - thickness // 2, y_mid + (thickness + 1) // 2),
            slice(margin, size - margin),
        ),
        "bottom": (
            slice(y_bottom, y_bottom + thickness),
            slice(margin, size - margin),
        ),
        "upper_left": (
            slice(margin, y_mid),
            slice(margin, margin + thickness),
        ),
        "upper_right": (
            slice(margin, y_mid),
            slice(x_right, x_right + thickness),
        ),
        "lower_left": (
            slice(y_mid, y_bottom + thickness),
            slice(margin, margin + thickness),
        ),
        "lower_right": (
            slice(y_mid, y_bottom + thickness),
            slice(x_right, x_right + thickness),
        ),
    }

    image[segments[segment]] = value


def _digit_image(digit, size, seed):
    """Create a noisy square image with a large seven-segment digit."""
    rng = np.random.default_rng(seed)
    image = rng.integers(5, 70, size=(size, size), dtype=np.uint8)

    for segment in _DIGIT_SEGMENTS[digit % 10]:
        _draw_segment(image, segment, value=240)

    return image


def _multiscale_digit_msim(origin, level_images, tile_size=500):
    """Build one prepositioned multiscale tile from per-level digit images."""
    sims = []
    for image in level_images:
        pixel_size = tile_size / image.shape[0]
        sim = spatial_image_utils.get_sim_from_array(
            image,
            dims=["y", "x"],
            scale={"y": pixel_size, "x": pixel_size},
            translation=origin,
            c_coords=["digits"],
            t_coords=[0],
            transform_key=METADATA_TRANSFORM_KEY,
        )
        sims.append(sim)

    return msi_utils.get_msim_from_sims(sims)


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

    sims = generate_tiled_dataset(
        ndim=2, N_t=20, N_c=1,
        tile_size=30, tiles_x=3, tiles_y=3, tiles_z=1,
        drift_scale=2., shift_scale=2.,
        overlap=0, zoom=8, dtype=np.uint8)
    
    msims = [get_msim_from_sim(sim) for sim in sims]

    layer_tuples = viewer_utils.create_image_layer_tuples_from_msims(
        msims, transform_key=METADATA_TRANSFORM_KEY)

    return layer_tuples


def timelapse_with_stage_shifts_with_overlap_3d():

    sims = generate_tiled_dataset(
        ndim=3, N_t=20, N_c=1,
        tile_size=30, tiles_x=3, tiles_y=3, tiles_z=1,
        drift_scale=0., shift_scale=2.,
        overlap=3, zoom=8, dtype=np.uint8)
    
    msims = [get_msim_from_sim(sim) for sim in sims]

    layer_tuples = viewer_utils.create_image_layer_tuples_from_msims(
        msims, transform_key=METADATA_TRANSFORM_KEY)

    return layer_tuples


def multiscale_digits():
    """Four 500x500 tiles whose pyramid levels contain scale-index digits."""
    tile_size = 500
    level_sizes = [500, 250, 125]
    level_images = [
        _digit_image(digit=iscale, size=size, seed=iscale)
        for iscale, size in enumerate(level_sizes)
    ]

    msims = []
    for tile_y, tile_x in np.ndindex(2, 2):
        origin = {"y": tile_y * tile_size, "x": tile_x * tile_size}
        msims.append(_multiscale_digit_msim(origin, level_images, tile_size))

    layer_tuples = viewer_utils.create_image_layer_tuples_from_msims(
        msims,
        positional_cmaps=False,
        name_prefix="multiscale_digits_tile",
        transform_key=METADATA_TRANSFORM_KEY,
        contrast_limits=[0, 255],
    )

    # Keep the tiles visually identical; only their grid positions differ.
    for _, add_image_kwargs, _ in layer_tuples:
        add_image_kwargs["colormap"] = "gray"
        add_image_kwargs["gamma"] = 1.0

    return layer_tuples
