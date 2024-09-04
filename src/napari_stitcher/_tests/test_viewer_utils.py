import numpy as np
import dask.array as da
import xarray as xr

from multiview_stitcher.io import METADATA_TRANSFORM_KEY
from multiview_stitcher import (
    msi_utils, sample_data, registration, fusion
    )
import pytest

from napari_stitcher import viewer_utils, _reader


@pytest.mark.parametrize(
    "ndim, N_c, N_t", [
        [ndim, N_c, N_t]
        for ndim in [2, 3]
        for N_c in [1, 2]
        for N_t in [1, 2]
    ]
)
def test_create_image_layer_tuples_from_msims(ndim, N_c, N_t, make_napari_viewer):
    """
    Basic test: scroll through time and confirm that no error is thrown.
    """

    viewer = make_napari_viewer()

    tiles_x, tiles_y, tiles_z = 2, 1, 1

    sims = sample_data.generate_tiled_dataset(ndim=ndim, N_t=N_t, N_c=N_c,
            tile_size=5, tiles_x=tiles_x, tiles_y=tiles_y, tiles_z=tiles_z)
    
    msims = [msi_utils.get_msim_from_sim(
        sim,
        scale_factors=[2]) for sim in sims]

    registered_transform_key = 'affine_registered'
    registration.register(
        msims,
        transform_key=METADATA_TRANSFORM_KEY,
        new_transform_key=registered_transform_key,
        reg_channel_index=0
        )
    
    fused = fusion.fuse(
        [msi_utils.get_sim_from_msim(msim) for msim in msims],
        transform_key=registered_transform_key)
    
    mfused = msi_utils.get_msim_from_sim(fused, scale_factors=[2])
    
    lds = viewer_utils.create_image_layer_tuples_from_msims(
        msims, transform_key=registered_transform_key)
    assert len(lds) == N_c * tiles_x * tiles_y * tiles_z
    viewer_utils.add_image_layer_tuples_to_viewer(
        viewer, lds, manage_viewer_transformations=True)

    lds = viewer_utils.create_image_layer_tuples_from_msims(
        [mfused], transform_key=registered_transform_key)
    assert len(lds) == N_c

    viewer_utils.add_image_layer_tuples_to_viewer(
        viewer, lds, manage_viewer_transformations=True)

    # wiggle time
    if N_t > 1:
        current_step = list(viewer.dims.current_step)
        current_step[0] = current_step[0] + 1
        viewer.dims.current_step = tuple(current_step)
