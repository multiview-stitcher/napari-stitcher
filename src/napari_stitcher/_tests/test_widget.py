import os
import numpy as np
from pathlib import Path
import tempfile
import tifffile

from napari_stitcher import (
    StitcherQWidget,
    _widget,
    _sample_data,
    viewer_utils,
)

from multiview_stitcher import msi_utils, registration
from multiview_stitcher.io import METADATA_TRANSFORM_KEY
from multiview_stitcher.sample_data import get_mosaic_sample_data_path

import pytest


def test_data_loading_while_plugin_open(make_napari_viewer):

    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()

    test_path = get_mosaic_sample_data_path()

    viewer.open(test_path, plugin='napari-stitcher')

    stitcher_widget = StitcherQWidget(viewer)
    viewer.window.add_dock_widget(stitcher_widget)

    # test also opening when plugin is already open
    viewer.layers.clear()
    viewer.open(test_path, plugin='napari-stitcher')



# make_napari_viewer is a pytest fixture that returns a napari viewer object
# capsys is a pytest fixture that captures stdout and stderr output streams
def test_stitcher_q_widget_integrated(make_napari_viewer, capsys):
    """
    Integration test covering typical pipeline.
    """

    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()

    stitcher_widget = StitcherQWidget(viewer)

    test_path = get_mosaic_sample_data_path()
    
    ndim = 2
    
    viewer.open(test_path, plugin='napari-stitcher')

    stitcher_widget.button_load_layers_all.clicked()

    # Run stitching
    # stitcher_widget.button_stitch.clicked()
    stitcher_widget.run_registration()

    # Check that parameters were obtained
    assert(stitcher_widget.params is not None)

    # Check that parameters are visualised

    # First, view 0 is not shifted
    assert(np.allclose(
        np.eye(ndim + 1),
        viewer.layers[0].affine.affine_matrix[-(ndim+1):, -(ndim+1):]))
    
    # Toggle showing the registrations
    stitcher_widget.visualization_type_rbuttons.value=_widget.CHOICE_REGISTERED

    # Make sure view 0 is shifted now
    assert(~np.allclose(
        np.eye(ndim + 1),
        viewer.layers[1].affine.affine_matrix[-(ndim+1):, -(ndim+1):]))

    # Run fusion
    # stitcher_widget.button_fuse.clicked()
    stitcher_widget.run_fusion()

    # # Make sure layers are created
    # assert(3, len(viewer.layers))
    # assert(True, min(['fused' in l.name for l in viewer.layers]))

    # create our widget, passing in the viewer


    # call our widget method
    # my_widget._on_click()

    # read captured output and check that it's as we expected
    # captured = capsys.readouterr()
    # assert captured.out == "napari has 1 layers\n"


# # make_napari_viewer is a pytest fixture that returns a napari viewer object
# # capsys is a pytest fixture that captures stdout and stderr output streams
# def test_stabilization_workflow(make_napari_viewer, capsys):
#     """
#     Integration test covering stabilization workflow.
#     """

#     # make viewer and add an image layer using our fixture
#     viewer = make_napari_viewer()

#     stitcher_widget = StitcherQWidget(viewer)

#     sims = _sample_data.generate_tiled_dataset(ndim=2, N_t=20, N_c=1, tile_size=30, tiles_x=2, tiles_y=1, tiles_z=1, overlap=3, zoom=10, dtype=np.uint16)
#     layer_tuples = _sample_data.create_image_layer_tuples_from_sims(sims)

#     for lt in layer_tuples:
#         viewer.add_image(lt[0], **lt[1])

#     stitcher_widget.button_load_layers_all.clicked()

#     # Run stabilization
#     stitcher_widget.run_stabilization()

#     # Check that parameters were obtained
#     assert(stitcher_widget.params is not None)

#     # Check that parameters are visualised

#     # # First, view 0 is not shifted
#     # assert(np.allclose(
#     #     # spatial_image_utils.get_data_to_world_matrix_from_spatial_image(viewer.layers[0].data),
#     #     np.eye(viewer.layers[0].data.ndim + 1),
#     #     viewer.layers[0].affine.affine_matrix))
    
#     # Toggle showing the registrations
#     stitcher_widget.visualization_type_rbuttons.value=_widget.CHOICE_REGISTERED
#     # Make sure view 0 is shifted now
#     assert(~np.allclose(
#         # spatial_image_utils.get_data_to_world_matrix_from_spatial_image(viewer.layers[0].data),
#         np.eye(viewer.layers[0].data.ndim + 1),
#         viewer.layers[0].affine.affine_matrix))

#     # Run fusion
#     # stitcher_widget.button_fuse.clicked()
#     stitcher_widget.run_fusion()


@pytest.mark.parametrize(
    "ndim, overlap, N_c, N_t, dtype", [
        (2, 1, 1, 3, np.uint16), # single pixel overlap not supported
        (2, 5, 1, 3, np.uint16),
        (2, 5, 1, 3, np.uint8),
        # (2, 5, 2, 3, np.uint8), # sporadically fails, need to investigate
        # (3, 5, 2, 3, np.uint16),
        (3, 1, 1, 3, np.uint8),
        (3, 5, 1, 3, np.uint8),
    ]
)
def test_diversity_stitching(ndim, overlap, N_c, N_t, dtype, make_napari_viewer):

    viewer = make_napari_viewer()

    wdg = StitcherQWidget(viewer)
    viewer.window.add_dock_widget(wdg)

    sims = _sample_data.generate_tiled_dataset(ndim=ndim, N_t=N_t, N_c=N_c,
            tile_size=30, tiles_x=2, tiles_y=1, tiles_z=1, overlap=overlap, zoom=10, dtype=dtype)

    msims = [msi_utils.get_msim_from_sim(sim, scale_factors=[]) for sim in sims]
    layer_tuples = viewer_utils.create_image_layer_tuples_from_msims(
        msims, transform_key=METADATA_TRANSFORM_KEY)
   
    for lt in layer_tuples:
        viewer.add_image(lt[0], **lt[1])

    wdg.button_load_layers_all.clicked()

    # Run registration
    if overlap > 1:
        wdg.run_registration()
    else:
        with pytest.raises(registration.NotEnoughOverlapError):
            wdg.run_registration()
        return

    # Run fusion
    wdg.run_fusion()

    # test scrolling
    if N_t > 1:
        current_step = list(viewer.dims.current_step)
        current_step[0] = 1
        viewer.dims.current_step = tuple(current_step)

    with tempfile.TemporaryDirectory() as tmpdir:
        outfile = str(Path(tmpdir) / "test.tif")
        viewer.layers[-N_c:].save(outfile, plugin='napari-stitcher')
        tifffile.imread(outfile)


def test_time_slider(make_napari_viewer):
    """
    Register and fuse only 3 out of 4 time points
    present in the input layers.
    """

    viewer = make_napari_viewer()

    wdg = StitcherQWidget(viewer)
    viewer.window.add_dock_widget(wdg)

    sims = _sample_data.generate_tiled_dataset(
        ndim=2, N_t=4, N_c=1,
        tile_size=30, tiles_x=2, tiles_y=1, tiles_z=1,
        overlap=5, zoom=10, dtype=np.uint8)

    msims = [msi_utils.get_msim_from_sim(sim, scale_factors=[]) for sim in sims]
    layer_tuples = viewer_utils.create_image_layer_tuples_from_msims(
        msims, transform_key=METADATA_TRANSFORM_KEY)
   
    for lt in layer_tuples:
        viewer.add_image(lt[0], **lt[1])
    
    wdg.button_load_layers_all.clicked()

    wdg.times_slider.min=-1
    wdg.times_slider.max=2

    # Run stitching
    wdg.run_registration()

    # Run fusion
    wdg.run_fusion()


@pytest.mark.parametrize(
    "ndim, N_c, N_t", [
        # (2, 1, 1),
        (2, 1, 2),
        # (3, 2, 1),
        (3, 2, 2),
    ]
)
def test_update_transformations(ndim, N_c, N_t, make_napari_viewer):
    """
    Basic test: scroll through time and confirm that no error is thrown.
    """

    viewer = make_napari_viewer()

    wdg = StitcherQWidget(viewer)
    viewer.window.add_dock_widget(wdg)

    sims = _sample_data.generate_tiled_dataset(ndim=ndim, N_t=N_t, N_c=N_c,
            tile_size=5, tiles_x=2, tiles_y=1, tiles_z=1)
    
    msims = [msi_utils.get_msim_from_sim(sim) for sim in sims]

    layer_tuples = viewer_utils.create_image_layer_tuples_from_msims(
        msims, positional_cmaps=False, transform_key=METADATA_TRANSFORM_KEY)

    for lt in layer_tuples:
        viewer.add_image(lt[0], **lt[1])

    # scroll in time
    current_step = list(viewer.dims.current_step)
    current_step[0] = current_step[0] + 1
    viewer.dims.current_step = tuple(current_step)


def test_fusion_without_registration(make_napari_viewer):

    viewer = make_napari_viewer()

    stitcher_widget = StitcherQWidget(viewer)
    test_path = get_mosaic_sample_data_path()
        
    viewer.open(test_path, plugin='napari-stitcher')

    stitcher_widget.button_load_layers_all.clicked()

    # Run stitching
    stitcher_widget.run_fusion()
    assert(len(viewer.layers) == 3)


def test_vanilla_layers_2D_no_time(make_napari_viewer):

    viewer = make_napari_viewer()

    D = 100
    im = np.random.random((D, D))

    im1 = im[:, :D//2+D//10]
    im2 = im[:, D//2-D//10:]

    viewer.add_image(im1, translate=(0, 0), name='im1')
    viewer.add_image(im2, translate=(0, D//2-D//10-5), name='im2')

    wdg = StitcherQWidget(viewer)
    viewer.window.add_dock_widget(wdg)

    wdg.button_load_layers_all.clicked()
    wdg.run_registration()
    wdg.run_fusion()
