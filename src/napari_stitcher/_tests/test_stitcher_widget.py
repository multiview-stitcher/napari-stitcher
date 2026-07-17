import os
import numpy as np
from unittest.mock import patch
from pathlib import Path
import tempfile
import tifffile

from napari_stitcher import (
    StitcherQWidget,
    _stitcher_widget,
    viewer_utils,
)

from multiview_stitcher import (
    misc_utils, msi_utils, mv_graph, spatial_image_utils)
from multiview_stitcher.io import METADATA_TRANSFORM_KEY
from multiview_stitcher.sample_data import (
    get_mosaic_sample_data_path, generate_tiled_dataset)

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



def test_stitcher_q_widget_integrated(make_napari_viewer):
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
    assert stitcher_widget.params is not None

    # Check that parameters are visualised

    # First, view 0 is not shifted
    assert np.allclose(
        np.eye(ndim + 1),
        viewer.layers[0].affine.affine_matrix[-(ndim+1):, -(ndim+1):])
    
    # Toggle showing the registrations
    stitcher_widget.visualization_type_rbuttons.value=_stitcher_widget.CHOICE_REGISTERED

    # Make sure view 0 is shifted now
    assert not np.allclose(
        np.eye(ndim + 1),
        viewer.layers[1].affine.affine_matrix[-(ndim+1):, -(ndim+1):])

    # Run fusion
    # stitcher_widget.button_fuse.clicked()
    stitcher_widget.run_fusion()


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

    sims = generate_tiled_dataset(ndim=ndim, N_t=N_t, N_c=N_c,
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
        with pytest.raises(mv_graph.NotEnoughOverlapError):
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

    sims = generate_tiled_dataset(
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

    sims = generate_tiled_dataset(ndim=ndim, N_t=N_t, N_c=N_c,
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
    assert len(viewer.layers) == 3

    #check that fusion can also be run twice
    stitcher_widget.run_fusion()


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


def test_load_layers_filters_non_image_layers(make_napari_viewer):
    viewer = make_napari_viewer()

    # Add image and non-image layers
    viewer.add_image(np.random.random((10, 10)), name='image_layer')
    viewer.add_points(np.random.random((10, 2)), name='points_layer')

    wdg = StitcherQWidget(viewer)
    viewer.window.add_dock_widget(wdg)

    wdg.button_load_layers_all.clicked()

    # Check if only image layer is loaded
    assert len(wdg.input_layers) == 1
    assert wdg.input_layers[0].name == 'image_layer'


def test_fuse_register_buttons_not_grayed_out(make_napari_viewer):
    viewer = make_napari_viewer()

    # Add image and labels layers
    viewer.add_image(np.random.random((10, 10)), name='image_layer')
    viewer.add_labels(np.random.randint(0, 2, (10, 10)), name='labels_layer')

    wdg = StitcherQWidget(viewer)
    viewer.window.add_dock_widget(wdg)

    wdg.button_load_layers_all.clicked()

    # Check if the buttons are not grayed out
    assert wdg.button_stitch.enabled
    assert wdg.button_fuse.enabled


def test_manual_transform_pre_registration(make_napari_viewer):
    """
    Pre-registration: manual layer adjustments should be captured into
    affine_metadata when _capture_layer_transforms_to_msims is called
    (as happens at the start of run_registration / run_fusion).
    """
    viewer = make_napari_viewer()
    wdg = StitcherQWidget(viewer)
    viewer.window.add_dock_widget(wdg)

    ndim = 2
    sims = generate_tiled_dataset(
        ndim=ndim, N_t=1, N_c=1,
        tile_size=30, tiles_x=2, tiles_y=1, tiles_z=1,
        overlap=5, zoom=10, dtype=np.uint8)
    msims = [msi_utils.get_msim_from_sim(sim, scale_factors=[]) for sim in sims]
    layer_tuples = viewer_utils.create_image_layer_tuples_from_msims(
        msims, transform_key=METADATA_TRANSFORM_KEY)
    for lt in layer_tuples:
        viewer.add_image(lt[0], **lt[1])

    wdg.button_load_layers_all.clicked()

    # Manually move the first tile in the viewer (simulating the transform tool)
    l = wdg.input_layers[0]
    original_affine = np.array(l.affine.affine_matrix).copy()
    custom_affine = original_affine.copy()
    custom_affine[-(ndim), -1] += 25   # shift in last spatial dim

    # Setting affine while _updating_viewer=False simulates a user drag
    wdg._updating_viewer = False
    l.affine = custom_affine

    # Directly call _capture (run_registration calls this when not in CHOICE_REGISTERED)
    wdg._capture_layer_transforms_to_msims()

    sim = msi_utils.get_sim_from_msim(wdg.msims[l.name])
    stored = np.array(
        spatial_image_utils.get_affine_from_sim(sim, 'affine_metadata').isel(t=0)
    ).squeeze()
    expected = custom_affine[-(ndim + 1):, -(ndim + 1):]
    assert np.allclose(stored, expected), \
        f"affine_metadata should reflect the manual adjustment.\nExpected:\n{expected}\nGot:\n{stored}"


@pytest.mark.parametrize("ndim", [2, 3])
def test_manual_registered_transform(ndim, make_napari_viewer):
    """
    Post-registration with Show=Registered: manually moving a layer should
    update affine_registered for the CURRENT timepoint only; all other
    timepoints must remain unchanged.
    """
    N_t = 3
    viewer = make_napari_viewer()
    wdg = StitcherQWidget(viewer)
    viewer.window.add_dock_widget(wdg)

    sims = generate_tiled_dataset(
        ndim=ndim, N_t=N_t, N_c=1,
        tile_size=30, tiles_x=2, tiles_y=1, tiles_z=1,
        overlap=5, zoom=10, dtype=np.uint8)
    msims = [msi_utils.get_msim_from_sim(sim, scale_factors=[]) for sim in sims]
    layer_tuples = viewer_utils.create_image_layer_tuples_from_msims(
        msims, transform_key=METADATA_TRANSFORM_KEY)
    for lt in layer_tuples:
        viewer.add_image(lt[0], **lt[1])

    wdg.button_load_layers_all.clicked()
    wdg.run_registration()
    wdg.visualization_type_rbuttons.value = _stitcher_widget.CHOICE_REGISTERED

    # Use the second tile (index 1) – it typically has a non-identity registered transform
    l = wdg.input_layers[1]
    sim = msi_utils.get_sim_from_msim(wdg.msims[l.name])
    original_params = spatial_image_utils.get_affine_from_sim(
        sim, 'affine_registered').copy()

    # Navigate to tp=1
    current_step = list(viewer.dims.current_step)
    current_step[-ndim - 1] = 1
    viewer.dims.current_step = tuple(current_step)

    # Build a custom affine that is different from the current one at tp=1
    custom_affine = np.array(l.affine.affine_matrix).copy()
    custom_affine[-(ndim), -1] += 99  # big shift in last spatial dim

    # Simulate user drag (flag must be False)
    wdg._updating_viewer = False
    l.affine = custom_affine

    # Re-fetch updated params
    updated_sim = msi_utils.get_sim_from_msim(wdg.msims[l.name])
    updated_params = spatial_image_utils.get_affine_from_sim(
        updated_sim, 'affine_registered')

    # tp=1 should now contain the custom affine (lower-right (ndim+1) block)
    p_modified = np.array(updated_params.isel(t=1)).squeeze()
    assert np.allclose(
        p_modified, custom_affine[-(ndim + 1):, -(ndim + 1):]
    ), "affine_registered for tp=1 should equal the custom affine"

    # tp=0 and tp=2 must be unchanged
    for tp_idx in [0, 2]:
        p_orig = np.array(original_params.isel(t=tp_idx)).squeeze()
        p_new  = np.array(updated_params.isel(t=tp_idx)).squeeze()
        assert np.allclose(p_orig, p_new), \
            f"affine_registered for tp={tp_idx} should be unchanged"


def test_manual_transform_show_original_no_msim_update(make_napari_viewer):
    """
    Post-registration with Show=Original: manually moving a layer must NOT
    update affine_metadata in the msim (it will be captured at registration
    time, not live).
    """
    viewer = make_napari_viewer()
    wdg = StitcherQWidget(viewer)
    viewer.window.add_dock_widget(wdg)

    ndim = 2
    sims = generate_tiled_dataset(
        ndim=ndim, N_t=2, N_c=1,
        tile_size=30, tiles_x=2, tiles_y=1, tiles_z=1,
        overlap=5, zoom=10, dtype=np.uint8)
    msims = [msi_utils.get_msim_from_sim(sim, scale_factors=[]) for sim in sims]
    layer_tuples = viewer_utils.create_image_layer_tuples_from_msims(
        msims, transform_key=METADATA_TRANSFORM_KEY)
    for lt in layer_tuples:
        viewer.add_image(lt[0], **lt[1])

    wdg.button_load_layers_all.clicked()
    wdg.run_registration()

    # Switch to Original
    wdg.visualization_type_rbuttons.value = _stitcher_widget.CHOICE_METADATA

    l = wdg.input_layers[0]
    sim = msi_utils.get_sim_from_msim(wdg.msims[l.name])
    original_metadata = spatial_image_utils.get_affine_from_sim(
        sim, 'affine_metadata').copy()

    # Simulate user drag while showing Original
    wdg._updating_viewer = False
    custom_affine = np.array(l.affine.affine_matrix).copy()
    custom_affine[-(ndim), -1] += 77
    l.affine = custom_affine

    # affine_metadata in the msim must remain unchanged
    updated_sim = msi_utils.get_sim_from_msim(wdg.msims[l.name])
    updated_metadata = spatial_image_utils.get_affine_from_sim(
        updated_sim, 'affine_metadata')
    assert np.allclose(
        np.array(updated_metadata), np.array(original_metadata)
    ), "affine_metadata should NOT be updated live when showing Original"


@pytest.mark.parametrize("all_multiscale", [True, False])
def test_fusion_output_scale_matches_input(all_multiscale, make_napari_viewer):
    """
    Fused result is multiscale (>1 resolution levels) iff all input layers
    are multiscale; a single non-multiscale input collapses the output to
    a single resolution level.
    """
    viewer = make_napari_viewer()
    wdg = StitcherQWidget(viewer)
    viewer.window.add_dock_widget(wdg)

    if all_multiscale:
        sims = generate_tiled_dataset(
            ndim=2, N_t=1, N_c=1, tile_size=30,
            tiles_x=2, tiles_y=1, tiles_z=1, overlap=5, zoom=10, dtype=np.uint8)
        msims = [msi_utils.get_msim_from_sim(sim, scale_factors=[2]) for sim in sims]
        layer_tuples = viewer_utils.create_image_layer_tuples_from_msims(
            msims, transform_key=METADATA_TRANSFORM_KEY)
        for lt in layer_tuples:
            viewer.add_image(lt[0], **lt[1])
    else:
        # Plain numpy arrays; l.multiscale is False for these layers
        D = 100
        arr = np.random.randint(0, 255, (D, D), dtype=np.uint8)
        viewer.add_image(arr[:, :D//2+D//10], translate=(0, 0), name='im1')
        viewer.add_image(arr[:, D//2-D//10:], translate=(0, D//2-D//10), name='im2')

    wdg.button_load_layers_all.clicked()
    assert wdg.all_layers_multiscale == all_multiscale

    wdg.run_fusion()
    fused_layer = wdg.fused_layers[-1]
    # Fused layer is multiscale iff all inputs were multiscale
    assert fused_layer.multiscale == all_multiscale


@pytest.mark.parametrize("precompute", [True, False])
def test_precompute_fusion(precompute, make_napari_viewer):
    """
    When Pre-compute fusion is enabled the result is written to a temporary
    zarr store before the layer is created; when disabled no disk write occurs
    and the layer is backed by a lazy dask graph.
    """
    viewer = make_napari_viewer()
    wdg = StitcherQWidget(viewer)
    viewer.window.add_dock_widget(wdg)

    D = 100
    arr = np.random.randint(0, 255, (D, D), dtype=np.uint8)
    viewer.add_image(arr[:, :D//2+D//10], translate=(0, 0), name='im1')
    viewer.add_image(arr[:, D//2-D//10:], translate=(0, D//2-D//10), name='im2')

    wdg.button_load_layers_all.clicked()
    wdg.precompute_fusion_n_jobs.value = 1
    wdg.precompute_fusion.value = precompute
    assert wdg.precompute_fusion_path.mode.value == 'd'
    assert wdg.precompute_fusion_path.enabled == precompute
    assert wdg.precompute_fusion_n_jobs.enabled == precompute
    wdg.run_fusion()

    zarr_written = any(name.endswith('.zarr') for name in os.listdir(wdg.tmpdir.name))
    assert zarr_written == precompute


def test_precompute_fusion_uses_custom_path(tmp_path, make_napari_viewer):
    """
    User-selected fusion paths are the top-level pre-computed zarr stores.
    """
    viewer = make_napari_viewer()
    wdg = StitcherQWidget(viewer)
    viewer.window.add_dock_widget(wdg)

    D = 100
    arr = np.random.randint(0, 255, (D, D), dtype=np.uint8)
    viewer.add_image(arr[:, :D//2+D//10], translate=(0, 0), name='im1')
    viewer.add_image(arr[:, D//2-D//10:], translate=(0, D//2-D//10), name='im2')

    wdg.button_load_layers_all.clicked()
    wdg.precompute_fusion_n_jobs.value = 1
    output_path = tmp_path / 'custom_fusion.zarr'
    wdg.precompute_fusion_path.value = output_path
    wdg.run_fusion()

    assert output_path.exists()
    assert not any(name.endswith('.zarr') for name in os.listdir(wdg.tmpdir.name))


def test_precompute_fusion_job_options(make_napari_viewer):
    """
    Pre-computed fusion uses joblib with batches scaled to the job count.
    """
    viewer = make_napari_viewer()
    wdg = StitcherQWidget(viewer)

    assert wdg.precompute_fusion_n_jobs.value == (os.cpu_count() or 1)

    n_jobs = min(3, int(wdg.precompute_fusion_n_jobs.max))
    wdg.precompute_fusion_n_jobs.value = n_jobs
    batch_options = wdg._get_precompute_fusion_batch_options()

    assert batch_options['batch_func'] is misc_utils.process_batch_using_joblib
    assert batch_options['n_batch'] == 5 * n_jobs
    assert batch_options['batch_func_kwargs']['n_jobs'] == n_jobs


@pytest.mark.parametrize("use_layer_res", [True, False])
def test_use_layer_resolution(use_layer_res, make_napari_viewer):
    """
    When 'Use layer resolution' is enabled, registration receives single-scale
    msims at the layer's current data_level.  When disabled it receives the
    full multiscale msims (> 1 scale level).
    """
    import multiview_stitcher.registration as mv_reg

    viewer = make_napari_viewer()
    wdg = StitcherQWidget(viewer)
    viewer.window.add_dock_widget(wdg)

    sims = generate_tiled_dataset(
        ndim=2, N_t=1, N_c=1, tile_size=30,
        tiles_x=2, tiles_y=1, tiles_z=1, overlap=5, zoom=10, dtype=np.uint8)
    msims = [msi_utils.get_msim_from_sim(sim, scale_factors=[2]) for sim in sims]
    for lt in viewer_utils.create_image_layer_tuples_from_msims(
            msims, transform_key=METADATA_TRANSFORM_KEY):
        viewer.add_image(lt[0], **lt[1])

    wdg.button_load_layers_all.clicked()

    # Lock level 0 (finest) on every multiscale layer, simulating the user
    # choosing a specific resolution in napari's layer controls (not Auto)
    for l in wdg.input_layers:
        if l.multiscale:
            l.locked_data_level = 0

    wdg.use_layer_resolution.value = use_layer_res

    captured = []
    real_register = mv_reg.register

    def capturing_register(msims_arg, **kwargs):
        captured.extend(msims_arg)
        return real_register(msims_arg, **kwargs)

    with patch.object(mv_reg, 'register', side_effect=capturing_register):
        wdg.run_registration()

    n_scales = [len(msi_utils.get_sorted_scale_keys(m)) for m in captured]
    if use_layer_res:
        assert all(n == 1 for n in n_scales), \
            f"expected single-scale reg inputs, got {n_scales}"
    else:
        assert all(n > 1 for n in n_scales), \
            f"expected multiscale reg inputs, got {n_scales}"
