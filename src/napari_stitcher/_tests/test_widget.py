import numpy as np
from pathlib import Path

# from napari_stitcher import ExampleQWidget, example_magic_widget
from napari_stitcher import StitcherQWidget, _widget, _spatial_image_utils


def test_data_loading_while_plugin_open(make_napari_viewer):

    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()

    test_path = Path(__file__).parent.parent.parent.parent /\
                             "image-datasets" / "mosaic_test.czi"
    viewer.open(test_path)

    stitcher_widget = StitcherQWidget(viewer)
    viewer.window.add_dock_widget(stitcher_widget)

    # test also opening when plugin is already open
    viewer.layers.clear()
    viewer.open(test_path)

    # import pdb; pdb.set_trace()


# make_napari_viewer is a pytest fixture that returns a napari viewer object
# capsys is a pytest fixture that captures stdout and stderr output streams
def test_stitcher_q_widget_integrated(make_napari_viewer, capsys):
    """
    Integration test covering typical pipeline.
    """

    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()

    stitcher_widget = StitcherQWidget(viewer)

    test_path = Path(__file__).parent.parent.parent.parent /\
                             "image-datasets" / "mosaic_test.czi"
    
    viewer.open(test_path)

    # Run stitching
    # stitcher_widget.button_stitch.clicked()
    stitcher_widget.run_stitching(scheduler='single-threaded')

    # Check that parameters were obtained
    assert(stitcher_widget.params is not None)

    # Check that parameters are visualised

    # First, view 0 is not shifted
    assert(np.allclose(
        # _spatial_image_utils.get_data_to_world_matrix_from_spatial_image(viewer.layers[0].data),
        np.eye(3),
        viewer.layers[0].affine.affine_matrix))
    
    # Toggle showing the registrations
    stitcher_widget.visualization_type_rbuttons.value=_widget.CHOICE_REGISTERED
    # Make sure view 0 is shifted now
    assert(~np.allclose(
        # _spatial_image_utils.get_data_to_world_matrix_from_spatial_image(viewer.layers[0].data),
        np.eye(3),
        viewer.layers[0].affine.affine_matrix))

    # # Run fusion
    # stitcher_widget.button_fuse.clicked()

    # # Make sure layers are created
    # assert(3, len(viewer.layers))
    # assert(True, min(['fused' in l.name for l in viewer.layers]))

    # create our widget, passing in the viewer


    # call our widget method
    # my_widget._on_click()

    # read captured output and check that it's as we expected
    # captured = capsys.readouterr()
    # assert captured.out == "napari has 1 layers\n"
