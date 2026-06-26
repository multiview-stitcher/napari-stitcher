import numpy as np

from napari_stitcher import (
    MosaicQWidget,
)

import pytest


@pytest.mark.parametrize(
    "ndim, n_rows, n_cols, mosaic_arr, n_channels",
    [
        [2, 2, 2, 'rows first', 1],
        [3, 2, 2, 'columns first', 2],
        [2, 2, 2, 'snake by rows', 2],
        [3, 2, 2, 'snake by columns', 1],
    ]
)
def test_mosaic_loading(
    ndim, n_rows, n_cols, mosaic_arr, n_channels, make_napari_viewer
    ):

    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()

    wdg = MosaicQWidget(viewer)
    viewer.window.add_dock_widget(wdg)

    for ch in range(n_channels):
        for irow in range(n_rows):
            for icol in range(n_cols):
                viewer.add_image(
                    np.ones([10] * ndim),
                    name=f'layer_{irow}_{icol} :: ch{ch}'
                    )

    initial_poss = np.array([l.translate[-2:] for l in viewer.layers])

    wdg.n_col.value = n_cols
    wdg.n_row.value = n_rows
    wdg.mosaic_arr.value = mosaic_arr

    wdg.button_arrange_tiles.clicked()

    final_poss = np.array([l.translate[-2:] for l in viewer.layers])

    # assert that the positions have changed
    assert not np.allclose(initial_poss, final_poss)

    # assert that the changed positions are the same for different channels
    assert np.allclose(
        final_poss[:n_cols*n_rows],
        final_poss[-n_cols*n_rows:]
        )


def test_mosaic_arrangement_uses_affine_translation(make_napari_viewer):
    viewer = make_napari_viewer()

    wdg = MosaicQWidget(viewer)
    viewer.window.add_dock_widget(wdg)

    for itile in range(2):
        viewer.add_image(
            np.ones((10, 10)),
            name=f'layer_{itile} :: ch0',
            translate=(5 + itile, 6 + itile),
            )

    first_affine = np.eye(3)
    first_affine[:-1, -1] = [25, 40]
    viewer.layers[0].affine = first_affine

    second_affine = np.eye(3)
    second_affine[:-1, -1] = [100, 200]
    viewer.layers[1].affine = second_affine

    tile_step = (
        viewer.layers[0].extent.world[1, -1]
        - viewer.layers[0].extent.world[0, -1]
    )

    wdg.n_col.value = 2
    wdg.n_row.value = 1
    wdg.overlap.value = 0
    wdg.mosaic_arr.value = 'rows first'

    wdg.button_arrange_tiles.clicked()

    expected_anchor = np.array([25, 40]) + np.array([5, 6])
    expected_positions = np.array([
        expected_anchor,
        [expected_anchor[0], expected_anchor[1] + tile_step],
    ])
    final_positions = np.array([l.translate[-2:] for l in viewer.layers])

    assert np.allclose(final_positions, expected_positions)

    affine_grid_translations = np.array([
        l.affine.affine_matrix[:-1, -1][-2:] for l in viewer.layers
    ])
    assert np.allclose(affine_grid_translations, 0)
