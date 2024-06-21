import numpy as np

from napari_stitcher import (
    MosaicQWidget,
)

import pytest


@pytest.mark.parametrize(
    "ndim, n_rows, n_cols, mosaic_arr", [
        [2, 2, 2, 'rows first'],
        [3, 2, 2, 'columns first'],
        [2, 2, 2, 'snake by rows'],
        [3, 2, 2, 'snake by columns'],
    ]
)
def test_data_loading_while_plugin_open(
    ndim, n_rows, n_cols, mosaic_arr, make_napari_viewer
    ):

    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()

    wdg = MosaicQWidget(viewer)
    viewer.window.add_dock_widget(wdg)

    for irow in range(n_rows):
        for icol in range(n_cols):
            viewer.add_image(np.ones([10] * ndim), name=f'layer_{irow}_{icol}')

    initial_poss = np.array([l.translate[-2:] for l in viewer.layers])

    wdg.n_col.value = n_cols
    wdg.n_row.value = n_rows
    wdg.mosaic_arr.value = mosaic_arr

    wdg.button_arrange_tiles.clicked()

    final_poss = np.array([l.translate[-2:] for l in viewer.layers])

    assert not np.allclose(initial_poss, final_poss)

