"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING
import os, tempfile, sys

import numpy as np

from napari.utils import notifications

from magicgui import widgets
from qtpy.QtWidgets import QVBoxLayout, QWidget

from multiview_stitcher import (
    registration,
    fusion,
    spatial_image_utils,
    msi_utils,
    )

from napari_stitcher import _reader, viewer_utils, _utils

if TYPE_CHECKING:
    import napari


class MosaicQWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.viewer.title = "Napari Stitcher"

        self.setLayout(QVBoxLayout())

        # mosaic arrangement widgets
        self.overlap = widgets.FloatSlider(
            value=0.1, min=0, max=0.9999, label='Overlap:')
        self.n_col = widgets.SpinBox(
            value=1, min=1, max=100, label='Number of columns:') 
        self.n_row = widgets.SpinBox(
            value=1, min=1, max=100, label='Number of rows:') 
        self.mosaic_arr = widgets.ComboBox(
            choices=['rows first','columns first','snake by rows','snake by columns'],
            value='snake by columns', 
            label='Mosaic arrangement:',
            tooltip='Select the type of grid arrangement. The tile order is taken from the list of layers.')
        self.input_order = widgets.ComboBox(
            choices=['forward','backward'],
            value='forward', 
            label='Image order:',
            tooltip='Whether to arrange the tiles in forward or backward order.')
        self.button_arrange_tiles = widgets.Button(text='Arrange tiles')

        # organize widgets        
        self.mosaic_widgets = [
                            self.overlap,
                            self.n_col, 
                            self.n_row, 
                            self.mosaic_arr,
                            self.input_order,
                            self.button_arrange_tiles,
                            ]

        self.container = widgets.VBox(widgets=\
                            self.mosaic_widgets
                            )

        self.container.native.setMinimumWidth = 50

        self.layout().addWidget(self.container.native)

        # connect callbacks
        self.button_arrange_tiles.clicked.connect(self.arrange_tiles)


    def reset(self):
            
        self.msims = {}
        self.input_layers = []


    def link_channel_layers(self, layers):

        # link channel layers
        from napari.experimental import link_layers

        sims = {l.name: msi_utils.get_sim_from_msim(self.msims[l.name])
                for l in layers}

        channels = [_utils.get_str_unique_to_ch_from_sim_coords(sim.coords) for sim in sims.values()]
        for ch in channels:
            ch_layers = list(_utils.filter_layers(layers, sims, ch=ch))

            if len(ch_layers):
                link_layers(ch_layers, ('contrast_limits', 'visible'))

    
    def arrange_tiles(self):
        """
        Arrange tiles in the viewer according to the selected mosaic arrangement.
        """

        layer_channels = [
            l.name if not '::' in l.name else _utils.get_str_unique_to_ch_from_layer_name(l.name)
            for l in self.viewer.layers]

        layer_views = [
            _utils.get_str_unique_to_view_from_layer_name(l.name)
            for l in self.viewer.layers]

        view_order = []
        for lview in layer_views:
            if lview not in view_order:
                view_order.append(lview)

        if self.input_order.value == 'backward':
            view_order = view_order[::-1]
        
        n_channels = len(np.unique(layer_channels))
        n_tiles = len(np.unique(layer_views))

        if n_tiles != self.n_col.value * self.n_row.value:
            notifications.notification_manager.receive_info(
                'Warning: Total number of tiles does not match the selected mosaic arrangement.'
            )
            return

        if n_tiles == 0:
            notifications.notification_manager.receive_info(
                'No layers loaded.'
            )
            return

        tile_w = self.viewer.layers[0].extent.world[1,-1] - self.viewer.layers[0].extent.world[0,-1]  # tile width
        tile_h = self.viewer.layers[0].extent.world[1,-2] - self.viewer.layers[0].extent.world[0,-2]  # tile height
        
        tile_indices = _utils.get_tile_indices(
                mosaic_arr=self.mosaic_arr.value,
                n_col=self.n_row.value,
                n_row=self.n_col.value,
                n_tiles=n_tiles
            )

        if self.input_order.value == 'forward':
            l0_translate = self.viewer.layers[0].translate[-2:]
        else:
            l0_translate = self.viewer.layers[-1].translate[-2:]

        for l in self.viewer.layers:
            view = _utils.get_str_unique_to_view_from_layer_name(l.name)
            itile = view_order.index(view)
            tile_index = tile_indices[itile]
            # print(f'Layer {l.name} -> tile index: {tile_index}')
            l.translate[-2:] = [
                l0_translate[0] + tile_index[1] * tile_w - tile_index[1] * tile_w * self.overlap.value,
                l0_translate[1] + tile_index[0] * tile_h - tile_index[0] * tile_h * self.overlap.value
            ]
            l.refresh()
        

    def __del__(self):

        print('Deleting napari-stitcher mosaic widget')


if __name__ == "__main__":

    import napari

    viewer = napari.Viewer()
    
    wdg = MosaicQWidget(viewer)
    viewer.window.add_dock_widget(wdg)

    for irow in range(2):
        for icol in range(2):
            for ch in range(2):
                viewer.add_image(np.random.randint(0, 100, [100] * 3), name=f'layer_{irow}_{icol} :: ch{ch}')

    wdg.n_col.value = 2
    wdg.n_row.value = 2

    wdg.button_arrange_tiles.clicked()
