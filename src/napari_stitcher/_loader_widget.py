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




class LoaderQWidget(QWidget):
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
        self.overlap = widgets.FloatSlider(value=0.1, min=0, max=0.9999, label='overlap:')
        self.n_col = widgets.SpinBox(value=1, min=1, max=100, label='number of columns:') 
        self.n_row = widgets.SpinBox(value=1, min=1, max=100, label='number of rows:') 
        self.mosaic_arr = widgets.ComboBox(choices=['rows first','columns first','snake by rows','snake by columns'],
                                             value='snake by columns', 
                                             label='Mosaic arrangement:')
        self.button_arrange_tiles = widgets.Button(text='Arrange tiles')

        # organize widgets        
        self.mosaic_widgets = [
                            self.overlap,
                            self.n_col, 
                            self.n_row, 
                            self.mosaic_arr, 
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

    
    def get_tiles_pos(self):
        """
        Calculate tiles' positions depending on the selected mosaic arrangement.
        """

        # get mosaic parameters assuming all tiles have the same size
        l0 = self.viewer.layers[0]
        #dims = viewer_utils.get_layer_dims(l0, self.viewer)
        #sdims = [dim for dim in dims if dim in ['x', 'y', 'z']]  # spatial dimensions
        # for now get tile size assuming that y and x are the last dimensions
        tile_w = l0.extent.world[1,-1]-l0.extent.world[0,-1]  # tile width
        tile_h = l0.extent.world[1,-2]-l0.extent.world[0,-2]  # tile height
        overlap = self.overlap.value
        n_col = self.n_col.value
        n_row = self.n_row.value
        mosaic_arr = self.mosaic_arr.value

        # define mosaic position
        x_spacing = tile_w * (1 - overlap)  # x spacing between two adjacent tiles
        x_max = tile_w * (0.5 + (n_col - 1) * (1 - overlap))  # last column tile center x coordinate
        total_w = tile_w * (1 + (n_col - 1) * (1 - overlap))  # width of the total mosaic

        y_spacing = tile_h * (1 - overlap)  # y spacing between two adjacent tiles
        y_max = tile_h * (0.5 + (n_row - 1) * (1 - overlap))  # last row tile center y coordinate
        total_h = tile_h * (1 + (n_row - 1) * (1 - overlap))  # height of the total mosaic

        x_array = np.linspace(tile_w/2, x_max, n_col)
        y_array = np.linspace(tile_h/2, y_max, n_row)

        # get tile arrangement
        ind_list = _utils.get_tile_indices(mosaic_arr=mosaic_arr,n_col=n_col,n_row=n_row,n_tiles=self.n_tiles)
        pos_list = []  # list of tiles' positions
        for i in range(len(ind_list)): 
            pos_list.append((x_array[ind_list[i][0]],y_array[ind_list[i][1]]))

        return pos_list
    
    def arrange_tiles(self):
        """
        Arrange tiles in the viewer according to the selected mosaic arrangement.
        """
        
        self.n_tiles = len(self.viewer.layers)  # not handling case of mulitple channels for now

        if self.n_tiles == 0:
            notifications.notification_manager.receive_info(
                'No layers loaded.'
            )
            return
        
        # get tiles' positions
        pos_list = self.get_tiles_pos()

        for i,pos in enumerate(pos_list):
            self.viewer.layers[i].translate[-2:] += [pos[1],pos[0]]

        

    def __del__(self):

        print('Deleting napari-stitcher widget')


if __name__ == "__main__":

    import napari
    from multiview_stitcher.sample_data import get_mosaic_sample_data_path

    # filename = get_mosaic_sample_data_path()
    filename = "/Users/malbert/software/multiview-stitcher/image-datasets/arthur_20220621_premovie_dish2-max.czi"

    viewer = napari.Viewer()
    
    wdg = LoaderQWidget(viewer)
    viewer.window.add_dock_widget(wdg)

    viewer.open(filename, scene_index=0, plugin='napari-stitcher')

    # wdg.button_load_layers_all.clicked()

    # wdg.times_slider.value = (-1, 1)

    # wdg.run_registration()
    # wdg.run_fusion()
