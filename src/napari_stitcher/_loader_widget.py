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

        self.button_load_layers_all = widgets.Button(text='All')
        self.button_load_layers_sel = widgets.Button(text='Selected')
        self.buttons_load_layers = widgets.HBox(
            widgets=\
                [self.button_load_layers_sel,
                    self.button_load_layers_all]
                    )
        self.layers_selection = widgets.Select(choices=[])
        self.load_layers_box = widgets.VBox(widgets=\
                                            [
            self.buttons_load_layers,
            self.layers_selection,
                                            ],
                                            label='Loaded\nlayers:')

        # mosaic arrangement widgets
        self.overlap = widgets.FloatSlider(value=0.1, min=0, max=0.9999, label='overlap:')
        self.n_col = widgets.SpinBox(value=1, min=1, max=100, label='number of columns:') 
        self.n_row = widgets.SpinBox(value=1, min=1, max=100, label='number of rows:') 
        self.mosaic_arr = widgets.ComboBox(choices=['rows first','columns first','snake by rows','snake by columns'],
                                             value='snake by rows', 
                                             label='Mosaic arrangement:')
        self.button_arrange_tiles = widgets.Button(text='Arrange tiles')

        # organize widgets
        self.loading_widgets = [
                            self.load_layers_box,
                            ]
        
        self.mosaic_widgets = [
                            self.overlap,
                            self.n_col, 
                            self.n_row, 
                            self.mosaic_arr, 
                            self.button_arrange_tiles,
                            ]


        self.container = widgets.VBox(widgets=\
                            self.loading_widgets+
                            self.mosaic_widgets
                            )

        self.container.native.setMinimumWidth = 50

        self.layout().addWidget(self.container.native)

        # initialize registration parameter dict
        self.input_layers= []
        self.msims = {}


        # create temporary directory for storing dask arrays
        self.tmpdir = tempfile.TemporaryDirectory()

        self.button_load_layers_all.clicked.connect(self.load_layers_all)
        self.button_load_layers_sel.clicked.connect(self.load_layers_sel)

        self.button_arrange_tiles.clicked.connect(self.arrange_tiles)


    def reset(self):
            
        self.msims = {}
        self.input_layers = []


    def load_layers_all(self):

        if not len(self.viewer.layers):
            notifications.notification_manager.receive_info(
                'No images in the layer list.'
            )
            return

        self.load_layers(self.viewer.layers)


    def load_layers_sel(self):

        if not len(self.viewer.layers.selection):
            notifications.notification_manager.receive_info(
                'Select layers from the layer list (mutliple using shift / %s'\
                    %('control' if ('command' in sys.platform) else 'shift')
            )
            return

        self.load_layers([l for l in self.viewer.layers.selection])


    def load_layers(self, layers):

        self.reset()
        self.layers_selection.choices = sorted([l.name for l in layers])

        self.input_layers = [l for l in layers]

        # load in layers as sims
        for l in layers:

            msim = viewer_utils.image_layer_to_msim(l, self.viewer)
            
            if 'c' in msim['scale0/image'].dims:
                notifications.notification_manager.receive_info(
                    "Layer '%s' has more than one channel.Consider splitting the stack (right click on layer -> 'Split Stack')." %l.name
                )
                self.layers_selection.choices = []
                self.reset()
                return
            
            msim = msi_utils.ensure_dim(msim, 't')
            self.msims[l.name] = msim

        sims = [msi_utils.get_sim_from_msim(msim) for l.name, msim in self.msims.items()]

        number_of_channels = len(np.unique([
            _utils.get_str_unique_to_ch_from_sim_coords(sim.coords)
                for sim in sims]))
        
        if len(layers) and number_of_channels > 1:
            self.link_channel_layers(layers)

        #self.load_metadata()  ## remove? 


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
        Arrange tiles depending on the selected mosaic arrangement.
        """

        if not len(self.msims):
            notifications.notification_manager.receive_info(
                'No layers loaded.'
            )
            return

        n_tiles = len(self.msims)

        # get mosaic parameters assuming all tiles have the same size
        tile_w = self.msims[self.input_layers[0].name]['scale0/image'].x.shape[0]
        tile_h = self.msims[self.input_layers[0].name]['scale0/image'].y.shape[0]
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
        ind_list = _utils.get_tile_indices(mosaic_arr=mosaic_arr,n_col=n_col,n_row=n_row,n_tiles=n_tiles)
        pos_list = []  # list of tiles' positions
        for i in range(len(ind_list)): 
            pos_list.append((x_array[ind_list[i][0]],y_array[ind_list[i][1]]))

        print(pos_list)

    def __del__(self):

        print('Deleting napari-stitcher widget')

        # clean up callbacks
        self.viewer.dims.events.disconnect(self.update_viewer_transformations)


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
