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


        self.loading_widgets = [
                            self.load_layers_box,
                            ]


        self.container = widgets.VBox(widgets=\
                            self.loading_widgets
                            )

        self.container.native.setMinimumWidth = 50

        self.layout().addWidget(self.container.native)

        # initialize registration parameter dict
        self.input_layers= []
        self.msims = {}
        self.fused_layers = []
        self.params = dict()

        # create temporary directory for storing dask arrays
        self.tmpdir = tempfile.TemporaryDirectory()

        self.button_load_layers_all.clicked.connect(self.load_layers_all)
        self.button_load_layers_sel.clicked.connect(self.load_layers_sel)


    def reset(self):
            
        self.msims = {}
        self.params = dict()
        self.reg_ch_picker.choices = ()
        self.times_slider.min, self.times_slider.max = (-1, 0)
        self.times_slider.value = (-1, 0)
        self.input_layers = []
        self.fused_layers = []


    def load_metadata(self):
        
        reference_sim = msi_utils.get_sim_from_msim(self.msims[self.input_layers[0].name])
        
        # assume dims are the same for all layers
        if 't' in reference_sim.dims:
            self.times_slider.enabled = True
            self.times_slider.min = -1
            self.times_slider.max = len(reference_sim.coords['t']) - 1
            self.times_slider.value = self.times_slider.min, self.times_slider.max

        if 'c' in reference_sim.coords.keys():
            self.reg_ch_picker.enabled = True
            self.reg_ch_picker.choices = np.unique([
                _utils.get_str_unique_to_ch_from_sim_coords(msi_utils.get_sim_from_msim(msim).coords)
                for l_name, msim in self.msims.items()])
            self.reg_ch_picker.value = self.reg_ch_picker.choices[0]

        from collections.abc import Iterable
        for w in self.reg_widgets + self.fusion_widgets:
            if isinstance(w, Iterable):
                for sw in w:
                    sw.enabled = True
            w.enabled = True


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

        self.load_metadata()


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
