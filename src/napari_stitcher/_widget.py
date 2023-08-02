"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING
import pathlib, os, tempfile, sys

import numpy as np
import dask
import dask.array as da

from napari.utils import notifications

from magicgui import magic_factory, magicgui, widgets
from qtpy.QtWidgets import QVBoxLayout, QPushButton, QWidget

# from mvregfus import mv_utils

from napari_stitcher import _utils, _registration, _fusion,\
    _mv_graph, _spatial_image_utils, _viewer_utils, _msi_utils

if TYPE_CHECKING:
    import napari


# define labels for visualization choices
CHOICE_METADATA = 'Original'
CHOICE_REGISTERED = 'Registered'


class StitcherQWidget(QWidget):
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
            # label='Load layers: ',
                                                widgets=\
                                                [self.button_load_layers_sel,
                                                 self.button_load_layers_all])
        self.layers_selection = widgets.Select(choices=[])
        self.load_layers_box = widgets.VBox(widgets=\
                                            [
            self.buttons_load_layers,
            self.layers_selection,
                                            ],
                                            label='Loaded\nlayers:')

        self.times_slider = widgets.RangeSlider(min=-1, max=0, label='Timepoints:', enabled=False,
            tooltip='Timepoints to process. Because the two sliders cannot coincide, positions are a bit criptic: E.g.\n(-1, 0) means timepoint 0 is processed\n(3, 5) means timepoints 4 and 5 are processed')
        
        self.reg_ch_picker = widgets.ComboBox(
            label='Reg channel: ',
            choices=[],
            tooltip='Choose a file to process using napari-stitcher.')

        self.button_stitch = widgets.Button(text='Stitch', enabled=False,
            tooltip='Use the overlaps between tiles to determine their relative positions.')
        self.button_stabilize = widgets.Button(text='Stabilize', enabled=False,
            tooltip='Use time lapse information to stabilize each tile over time,'+\
                    'eliminating abrupt shifts between frames. No tile overlap needed.')

        self.buttons_register_tracks = widgets.HBox(
            widgets=[
                    self.button_stitch,
                    self.button_stabilize
                    ]
                    )

        self.visualization_type_rbuttons = widgets.RadioButtons(
            choices=[CHOICE_METADATA, CHOICE_REGISTERED],
            label="Show:",
            value=CHOICE_METADATA, enabled=False,
            orientation='horizontal')

        self.button_fuse = widgets.Button(text='Fuse', enabled=False,
            tooltip='Fuse the tiles using the parameters obtained'+\
                    'from stitching or stabilization.\nCombines all'+\
                    'tiles and timepoints into a single image, smoothly'+\
                    'blending the overlaps and filling in gaps.')

        self.loading_widgets = [
                            self.load_layers_box,
                            ]

        self.reg_widgets = [
                            self.times_slider,
                            self.reg_ch_picker,
                            self.buttons_register_tracks,
                            ]

        self.visualization_widgets = [
                            self.visualization_type_rbuttons,
        ]

        self.fusion_widgets = [
                            widgets.HBox(widgets=[self.button_fuse]),
                            ]


        self.container = widgets.VBox(widgets=\
                            self.loading_widgets+
                            self.reg_widgets+
                            self.visualization_widgets+
                            self.fusion_widgets
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
        
        self.visualization_type_rbuttons.changed.connect(self.update_viewer_transformations)
        self.viewer.dims.events.connect(self.update_viewer_transformations)

        self.button_stitch.clicked.connect(self.run_stitching)
        self.button_stabilize.clicked.connect(self.run_stabilization)
        self.button_fuse.clicked.connect(self.run_fusion)

        self.button_load_layers_all.clicked.connect(self.load_layers_all)
        self.button_load_layers_sel.clicked.connect(self.load_layers_sel)


    def update_viewer_transformations(self):
        """
        set transformations for current timepoint
        """

        if not len(self.params): return

        reference_xim = _msi_utils.get_xim_from_msim(self.msims[self.input_layers[0].name])
        spatial_dims = _spatial_image_utils.get_spatial_dims_from_xim(reference_xim)

        for l in self.input_layers:

            # layer_xim = self.msims[l.name]
            layer_xim = _msi_utils.get_xim_from_msim(self.msims[l.name])

            ndim = len(_spatial_image_utils.get_spatial_dims_from_xim(layer_xim))

            # get curr tp
            # handle possibility that there had been no T dimension
            # when collecting xims from layers
            if len(self.viewer.dims.current_step) > len(spatial_dims):
                curr_tp = self.viewer.dims.current_step[-len(spatial_dims)-1]
            else:
                curr_tp = 0
        
            if self.visualization_type_rbuttons.value == CHOICE_REGISTERED:

                params = self.params[_utils.get_str_unique_to_view_from_layer_name(l.name)]

                try:
                    p = np.array(params.sel(t=layer_xim.coords['t'][curr_tp])).squeeze()
                except:

                    # notifications.notification_manager.receive_info(
                    #     'Timepoint %s: no parameters available, register first.' % curr_tp)
                    # self.visualization_type_rbuttons.value = CHOICE_METADATA
                    # return

                    # if curr_tp not available, use nearest available parameter
                    notifications.notification_manager.receive_info(
                        'Timepoint %s: no parameters available, taking nearest available one.' % curr_tp)
                    p = np.array(params.sel(t=layer_xim.coords['t'][curr_tp], method='nearest')).squeeze()


                p = np.linalg.inv(p)
            else:
                p = np.eye(ndim + 1)

            ndim_layer_data = len(layer_xim.shape)

            # if stitcher xim has more dimensions than layer data (i.e. time)
            vis_p = p[-(ndim_layer_data + 1):, -(ndim_layer_data + 1):]

            # if layer data has more dimensions than stitcher xim
            full_vis_p = np.eye(ndim_layer_data + 1)
            full_vis_p[-len(vis_p):, -len(vis_p):] = vis_p

            l.affine.affine_matrix = full_vis_p

            l.refresh()

        # fused layers
        for l in self.fused_layers:
            continue


    def run_stabilization(self):

        layers = list(_utils.filter_layers(
            self.input_layers,
            {lname: _msi_utils.get_xim_from_msim(msim) for lname, msim in self.msims.items()},
            ch=self.reg_ch_picker.value))

        xims = [_msi_utils.get_xim_from_msim(self.msims[l.name]) for l in layers]

        times = range(self.times_slider.value[0] + 1, self.times_slider.value[1] + 1)

        if len(times) < 8:
            message = 'Need at least 8 time points to perform stabilization.'
            notifications.notification_manager.receive_info(message)
            return
        
        xims_tsel = [x.sel(t=[x.coords['t'][it] for it in times]) for x in xims]

        # ndim = len(_spatial_image_utils.get_spatial_dims_from_xim(xims_tsel[0]))

        params = [_registration.get_stabilization_parameters_from_xim(xim) for xim in xims_tsel]

        with _utils.TemporarilyDisabledWidgets([self.container]),\
            _utils.VisibleActivityDock(self.viewer),\
            _utils.TqdmCallback(tqdm_class=_utils.progress,
                                desc='Stabilize tiles', bar_format=" "):
            
            params = dask.compute(params)[0]
            
        self.params.update(
            {_utils.get_str_unique_to_view_from_layer_name(l.name): params[il]
                for il, l in enumerate(layers)})

        self.visualization_type_rbuttons.enabled = True


    def run_stitching(self):

        layers = list(_utils.filter_layers(self.input_layers, self.msims,
                                      ch=self.reg_ch_picker.value))

        msims = [msim for lname, msim in self.msims
                 if self.reg_ch_picker.value in _msi_utils.get_xim_from_msim(msim).coords['c']]

        sims = [_msi_utils.get_xim_from_msim(msim) for msim in msims]

        # restrict timepoints
        _spatial_image_utils.xim_sel_coords()

        sims = [_spatial_image_utils.xim_sel_coords(sim,
                {'t': [sim.coords['t'][it]
                         for it in range(self.times_slider.value[0] + 1,
                                         self.times_slider.value[1] + 1)]})
                  for sim in sims]

        # xims = [x.sel(t=[x.coords['t'][it]
        #                  for it in range(self.times_slider.value[0] + 1,
        #                                  self.times_slider.value[1] + 1)])
        #                                  for x in xims]

        # calculate overlap graph with overlap as edge attributes
        g = _mv_graph.build_view_adjacency_graph_from_xims(sims)

        g_reg = _registration.get_registration_graph_from_overlap_graph(g)

        if not len(g_reg.edges):
            message = 'No overlap between views for stitching. Consider stabilizing the tiles instead.'
            notifications.notification_manager.receive_info(message)
            return
        
        # # restrict tps
        # if 't' in xims[0].dims:
        #     g_reg = _mv_graph.sel_coords_from_graph(g_reg,
        #                 {'t': range(self.times_slider.value[0] + 1, self.times_slider.value[1] + 1)},
        #                 edge_attributes=['transform'],
        #                 node_attributes=['xim'],
        #                 sel_or_isel='isel',
        #                 )

        with _utils.TemporarilyDisabledWidgets([self.container]),\
            _utils.VisibleActivityDock(self.viewer),\
            _utils.TqdmCallback(tqdm_class=_utils.progress,
                                desc='Registering tiles', bar_format=" "):
            g_reg_computed = _mv_graph.compute_graph_edges(g_reg, scheduler='threads')

        # get node parameters
        g_reg_nodes = _registration.get_node_params_from_reg_graph(g_reg_computed)

        node_transforms = _mv_graph.get_nodes_dataset_from_graph(g_reg_nodes, node_attribute='transforms')

        self.params.update({_utils.get_str_unique_to_view_from_layer_name(l.name): node_transforms[il]
                            for il, l in enumerate(layers)})

        self.visualization_type_rbuttons.enabled = True


    def run_fusion(self):

        """
        Split layers into channel groups and fuse each group separately.
        """

        layer_chs = [_utils.get_str_unique_to_ch_from_xim_coords(
            _msi_utils.get_xim_from_msim(msim).coords)
                    for l_name, msim in self.msims.items()]
        
        channels = np.unique(layer_chs)

        channels = self.reg_ch_picker.choices

        for ch in channels:

            layers_to_fuse = list(_utils.filter_layers(self.input_layers, self.msims, ch=ch))

            xims_to_fuse = [self.msims[l.name] for l in layers_to_fuse]

            # restrict timepoints
            xims_to_fuse = [xim.sel(t=[xim.coords['t'][it]
                            for it in range(self.times_slider.value[0] + 1,
                                            self.times_slider.value[1] + 1)])
                                            for xim in xims_to_fuse]
            
            params_to_fuse = [self.params[_utils.get_str_unique_to_view_from_layer_name(l.name)]
                              for l in layers_to_fuse]

            with _utils.TemporarilyDisabledWidgets([self.container]),\
                _utils.VisibleActivityDock(self.viewer),\
                _utils.TqdmCallback(tqdm_class=_utils.progress,
                                    desc='Fusing tiles of channel %s' %ch, bar_format=" "):
                
                xfused = _fusion.fuse(xims_to_fuse, params_to_fuse, self.tmpdir)
            
            xfused = xfused.assign_coords(c=ch)
            
            fused_ch_layer_tuple = _viewer_utils.create_image_layer_tuple_from_spatial_xim(
                xfused,
                colormap=None,
                name_prefix='fused',
            )

            fused_layer = self.viewer.add_image(fused_ch_layer_tuple[0], **fused_ch_layer_tuple[1])
        
            self.fused_layers.append(fused_layer)

        self.update_viewer_transformations()


    def reset(self):
            
            self.msims = {}
            self.params = dict()
            self.reg_ch_picker.choices = ()
            self.visualization_type_rbuttons.value = CHOICE_METADATA
            self.times_slider.min, self.times_slider.max = (-1, 0)
            self.times_slider.value = (-1, 0)
            self.input_layers = []
            self.fused_layers = []


    def load_metadata(self):
        
        # reference_xim = self.msims[self.input_layers[0].name]
        reference_xim = _msi_utils.get_xim_from_msim(self.msims[self.input_layers[0].name])
        
        # assume dims are the same for all layers
        if 't' in reference_xim.dims:
            self.times_slider.enabled = True
            # self.times_slider.min = int(l0.data.coords['t'][0] - 1)
            # self.times_slider.max = int(l0.data.coords['t'][-1] - 1)
            self.times_slider.min = -1
            self.times_slider.max = len(reference_xim.coords['t']) - 1
            self.times_slider.value = self.times_slider.min, self.times_slider.max

        if 'c' in reference_xim.coords.keys():
            self.reg_ch_picker.enabled = True
            self.reg_ch_picker.choices = np.unique([
                _utils.get_str_unique_to_ch_from_xim_coords(_msi_utils.get_xim_from_msim(msim).coords)
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

        # load in layers as xims
        for l in layers:

            msim = _viewer_utils.image_layer_to_msim()

            # xim = l.data
            # try:
            #     len(xim.coords['c'])
            if 'c' in msim['scale0/image'].dims:
                notifications.notification_manager.receive_info(
                    "Layer '%s' has more than one channel.Consider splitting the stack (right click on layer -> 'Split Stack')." %l.name
                )
                self.layers_selection.choices = []
                self.reset()
                return
            
            msim = _msi_utils.ensure_time_dim(msim)
            msim.attrs['layer_name'] = l.name
            self.msims[l.name] = msim

        number_of_channels = len(np.unique([_utils.get_str_unique_to_ch_from_xim_coords(xim.coords)
                                            for l_name, xim in self.msims.items()]))
        
        if len(layers) and number_of_channels > 1:
            self.link_channel_layers(layers)

        self.load_metadata()


    def link_channel_layers(self, layers):

        # link channel layers
        from napari.experimental import link_layers

        msims = [self.msims[l.name] for l in layers]
        sims = [_msi_utils.get_sim_from_msim(msim) for msim in msims]

        channels = [_utils.get_str_unique_to_ch_from_xim_coords(sim.coords) for sim in sims]
        for ch in channels:
            ch_layers = list(_utils.filter_layers(layers, self.msims, ch=ch))

            if len(ch_layers):
                link_layers(ch_layers, ('contrast_limits', 'visible'))


    def __del__(self):

        print('Deleting napari-stitcher widget')

        # clean up callbacks
        self.viewer.dims.events.disconnect(self.update_viewer_transformations)


if __name__ == "__main__":
    import napari

    # filename = "/Users/malbert/software/napari-stitcher/image-datasets/04_stretch-01_AcquisitionBlock2_pt2.czi"
    # filename = "/Users/malbert/software/napari-stitcher/image-datasets/yu_220829_WT_quail_st6_x10_zoom0.7_1x3_488ZO1-568Sox2-647Tbra.czi"

    # filename = "/Users/malbert/software/napari-stitcher/image-datasets/arthur_20220621_premovie_dish2-max.czi"
    # filename = "/Users/malbert/software/napari-stitcher/image-datasets/mosaic_test.czi"
    # filename = "/Users/malbert/software/napari-stitcher/image-datasets/MAX_LSM900.czi"
    # filename = '/Users/malbert/software/napari-stitcher/image-datasets/arthur_20210216_highres_TR2.czi'
    filename = '/Users/malbert/software/napari-stitcher/image-datasets/arthur_20230710_03_high-res_ant_20X-max-e1-bin-8bit.czi'


    viewer = napari.Viewer()
    
    # viewer.open("/Users/malbert/software/napari-stitcher/image-datasets/arthur_20220609_WT_emb2_5X_part1_max.czi")

    wdg = StitcherQWidget(viewer)
    viewer.window.add_dock_widget(wdg)

    viewer.open(filename)

    # wdg.times_slider.value = (-1, 0)
    # wdg.params = xr.open_dataset('test.netcdf')
    # wdg.visualization_type_rbuttons.enabled = True

    wdg.button_load_layers_all.clicked()

    # wdg.times_slider.min = -1
    # wdg.times_slider.max = 1

    # wdg.run_stitching()
