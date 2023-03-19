"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING
import pathlib, os, tempfile

import numpy as np
import dask
import dask.array as da

from napari.utils import notifications

from magicgui import magic_factory, magicgui, widgets
from qtpy.QtWidgets import QVBoxLayout, QPushButton, QWidget

from mvregfus import mv_utils

from napari_stitcher import _utils, _registration, _fusion, _file_utils

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
        
        self.source_identifier = None

        # self.outdir_picker = widgets.FileEdit(label='Output dir:',
        #         value=default_outdir, mode='r')

        # self.button_load_metadata = widgets.Button(text='Load metadata')
        self.source_identifier_picker = widgets.ComboBox(
            label='Input file: ',
            # choices=[(os.path.basename(p), p)
            choices=[_utils.source_identifier_to_str(p)
                for p in _utils.get_list_of_source_identifiers_from_layers(self.viewer.layers)])

        self.times_slider = widgets.RangeSlider(min=0, max=1, label='Timepoints:', enabled=False)
        self.regch_slider = widgets.Slider(min=0, max=1, label='Reg channel:', enabled=False)

        # self.button_visualize_input = widgets.Button(text='Visualize input', enabled=False)
        # self.button_register = widgets.Button(text='Register', enabled=False)
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
                            self.source_identifier_picker,
                            ]

        self.reg_widgets = [
                            self.times_slider,
                            self.regch_slider,
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

        # self.container.native.maximumWidth = 50
        self.container.native.setMinimumWidth = 50

        self.layout().addWidget(self.container.native)

        # initialize registration parameter dict
        self.params = dict()

        # create temporary directory for storing dask arrays
        self.tmpdir = tempfile.TemporaryDirectory()

        # run on startup
        self.load_metadata()
        self.link_channel_layers()

        # link callbacks
        self.button_fuse.clicked.connect(self.run_fusion)
        self.viewer.layers.events.inserted.connect(self.link_channel_layers)
        self.source_identifier_picker.changed.connect(self.load_metadata)
        
        self.visualization_type_rbuttons.changed.connect(self.update_viewer_transformations)
        self.viewer.dims.events.connect(self.update_viewer_transformations)

        self.viewer.layers.events.inserted.connect(self.on_layer_inserted)
        self.viewer.layers.events.removed.connect(self.on_layer_inserted)

        self.button_stitch.clicked.connect(self.run_stitching)
        self.button_stabilize.clicked.connect(self.run_stabilization)


    def update_viewer_transformations(self):
        """
        set transformations for current timepoint
        """

        for l in self.viewer.layers:

            # if l.visible is False: continue

            # unfused layers
            if _utils.layer_was_loaded_by_own_reader(l) and\
                _utils.layer_coincides_with_source_identifier(l, self.source_identifier):

                if 'times' in l.metadata.keys() and len(l.metadata['times']) > 1:
                    curr_tp = self.viewer.dims.current_step[0]
                else:
                    curr_tp = 0

                if self.visualization_type_rbuttons.value == 'Registered'\
                        and curr_tp not in self.params:
                    notifications.notification_manager.receive_info(
                        'Timepoint %s: no parameters available, register first.' % curr_tp)
                    
                    self.visualization_type_rbuttons.value = CHOICE_METADATA
                    return

                view = l.metadata['view']
                
                if self.visualization_type_rbuttons.value == CHOICE_REGISTERED:
                    p = self.params[curr_tp][view]
                else:
                    p = mv_utils.matrix_to_params(np.eye(l.metadata['ndim'] + 1))

                # print(self.visualization_type_rbuttons.value, view, ch)

                p_napari = _utils.params_to_napari_affine(p,
                    l.metadata['stack_props'],
                    l.metadata['view_dict'])

                # embed parameters into ndim + 2 matrix because of time axis
                time_p = np.eye(l.metadata['ndim'] + 2)
                time_p[-len(p_napari):, -len(p_napari):] = p_napari

                l.affine.affine_matrix = time_p
                l.refresh()

            # fused layers
            elif ('processing_state' in l.metadata.keys()):
                
                if 'times' in l.metadata.keys() and len(l.metadata['times']) > 1:
                    curr_tp = self.viewer.dims.current_step[0]
                else:
                    curr_tp = 0

                if curr_tp not in l.metadata['times']:
                    continue

                ndim = l.metadata['ndim']
                p_napari = _utils.params_to_napari_affine(
                    mv_utils.matrix_to_params(np.eye(ndim + 1)),
                    l.metadata['view_dict'][0],
                    l.metadata['stack_props'],
                    )

                # embed parameters into ndim + 2 matrix because of time axis
                time_p = np.eye(l.metadata['ndim'] + 2)
                time_p[-len(p_napari):, -len(p_napari):] = p_napari

                l.affine.affine_matrix = time_p

                l.refresh()

            else: continue


    def on_layer_inserted(self):
        available_source_identifiers =\
            _utils.get_list_of_source_identifiers_from_layers(self.viewer.layers)
        self.source_identifier_picker.choices = [_utils.source_identifier_to_str(si)
            for si in available_source_identifiers]
        self.source_identifier_values = available_source_identifiers


    def run_stabilization(self):

        times = range(self.times_slider.value[0] + 1, self.times_slider.value[1] + 1)

        if len(times) < 8:
            message = 'Need at least 8 time points to perform stabilization.'
            notifications.notification_manager.receive_info(message)
            return

        ndim = len(self.view_dict[self.views[0]]['origin'])

        ps = {view: _registration.get_stabilization_parameters(
                _utils.get_layer_from_source_identifier_view_and_ch(self.viewer.layers,
                                                                    self.source_identifier,
                                                                    view,
                                                                    self.regch_slider.value)\
                                                                        .data[slice(times[0], times[-1] + 1)])
                for view in self.views}
    
        ps = {t: {view: da.concatenate([da.eye(ndim).flatten(),
                                        ps[view][it] * self.view_dict[view]['spacing']], axis=0)
                    for view in self.views} for it, t in enumerate(times)}
        
        psc = _utils.compute_dask_object(ps,
                                         self.viewer,
                                         widgets_to_disable=[self.container],
                                         message='Stabilizing tiles',
                                         scheduler='threading',
                                         )

        self.params.update(psc)
        self.visualization_type_rbuttons.enabled = True


    def run_stitching(self):

        pairs = mv_utils.get_registration_pairs_from_view_dict(self.view_dict)

        pair_has_overlap = []
        for pair in pairs:

            slices_f, slices_m, lower_f_phys, lower_m_phys = \
                mv_utils.get_overlap_between_pair_of_views(self.view_dict[pair[0]],
                                                            self.view_dict[pair[1]])

            if np.max([[s.start == s.stop for s in slices_f],
                        [s.start == s.stop for s in slices_m]]):
                nonzero_overlap = False
            else: nonzero_overlap = True

            pair_has_overlap.append(nonzero_overlap)

        times = range(self.times_slider.value[0] + 1, self.times_slider.value[1] + 1)

        if not min(pair_has_overlap):
            # message = 'No overlap between views, so no registration needs to be performed.'
            message = 'Not sufficient overlap between views for stitching. Consider stabilizing the tiles instead.'
            notifications.notification_manager.receive_info(message)
            return

        viewims = _utils.load_tiles_from_layers(
            self.viewer.layers,
            self.view_dict,
            [self.regch_slider.value],
            times,
            source_identifier=self.source_identifier
            )
        
        viewims = _utils.add_metadata_to_tiles(viewims, self.view_dict)

        # choose central view as reference
        self.ref_view_index = len(self.views) // 2
        ps = _registration.register_tiles(
                            viewims,
                            self.view_dict,
                            pairs,
                            reg_channel = self.regch_slider.value,
                            times = times,
                            registration_binning=[2] * len(self.view_dict[self.views[0]]['shape']),
                            # registration_binning=None,
                            ref_view_index = self.ref_view_index,
                            )

        psc = _utils.compute_dask_object(ps,
                                         self.viewer,
                                         widgets_to_disable=[self.container],
                                         message='Stitching tiles',
                                         scheduler='threading',
                                         )

        self.params.update(psc)
        self.visualization_type_rbuttons.enabled = True


    def run_fusion(self):

        # assume view_dict, pairs and params are already defined

        times = sorted(self.params.keys())
        channels = range(self.dims['C'][0], self.dims['C'][1])

        viewims = _utils.load_tiles_from_layers(
            self.viewer.layers,
            self.view_dict,
            channels,
            times,
            source_identifier=self.source_identifier
            )

        from napari.utils import progress
        from tqdm.dask import TqdmCallback

        fused_da, fusion_stack_props, field_stack_props = \
            _fusion.fuse_tiles(viewims, self.params, self.view_dict)
        
        fused = zarr_backed_dask_array = da.to_zarr(
            fused_da,
            os.path.join(self.tmpdir.name, fused_da.name+'.zarr'),
            return_stored=True,
            overwrite=True,
            compute=False,
            )

        fused = _utils.compute_dask_object(
            fused,
            self.viewer,
            widgets_to_disable=[self.container],
            message="Fusing tiles",
            scheduler='single-threaded',
            )

        self.viewer.add_image(
            fused,
            channel_axis=0,
            name=[_utils.source_identifier_to_str(self.source_identifier) + '_fused_ch_%03d' %ch
                    for ch in channels],
            colormap='gray',
            blending='additive',
            metadata=dict(
                            view_dict=self.view_dict,
                            stack_props=fusion_stack_props,
                            field_stack_props={t: field_stack_props[it]
                                for it, t in enumerate(times)},
                            view=-1,
                            times=times,
                            processing_state='fused',
                            ndim=len(fusion_stack_props['origin']),
                            )
        )

        self.update_viewer_transformations()


    def load_metadata(self):
        if self.source_identifier_picker.value is None: return

        curr_source_identifier = [si for si in _utils.get_list_of_source_identifiers_from_layers(self.viewer.layers)
            if _utils.source_identifier_to_str(si) == self.source_identifier_picker.value][0]
        
        if self.source_identifier != curr_source_identifier:
            self.params = dict()
            self.visualization_type_rbuttons.value = CHOICE_METADATA
            self.visualization_type_rbuttons.enabled = False

        self.source_identifier = curr_source_identifier
        if self.source_identifier is None:
            notifications.notification_manager.receive_info('No CZI file loaded.')
            return

        self.dims = _file_utils.get_dims_from_multitile_czi(self.source_identifier['filename'],
                                                            self.source_identifier['sample_index'])
        
        self.times_slider.min, self.times_slider.max = self.dims['T'][0] - 1, self.dims['T'][1] - 1
        self.times_slider.value = (self.dims['T'][0] - 1, self.dims['T'][0])
        # self.times_slider.value = (self.dims['T'][0] - 1, self.dims['T'][-1] - 1)

        self.regch_slider.min, self.regch_slider.max = self.dims['C'][0], self.dims['C'][1] - 1
        # self.regch_slider.value = self.dims['C'][0]

        from collections.abc import Iterable
        for w in self.reg_widgets + self.fusion_widgets:
            if isinstance(w, Iterable):
                for sw in w:
                    sw.enabled = True
            w.enabled = True

        max_project = False

        self.view_dict = _file_utils.build_view_dict_from_multitile_czi(
            filename=self.source_identifier['filename'],
            sample_index=self.source_identifier['sample_index'],
            max_project=max_project)
        self.views = np.array([view for view in sorted(self.view_dict.keys())])


    def link_channel_layers(self):

        if self.source_identifier is None:
            return

        # link channel layers
        from napari.experimental import link_layers
        for ch in range(self.dims['C'][0], self.dims['C'][1]):

            layers_to_link = [_utils.get_layer_from_source_identifier_view_and_ch(
                self.viewer.layers, self.source_identifier, view, ch)
                    for view in range(self.dims['M'][0], self.dims['M'][1])]

            layers_to_link = [l for l in layers_to_link if l is not None]
            if len(layers_to_link):
                link_layers(layers_to_link, ('contrast_limits', 'visible'))


    def __del__(self):
        print('deleting widget')

        # clean up callbacks
        self.viewer.layers.events.changed.disconnect(self.update_metadata)
        self.viewer.layers.events.inserted.disconnect(self.link_channel_layers)
        self.viewer.dims.events.disconnect(self.update_viewer_transformations)
        self.viewer.layers.events.inserted.disconnect(self.on_layer_inserted)
        self.viewer.layers.events.removed.disconnect(self.on_layer_inserted)


# simple widget to reload the plugin during development
def reload_plugin_widget(viewer: "napari.Viewer"):
    import importlib
    from napari_stitcher import _widget, _utils, _reader, _fusion, _registration
    _widget = importlib.reload(_widget)
    _utils = importlib.reload(_utils)
    _reader = importlib.reload(_reader)
    _fusion = importlib.reload(_fusion)
    _registration = importlib.reload(_registration)

    from mvregfus import mv_visualization, mv_utils, io_utils
    mv_visualization = importlib.reload(mv_visualization)
    mv_utils = importlib.reload(mv_utils)
    io_utils = importlib.reload(io_utils)
    
    # viewer.window.remove_dock_widget('all')
    # viewer.events.disconnect()
    viewer.layers.events.disconnect()
    viewer.dims.events.disconnect()
    viewer.window.add_dock_widget(_widget.StitcherQWidget(viewer))
    

if __name__ == "__main__":
    import napari

    # filename = "/Users/malbert/software/napari-stitcher/image-datasets/04_stretch-01_AcquisitionBlock2_pt2.czi"
    # filename = "/Users/malbert/software/napari-stitcher/image-datasets/yu_220829_WT_quail_st6_x10_zoom0.7_1x3_488ZO1-568Sox2-647Tbra.czi"

    filename = "/Users/malbert/software/napari-stitcher/image-datasets/arthur_20220621_premovie_dish2-max.czi"
    # filename = "/Users/malbert/software/napari-stitcher/image-datasets/MAX_LSM900.czi"

    viewer = napari.Viewer()
    
    viewer.open(filename)
    # viewer.open("/Users/malbert/software/napari-stitcher/image-datasets/arthur_20220609_WT_emb2_5X_part1_max.czi")

    wdg = StitcherQWidget(viewer)
    viewer.window.add_dock_widget(wdg)

    napari.run()