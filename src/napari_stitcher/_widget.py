"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING
import pathlib, os

import numpy as np
import dask
import dask.array as da

from napari.utils import notifications

from magicgui import magic_factory
from qtpy.QtWidgets import QVBoxLayout, QPushButton, QWidget

from magicgui import magicgui

from magicgui import widgets

from mvregfus import io_utils, mv_utils, mv_visualization

from napari_stitcher import _utils, _registration, _fusion

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
        
        self.source_path = _utils.get_source_path_from_viewer(napari_viewer)

        # if self.source_path is not None:
        #     default_outdir = self.source_path+'_stitched'
        # else:
        #     default_outdir = os.path.join('.', 'napari_stitcher_output')

        # self.outdir_picker = widgets.FileEdit(label='Output dir:',
        #         value=default_outdir, mode='r')

        # self.button_load_metadata = widgets.Button(text='Load metadata')
        self.source_path_picker = widgets.ComboBox(
            label='Input file: ',
            choices=[(os.path.basename(p), p)
                for p in _utils.get_list_of_source_paths_from_layers(napari_viewer.layers)])

        # self.dimension_rbuttons = widgets.RadioButtons(
        #     choices=['2D', '3D'], label="Process in:", value="2D", enabled=False, orientation='horizontal')
        
        self.times_slider = widgets.RangeSlider(min=0, max=1, label='Timepoints:', enabled=False)
        self.regch_slider = widgets.Slider(min=0, max=1, label='Reg channel:', enabled=False)

        # self.button_visualize_input = widgets.Button(text='Visualize input', enabled=False)
        self.button_register = widgets.Button(text='Register', enabled=False)

        self.visualization_type_rbuttons = widgets.RadioButtons(
            choices=[CHOICE_METADATA, CHOICE_REGISTERED], label="Show:", value=CHOICE_METADATA, enabled=False,
            orientation='horizontal')

        self.button_fuse = widgets.Button(text='Fuse', enabled=False)

        self.loading_widgets = [
                            self.source_path_picker,
                            # self.button_load_metadata,
                            # self.outdir_picker,
                            ]

        self.reg_widgets = [
                            # self.dimension_rbuttons,
                            self.times_slider,
                            self.regch_slider,
                            self.button_register,
                            ]

        self.visualization_widgets = [
                            self.visualization_type_rbuttons,
        ]
        # self.visualization_widgets = [self.button_visualize_input,
        #                               self.visualization_type_rbuttons,
        #                               self.vis_ch_slider,
        #                               self.vis_times_slider,
        #                               ]

        self.fusion_widgets = [
                            self.button_fuse,
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

        @self.visualization_type_rbuttons.changed.connect
        @self.viewer.dims.events.connect
        def update_viewer_transformations(event):
            """
            set transformations for current timepoint
            """

            for l in self.viewer.layers:

                # if l.visible is False: continue

                # unfused layers
                if (l.source is not None and (l.source.path == self.source_path)):

                    if 'times' in l.metadata.keys() and len(l.metadata['times']) > 1:
                        curr_tp = self.viewer.dims.current_step[0]
                    else:
                        curr_tp = 0

                    if self.visualization_type_rbuttons.value == 'Registered'\
                            and curr_tp not in self.params:
                        notifications.notification_manager.receive_info(
                            'Timepoint %s: no parameters available, register first.' % curr_tp)
                        self.visualization_type_rbuttons.value == CHOICE_METADATA
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


        @self.viewer.layers.events.inserted.connect
        @self.viewer.layers.events.removed.connect
        def on_layer_inserted(event):
            self.source_path_picker.choices = _utils.get_list_of_source_paths_from_layers(napari_viewer.layers)


        @self.button_register.clicked.connect
        def run_registration(value: str):

            # max_project = True if self.dimension_rbuttons.value == '2D' else False
            max_project = False

            self.view_dict = io_utils.build_view_dict_from_multitile_czi(self.source_path, max_project=max_project)
            self.views = np.array([view for view in sorted(self.view_dict.keys())])
            self.pairs = mv_utils.get_registration_pairs_from_view_dict(self.view_dict)

            pair_requires_registration = []
            for pair in self.pairs:

                slices_f, slices_m, lower_f_phys, lower_m_phys = \
                    mv_utils.get_overlap_between_pair_of_views(self.view_dict[pair[0]],
                                                               self.view_dict[pair[1]])

                if np.max([[s.start == s.stop for s in slices_f],
                           [s.start == s.stop for s in slices_m]]):
                    nonzero_overlap = False
                else: nonzero_overlap = True

                pair_requires_registration.append(nonzero_overlap)
            
            if not max(pair_requires_registration):
                message = 'No overlap between views, so no registration needs to be performed.'
                notifications.notification_manager.receive_info(message)
                return

            times = range(self.times_slider.value[0] + 1, self.times_slider.value[1] + 1)
            viewims = _utils.load_tiles(self.view_dict, [self.regch_slider.value],
                            times, max_project=max_project)

            times = range(self.times_slider.value[0] + 1, self.times_slider.value[1] + 1)

            # choose central view as reference
            self.ref_view_index = len(self.views) // 2
            ps = _registration.register_tiles(
                                viewims,
                                self.pairs,
                                reg_channel = self.regch_slider.value,
                                times = times,
                                registration_binning=[2] * len(self.view_dict[self.views[0]]['shape']),
                                ref_view_index = self.ref_view_index,
                                )

            from napari.utils import progress
            from tqdm.dask import TqdmCallback

            # make evident that processing is happening by disabling widgets
            # this command raises a warning regarding accessing a private attribute
            napari_viewer.window._status_bar._toggle_activity_dock(True)
            enabled = [w.enabled for w in self.container]
            for w in self.container:
                w.enabled = False

            with TqdmCallback(tqdm_class=progress, desc="Registering tiles"):
                psc = dask.compute(ps, scheduler='threading')[0]

            # enable widgets again
            for w, e in zip(self.container, enabled):
                w.enabled = e

            # this command raises a warning regarding accessing a private attribute
            # napari_viewer.window._status_bar._toggle_activity_dock(False)

            self.params.update(psc)
            self.visualization_type_rbuttons.enabled = True

        @self.button_fuse.clicked.connect
        def run_fusion(value: str):

            # assume view_dict, pairs and params are already defined

            times = sorted(self.params.keys())
            channels = range(self.dims['C'][0], self.dims['C'][1])

            viewims = _utils.load_tiles(
                            self.view_dict,
                            channels,
                            times, max_project=False)

            from napari.utils import progress
            from tqdm.dask import TqdmCallback

            fused_da, fusion_stack_props, field_stack_props = \
                _fusion.fuse_tiles(viewims, self.params, self.view_dict)

            with TqdmCallback(tqdm_class=progress, desc="Fusing tiles"):
                fused = dask.compute(fused_da, scheduler='threading')

            # self.translation_fusion_rel_to_metadata.update(
            #     {t: fusion_stack_props['origin'] - \
            #         np.min([self.view_dict[v]['origin'] for v in self.view_dict.keys()], 0)
            #             for t in times})

            self.viewer.add_image(fused,
                channel_axis=0,
                name=[os.path.basename(self.source_path)[:-4] + '_fused_ch_%03d' %ch
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

            update_viewer_transformations(None)

        
        # @self.button_load_metadata.clicked.connect
        @self.source_path_picker.changed.connect
        def load_metadata():

            # curr_source_path = _utils.get_source_path_from_viewer(self.viewer)
            curr_source_path = self.source_path_picker.value
            if self.source_path != curr_source_path:
                self.params = dict()
                self.visualization_type_rbuttons.value = CHOICE_METADATA
                self.visualization_type_rbuttons.enabled = False

            self.source_path = curr_source_path
            if self.source_path is None:
                notifications.notification_manager.receive_info('No CZI file loaded.')
                return

            self.dims = io_utils.get_dims_from_multitile_czi(self.source_path)
            # print(self.dims)

            # self.view_dict = io_utils.build_view_dict_from_multitile_czi(self.source_path, max_project=True)
            # self.views = np.array([view for view in sorted(self.view_dict.keys())])
            # self.pairs = mv_utils.get_registration_pairs_from_view_dict(self.view_dict)
            # self.ndim = [2, 3][int(self.dims['Z'][1] > 1)]

            # if self.dims['Z'][1] > 1:
            #     self.dimension_rbuttons.enabled = True
            # else:
            #     self.dimension_rbuttons.enabled = False
            
            self.times_slider.min, self.times_slider.max = self.dims['T'][0] - 1, self.dims['T'][1] - 1
            self.times_slider.value = (self.dims['T'][0] - 1, self.dims['T'][0])
            # self.times_slider.value = (self.dims['T'][0] - 1, self.dims['T'][-1] - 1)

            self.regch_slider.min, self.regch_slider.max = self.dims['C'][0], self.dims['C'][1] - 1
            # self.regch_slider.value = self.dims['C'][0]

            for w in self.reg_widgets + self.fusion_widgets:
                w.enabled = True


        @self.viewer.layers.events.inserted.connect
        def link_channel_layers():

            if self.source_path is None:
                return

            # link channel layers
            from napari.experimental import link_layers
            for ch in range(self.dims['C'][0], self.dims['C'][1]):

                layers_to_link = [_utils.get_layer_from_source_path_view_and_ch(
                    self.viewer.layers, self.source_path, view, ch)
                        for view in range(self.dims['M'][0], self.dims['M'][1])]
                # import pdb; pdb.set_trace()
                layers_to_link = [l for l in layers_to_link if l is not None]
                # print([l is None for l in layers_to_link if l is not None])
                if len(layers_to_link):
                    link_layers(layers_to_link, ('contrast_limits', 'visible'))


        # run on startup
        load_metadata()
        link_channel_layers()


    def __del__(self):
        print('deleting widget')
        # self.viewer.layers.events.changed.disconnect(self.update_metadata)


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
    filename = "/Users/malbert/software/napari-stitcher/image-datasets/yu_220829_WT_quail_st6_x10_zoom0.7_1x3_488ZO1-568Sox2-647Tbra.czi"

    # filename = "/Users/malbert/software/napari-stitcher/image-datasets/arthur_20220621_premovie_dish2-max.czi"

    viewer = napari.Viewer()
    viewer.open(filename)
    # viewer.open("/Users/malbert/software/napari-stitcher/image-datasets/arthur_20220609_WT_emb2_5X_part1_max.czi")

    wdg = StitcherQWidget(viewer)
    viewer.window.add_dock_widget(wdg)
    napari.run()