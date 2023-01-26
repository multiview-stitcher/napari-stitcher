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

from napari_stitcher import _utils

if TYPE_CHECKING:
    import napari


class StitcherQWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        self.setLayout(QVBoxLayout())
        default_input_file = pathlib.Path(__file__).parents[2]\
            /'image-datasets/arthur_20220609_WT_emb2_5X_part1_max.czi'
        
        self.file_picker = widgets.FileEdit(label='Input file:', value=default_input_file)
        self.outdir_picker = widgets.FileEdit(label='Output dir:',
                value=default_input_file.joinpath(default_input_file.name[:-4]+'_stitched'), mode='r')

        self.button_load_metadata = widgets.Button(text='Load metadata')

        self.dimension_rbuttons = widgets.RadioButtons(
            choices=['2D', '3D'], label="Process in:", value="2D", enabled=False, orientation='horizontal')
        
        self.times_slider = widgets.RangeSlider(min=0, max=1, label='Timepoints', enabled=False)
        self.regch_slider = widgets.Slider(min=0, max=1, label='Reg channel', enabled=False)

        self.button_visualize_input = widgets.Button(text='Visualize input', enabled=False)
        self.button_run = widgets.Button(text='Stitch', enabled=False)

        self.visualization_type_rbuttons = widgets.RadioButtons(
            choices=['Metadata', 'Registered'], label="Show:", value="Metadata", enabled=False,
            orientation='horizontal')

        self.vis_ch_slider = widgets.Slider(min=0, max=1, label='Channel', enabled=False)
        self.vis_times_slider = widgets.Slider(min=0, max=1, label='Timepoint', enabled=False)

        self.loading_widgets = [self.file_picker,
                                self.outdir_picker,
                                self.button_load_metadata,
                                ]
        self.reg_setting_widgets = [self.dimension_rbuttons,
                                    self.times_slider,
                                    self.regch_slider,
                                    self.button_run,
                                    ]
        self.visualization_widgets = [self.button_visualize_input,
                                      self.visualization_type_rbuttons,
                                      self.vis_ch_slider,
                                      self.vis_times_slider,
                                      ]

        self.container = widgets.VBox(widgets=
                            self.loading_widgets+\
                            self.reg_setting_widgets+\
                            self.visualization_widgets
                            )
                                        
        self.layout().addWidget(self.container.native)

        # clear napari layers
        # self.file_picker
        @self.button_visualize_input.changed.connect

        @self.button_visualize_input.changed.connect
        def visualize_views(value: str):

            reset_view_at_end = False
            if not len(napari_viewer.layers):
                reset_view_at_end = True

            # napari_viewer.layers.clear()

            # filename = self.file_picker.value
            max_project = self.dimension_rbuttons.value == '2D'

            view_stack_props = [self.view_dict[view] for view in self.views]
            # input_views = np.array([io_utils.read_tile_from_multitile_czi(filename, view,
            #                         channel_index=self.vis_ch_slider.value,
            #                         time_index=self.vis_times_slider.value,
            #                         max_project=max_project)
            #                             for view in self.views])

            # load tiles lazily (not sure if this is working already)
            ch = self.vis_ch_slider.value
            t = self.vis_times_slider.value
            input_views = _utils.load_tiles(self.view_dict, [ch],
                                [t], max_project=max_project)[ch][t]

            # debug whether tiles are loaded lazily
            # def _print_load_tile(x):
            #     print('loading tile')
            #     return x
            # input_views = _utils.apply_recursive_dict(_print_load_tile, input_views)

            input_views = [da.from_delayed(input_views[iiv], shape=self.view_dict[iiv]['shape'],
                                                             dtype=np.uint16)
                                                for iiv in input_views.keys()]

            if self.visualization_type_rbuttons.value == 'Metadata':
                params = None
            else:
                params = np.array([self.params[self.vis_times_slider.value][view]
                                    for view in self.views])

            mv_visualization.visualize_views(input_views, params, None,
                                             view_stack_props,
                                             viewer=napari_viewer,
                                             clims=[0,255],
                                             edge_width=5)
            # self.button_reset_view.enabled=True
            # reset view after first visualization
            if reset_view_at_end:
                napari_viewer.reset_view()


        @self.visualization_type_rbuttons.changed.connect
        @self.vis_ch_slider.changed.connect
        @self.vis_times_slider.changed.connect
        def update_vis(value: str):
            if self.visualization_type_rbuttons.value == 'Metadata':
                self.vis_times_slider.min, self.vis_times_slider.max = self.dims['T'][0], self.dims['T'][1]-1
            else:
                times = self.params.keys()
                self.vis_times_slider.min, self.vis_times_slider.max = min(times), max(times)
            visualize_views('')


        @self.button_run.clicked.connect
        def run_sitching(value: str):

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

            else:

                times = range(self.times_slider.value[0] + 1, self.times_slider.value[1] + 1)
                viewims = _utils.load_tiles(self.view_dict, [self.regch_slider.value],
                                times, max_project=True)

                times = range(self.times_slider.value[0] + 1, self.times_slider.value[1] + 1)
                ps = _utils.register_tiles(
                                    viewims,
                                    self.pairs,
                                    reg_channel = self.regch_slider.value,
                                    times = times,
                                    registration_binning=[2, 2],
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
                # napari_viewer.window._status_bar._toggle_activity_dock(False)

                self.params = psc
                self.visualization_type_rbuttons.enabled = True

            return
        
        @self.button_load_metadata.clicked.connect
        def load_metadata(value: str):

            self.dims = io_utils.get_dims_from_multitile_czi(self.file_picker.value)
            # print(self.dims)

            self.view_dict = io_utils.build_view_dict_from_multitile_czi(self.file_picker.value, max_project=True)
            self.views = np.array([view for view in sorted(self.view_dict.keys())])
            self.pairs = mv_utils.get_registration_pairs_from_view_dict(self.view_dict)
            self.ndim = [2, 3][int(self.dims['Z'][1] > 1)]

            if self.dims['Z'][1] > 1:
                self.dimension_rbuttons.enabled = True
            else:
                self.dimension_rbuttons.enabled = False
            
            self.times_slider.min, self.times_slider.max = self.dims['T'][0] - 1, self.dims['T'][1] - 1
            self.times_slider.value = (self.dims['T'][0] - 1, self.dims['T'][0])

            self.regch_slider.min, self.regch_slider.max = self.dims['C'][0], self.dims['C'][1] - 1
            self.regch_slider.value = self.dims['C'][0]

            self.vis_ch_slider.min, self.vis_ch_slider.max = self.dims['C'][0], self.dims['C'][1]-1
            self.vis_times_slider.min, self.vis_times_slider.max = self.dims['T'][0], self.dims['T'][1]-1

            for w in self.reg_setting_widgets + self.visualization_widgets:
                w.enabled = True
            self.visualization_type_rbuttons.enabled = False


# simple widget to reload the plugin during development
def reload_plugin_widget(viewer: "napari.Viewer"):
    import importlib
    from napari_stitcher import _widget, _utils, _reader
    _widget = importlib.reload(_widget)
    _utils = importlib.reload(_utils)
    _reader = importlib.reload(_reader)

    from mvregfus import mv_visualization, mv_utils, io_utils
    mv_visualization = importlib.reload(mv_visualization)
    mv_utils = importlib.reload(mv_utils)
    io_utils = importlib.reload(io_utils)
    
    # viewer.window.remove_dock_widget('all')
    viewer.window.add_dock_widget(_widget.StitcherQWidget(viewer))
    

