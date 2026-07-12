"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING
import os, tempfile, sys, shutil
from collections.abc import Iterable

import numpy as np

from napari.utils import notifications

from magicgui import widgets
from qtpy.QtWidgets import QVBoxLayout, QHBoxLayout, QWidget, QTabWidget, QLabel
from qtpy.QtGui import QPixmap

from multiview_stitcher import (
    registration,
    fusion,
    spatial_image_utils,
    msi_utils,
    param_utils,
    )
from napari.layers import Image, Labels

from napari_stitcher import _reader, viewer_utils, _utils

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

        self.button_load_layers_sel = widgets.Button(text='Selected')
        self.button_load_layers_all = widgets.Button(text='All')
        self.buttons_load_layers = widgets.HBox(
            widgets=[
                widgets.Label(
                    value='Load layers:',
                    tooltip='Load layers to stitch from the napari layer list.'),
                self.button_load_layers_sel,
                self.button_load_layers_all,
                ]
                    )
        self.layers_selection = widgets.Select(choices=[])
        self.load_layers_box = widgets.VBox(widgets=\
                                            [
            self.buttons_load_layers,
            self.layers_selection,
                                            ],
                                            # label='Loaded\nlayers:',
                                            )

        self.times_slider = widgets.RangeSlider(
            min=-1, max=0, label='Timepoints:',
            tooltip='Timepoints to process. Because the two sliders cannot coincide, positions are a bit criptic: E.g.\n(-1, 0) means timepoint 0 is processed\n(3, 5) means timepoints 4 and 5 are processed')
        
        self.reg_ch_picker = widgets.ComboBox(
            label='Reg channel: ',
            choices=[],
            tooltip='Choose a file to process using napari-stitcher.')
        
        self.custom_reg_binning = widgets.CheckBox(value=False, text='Use custom binning for registration')
        self.x_reg_binning = widgets.Slider(value=1, min=1, max=10, label='X binning:')
        self.y_reg_binning = widgets.Slider(value=1, min=1, max=10, label='Y binning:')

        self.custom_fuse_binning = widgets.CheckBox(value=False, text='Use custom binning for fusion')
        self.x_fuse_binning = widgets.Slider(value=1, min=1, max=10, label='X binning:')
        self.y_fuse_binning = widgets.Slider(value=1, min=1, max=10, label='Y binning:')

        self.do_quality_filter = widgets.CheckBox(value=False, text='Filter registrations by quality')
        self.quality_threshold = widgets.FloatSlider(value=0.2, min=0, max=1, label='Quality threshold:')
        
        # widget giving options between different strings
        self.pair_pruning_method = widgets.ComboBox(
            choices=[
                'None',
                'Only keep axis-aligned',
                'Alternating pattern'
                ],
            value='None',
            label='Pre-registration pruning method:',
            tooltip='Choose the method to prune pairs of tiles before registration. '+\
            'By default, all pairs of overlapping views are registered (None). '+\
            'Recommended for best performance on regular grids: "Only keep axis-aligned".')

        self.pair_pruning_method_mapping = {
            'None': None,
            'Only keep axis-aligned': 'keep_axis_aligned',
            'Alternating pattern': 'alternating_pattern',
        }

        self.reg_method = widgets.ComboBox(
            choices=['Phase Correlation', 'ITKElastix'],
            value='Phase Correlation',
            label='Registration method:',
            tooltip='Choose the pairwise registration method.\n'
                    '"Phase Correlation" is fast and works well for translation.\n'
                    '"ITKElastix" supports more transform types but requires the itk-elastix package.')

        self.antspy_transform_types = widgets.Select(
            choices=['Translation', 'Rigid', 'Affine'],
            value=['Translation', 'Rigid'],
            label='Transform types:',
            tooltip='Sequence of transform types applied in order. The last selected type is also used for global optimization.')

        self.reg_method.changed.connect(self._on_reg_method_changed)
        self._on_reg_method_changed()

        self.button_stitch = widgets.Button(text='Register',
            tooltip='Use the overlaps between tiles to determine their relative positions.')
        
        # self.button_stabilize = widgets.Button(text='Stabilize',
        #     tooltip='Use time lapse information to stabilize each tile over time,'+\
        #             'eliminating abrupt shifts between frames. No tile overlap needed.')

        self.visualization_type_rbuttons = widgets.RadioButtons(
            choices=[CHOICE_METADATA, CHOICE_REGISTERED],
            label="Show:",
            value=CHOICE_METADATA,
            orientation='horizontal')

        self.button_fuse = widgets.Button(text='Fuse',
            tooltip='Fuse the tiles using the parameters obtained'+\
                    'from stitching or stabilization.\nCombines all'+\
                    'tiles and timepoints into a single image, smoothly'+\
                    'blending the overlaps and filling in gaps.')

        self.loading_widgets = [
                            self.load_layers_box,
                            ]

        self.reg_config_widgets_basic = [
                            self.times_slider,
                            self.reg_ch_picker,
                            ]

        self.reg_config_widgets_advanced = [
                            self.custom_reg_binning,
                            self.x_reg_binning,
                            self.y_reg_binning,
                            self.custom_fuse_binning,
                            self.x_fuse_binning,
                            self.y_fuse_binning,
                            self.do_quality_filter,
                            self.quality_threshold,
                            self.pair_pruning_method,
        ]

        self.reg_config_widgets_method = [
                            self.reg_method,
                            self.antspy_transform_types,
        ]

        self.reg_config_widgets = self.reg_config_widgets_basic + self.reg_config_widgets_advanced + self.reg_config_widgets_method

        # Initialize tab screen 
        self.reg_config_widgets_tabs = QTabWidget() 
        self.reg_config_widgets_tabs.resize(300, 200) 
   
        # Add tabs 
        self.reg_config_widgets_tabs.addTab(
            widgets.VBox(widgets=self.reg_config_widgets_basic).native, "Basic") 
        self.reg_config_widgets_tabs.addTab(
            widgets.VBox(widgets=self.reg_config_widgets_advanced).native, "More")
        self.reg_config_widgets_tabs.addTab(
            widgets.VBox(widgets=self.reg_config_widgets_method).native, "Method")

        self.visualization_widgets = [
                            self.visualization_type_rbuttons,
        ]

        self.all_widgets = \
            self.loading_widgets +\
            self.reg_config_widgets +\
            [self.button_stitch] +\
            [self.button_fuse] +\
            self.visualization_widgets

        self.container = QWidget()
        self.container.setLayout(QVBoxLayout())

        for w in self.loading_widgets:
            if hasattr(w, 'native'):
                self.container.layout().addWidget(w.native)
            else:
                self.container.layout().addWidget(w)

        self.container.layout().addWidget(self.reg_config_widgets_tabs)
        self.container.layout().addWidget(self.button_stitch.native)
        self.container.layout().addWidget(self.button_fuse.native)

        # add horizontal widget with visualization options
        self.visualization_widgets_qt = QWidget()
        self.visualization_widgets_qt.setLayout(QHBoxLayout())
        self.visualization_widgets_qt.layout().addWidget(QLabel('Show: '))
        self.visualization_widgets_qt.layout().addWidget(self.visualization_widgets[0].native)

        self.container.layout().addWidget(self.visualization_widgets_qt)

        self.container.setMinimumWidth = 50
        self.layout().addWidget(self.container)

        # disable all widgets (apart from loading) until layers are loaded
        for w in self.reg_config_widgets + self.visualization_widgets +\
            [self.button_stitch, self.button_fuse]:
            w.enabled = False
            if isinstance(w, Iterable):
                for sw in w:
                    sw.enabled = False
            w.enabled = False

        # initialize registration parameter dict
        self.input_layers= []
        self.msims = {}
        self.fused_layers = []
        self.params = dict()

        # flag to suppress watch_layer_changes during programmatic affine updates
        self._updating_viewer = False
        # last timepoint that was applied to the viewer; used to skip no-op current_step events
        self._last_applied_tp = None

        # create temporary directory for storing dask arrays
        self.tmpdir = tempfile.TemporaryDirectory()
        
        self.visualization_type_rbuttons.changed.connect(self.update_viewer_transformations)
        self.viewer.dims.events.current_step.connect(self.update_viewer_transformations)

        self.button_stitch.clicked.connect(self.run_registration)
        # self.button_stabilize.clicked.connect(self.run_stabilization)
        self.button_fuse.clicked.connect(self.run_fusion)

        self.button_load_layers_all.clicked.connect(self.load_layers_all)
        self.button_load_layers_sel.clicked.connect(self.load_layers_sel)


    def _on_reg_method_changed(self, event=None):
        """Show/hide transform type widget based on the selected method."""
        self.antspy_transform_types.visible = self.reg_method.value == 'ITKElastix'

    def update_viewer_transformations(self, event=None):
        """
        set transformations
        - for current timepoint
        - for each (compatible) layer loaded in viewer

        Called from exactly two sources:
          1. viewer.dims.events.current_step  (timepoint scroll)
          2. visualization_type_rbuttons.changed  (Show toggle)
        """

        # When called from a current_step event:
        #   - only proceed after registration has been performed
        #   - only proceed when the timepoint actually changed (transform-mode
        #     interactions also fire current_step without changing the tp)
        if hasattr(event, 'type'):
            if not self.visualization_type_rbuttons.enabled:
                return
            # Compute the candidate tp now so we can compare
            # (replicated from the block below; simims may not be loaded yet)
            if not len(self.msims):
                return
            _sims_check = [msi_utils.get_sim_from_msim(self.msims[l.name])
                           for l in self.viewer.layers if l.name in self.msims]
            if not _sims_check:
                return
            _highest_sdim = max(
                len(spatial_image_utils.get_spatial_dims_from_sim(s))
                for s in _sims_check)
            _candidate_tp = (
                self.viewer.dims.current_step[-_highest_sdim - 1]
                if len(self.viewer.dims.current_step) > _highest_sdim
                else 0)
            if _candidate_tp == self._last_applied_tp:
                return

        if not len(self.msims): return

        compatible_layers = [l for l in self.viewer.layers
                             if l.name in self.msims.keys()]
        
        if not len(compatible_layers): return
        
        sims = [msi_utils.get_sim_from_msim(self.msims[l.name])
                for l in compatible_layers]

        # determine spatial dimensions from layers
        all_spatial_dims = [spatial_image_utils.get_spatial_dims_from_sim(
            sims[il])
            for il, l in enumerate(compatible_layers)]
        
        highest_sdim = max([len(sdim) for sdim in all_spatial_dims])

        # get curr tp
        # handle possibility that there had been no T dimension
        # when collecting sims from layers

        if len(self.viewer.dims.current_step) > highest_sdim:
            curr_tp = self.viewer.dims.current_step[-highest_sdim-1]
        else:
            curr_tp = 0

        if self.visualization_type_rbuttons.value == CHOICE_METADATA:
            transform_key=_reader.METADATA_TRANSFORM_KEY
        else:
            transform_key = 'affine_registered'

        self._last_applied_tp = curr_tp
        self._updating_viewer = True
        try:
            for il, l in enumerate(compatible_layers):

                try:
                    params = spatial_image_utils.get_affine_from_sim(
                        sims[il], transform_key=transform_key
                        )
                except:
                    continue

                try:
                    p = np.array(params.sel(t=sims[il].coords['t'][curr_tp])).squeeze()
                    if np.isnan(p).any():
                        raise(Exception())
                except:
                    notifications.notification_manager.receive_info(
                        'Timepoint %s: no parameters available, register first.' % curr_tp)
                    continue

                ndim_layer_data = l.ndim

                # if stitcher sim has more dimensions than layer data (i.e. time)
                vis_p = p[-(ndim_layer_data + 1):, -(ndim_layer_data + 1):]

                # if layer data has more dimensions than stitcher sim
                full_vis_p = np.eye(ndim_layer_data + 1)
                full_vis_p[-len(vis_p):, -len(vis_p):] = vis_p

                l.affine = full_vis_p
        finally:
            self._updating_viewer = False


    def _capture_layer_transforms_to_msims(self):
        """
        Capture the current layer affines into affine_metadata in msims.
        Called before registration/fusion when showing Original transforms, so
        any manual layer adjustments made by the user are used as the starting point.
        """
        for l in self.input_layers:
            if l.name not in self.msims:
                continue
            msim = self.msims[l.name]
            sim = msi_utils.get_sim_from_msim(msim)
            ndim = spatial_image_utils.get_ndim_from_sim(sim)
            affine = np.array(l.affine.affine_matrix)[-(ndim + 1):, -(ndim + 1):]
            t_coords = sim.coords['t'] if 't' in sim.dims else None
            affine_xr = param_utils.affine_to_xaffine(affine, t_coords=t_coords)
            msi_utils.set_affine_transform(msim, affine_xr, transform_key='affine_metadata')

    def _update_registered_param_for_current_tp(self, l):
        """
        Update affine_registered for the current timepoint from l.affine.
        Only the current timepoint is modified; all others remain unchanged.
        Called live when the user manually transforms a layer while showing Registered.
        """
        if l.name not in self.msims:
            return
        msim = self.msims[l.name]
        sim = msi_utils.get_sim_from_msim(msim)
        ndim = spatial_image_utils.get_ndim_from_sim(sim)

        # Determine current timepoint (mirrors the logic in update_viewer_transformations)
        sdims = spatial_image_utils.get_spatial_dims_from_sim(sim)
        if len(self.viewer.dims.current_step) > len(sdims):
            curr_tp = self.viewer.dims.current_step[-len(sdims) - 1]
        else:
            curr_tp = 0

        curr_affine = np.array(l.affine.affine_matrix)[-(ndim + 1):, -(ndim + 1):]

        try:
            existing_params = spatial_image_utils.get_affine_from_sim(
                sim, transform_key='affine_registered').copy()
        except Exception:
            return  # not registered yet

        if 't' in existing_params.dims:
            t_val = sim.coords['t'][curr_tp]
            existing_params.loc[{'t': t_val}] = curr_affine
        else:
            existing_params = param_utils.affine_to_xaffine(curr_affine, t_coords=None)

        msi_utils.set_affine_transform(msim, existing_params, transform_key='affine_registered')

    def _promote_registered_to_metadata(self):
        """
        Copy affine_registered → affine_metadata for all msims so that a
        subsequent registration uses the manually corrected positions as its
        starting point rather than the original metadata positions.
        """
        for l_name, msim in self.msims.items():
            sim = msi_utils.get_sim_from_msim(msim)
            try:
                registered_params = spatial_image_utils.get_affine_from_sim(
                    sim, transform_key='affine_registered')
            except Exception:
                continue
            msi_utils.set_affine_transform(
                msim, registered_params.copy(), transform_key='affine_metadata')

    def run_registration(self):            

        # Promote the current starting-point transforms into affine_metadata so
        # that registration always uses the most up-to-date positions:
        #   - Showing Original (or pre-registration): capture layer affines → affine_metadata
        #   - Showing Registered: copy affine_registered → affine_metadata
        if (self.visualization_type_rbuttons.enabled and
                self.visualization_type_rbuttons.value == CHOICE_REGISTERED):
            self._promote_registered_to_metadata()
        else:
            self._capture_layer_transforms_to_msims()

        # select layers corresponding to the chosen registration channel
        msims_dict = {_utils.get_str_unique_to_view_from_layer_name(lname): msim
                      for lname, msim in self.msims.items()
                      if self.reg_ch_picker.value in msi_utils.get_sim_from_msim(msim).coords['c']}
        
        # sort layers by name
        sorted_lnames = sorted(list(msims_dict.keys()))

        msims = [msims_dict[lname] for lname in sorted_lnames]

        msims = [msi_utils.multiscale_sel_coords(msim,
                {'t': [msi_utils.get_sim_from_msim(msim).coords['t'][it]
                         for it in range(self.times_slider.value[0] + 1,
                                         self.times_slider.value[1] + 1)]})
                  for msim in msims]

        # with _utils.TemporarilyDisabledWidgets([self.container]),\
        with _utils.TemporarilyDisabledWidgets(self.all_widgets),\
            _utils.VisibleActivityDock(self.viewer),\
            _utils.TqdmCallback(tqdm_class=_utils.progress,
                                desc='Registering tiles', bar_format=" "):
            
            if self.custom_reg_binning.value:
                registration_binning = {'y': self.x_reg_binning.value, 'x': self.y_reg_binning.value}
            else:
                registration_binning = None

            if self.reg_method.value == 'ITKElastix':
                pairwise_reg_func = registration.registration_ITKElastix
                transform_types = list(self.antspy_transform_types.value)
                pairwise_reg_func_kwargs = {'transform_types': transform_types}
                groupwise_resolution_kwargs = {'transform': transform_types[-1].lower()}
            else:
                pairwise_reg_func = registration.phase_correlation_registration
                pairwise_reg_func_kwargs = None
                groupwise_resolution_kwargs = None

            params = registration.register(
                msims,
                registration_binning=registration_binning,
                pairwise_reg_func=pairwise_reg_func,
                pairwise_reg_func_kwargs=pairwise_reg_func_kwargs,
                groupwise_resolution_kwargs=groupwise_resolution_kwargs,
                pre_registration_pruning_method=self.pair_pruning_method_mapping[self.pair_pruning_method.value],
                post_registration_do_quality_filter=self.do_quality_filter.value,
                post_registration_quality_threshold=self.quality_threshold.value,
                transform_key='affine_metadata',
            )

        for lname, msim in self.msims.items():
            params_index = sorted_lnames.index(_utils.get_str_unique_to_view_from_layer_name(lname))
            msi_utils.set_affine_transform(
                msim, params[params_index],
                transform_key='affine_registered', base_transform_key='affine_metadata')
            
        for l in self.input_layers:
            params_index = sorted_lnames.index(_utils.get_str_unique_to_view_from_layer_name(l.name))
            try:
                viewer_utils.set_layer_xaffine(
                    l, params[params_index],
                    transform_key='affine_registered', base_transform_key='affine_metadata')
            except:
                pass
        
        self.visualization_type_rbuttons.enabled = True
        self.visualization_type_rbuttons.value = CHOICE_REGISTERED
        # Always refresh the viewer after registration, even if already showing
        # CHOICE_REGISTERED (setting the same value doesn't fire a changed event).
        self._last_applied_tp = None
        self.update_viewer_transformations()


    def run_fusion(self):

        """
        Split layers into channel groups and fuse each group separately.
        """

        # Capture manual layer adjustments if fusing with original transforms
        if not (self.visualization_type_rbuttons.enabled and
                self.visualization_type_rbuttons.value == CHOICE_REGISTERED):
            self._capture_layer_transforms_to_msims()

        channels = self.reg_ch_picker.choices

        for _, ch in enumerate(channels):

            msims = [msim for _, msim in self.msims.items()
                    if ch in msi_utils.get_sim_from_msim(msim).coords['c']]

            sims = [msi_utils.get_sim_from_msim(msim) for msim in msims]

            sims = [spatial_image_utils.sim_sel_coords(sim,
                    {'t': [sim.coords['t'][it]
                            for it in range(self.times_slider.value[0] + 1,
                                            self.times_slider.value[1] + 1)]})
                    for sim in sims]

            # check which keys are in spacing that are missing in fusion_binning and add them
            if self.custom_fuse_binning.value:
                fusion_binning = {'y': self.x_fuse_binning.value, 'x': self.x_fuse_binning.value}
                fusing_spacing = spatial_image_utils.get_spacing_from_sim(sims[0])
                fusing_spacing = {
                    key: fusing_spacing[key] * fusion_binning[key]
                    for key in fusing_spacing.keys() if key in fusion_binning
                }
            else:
                fusing_spacing = None

            fused = fusion.fuse(
                sims,
                transform_key='affine_registered'
                if self.visualization_type_rbuttons.value == CHOICE_REGISTERED
                else 'affine_metadata',
                output_spacing=fusing_spacing,
            )

            fused = fused.expand_dims({'c': [sims[0].coords['c'].values]})

            mfused = msi_utils.get_msim_from_sim(fused, scale_factors=[])

            tmp_fused_path = os.path.join(self.tmpdir.name, 'fused_%s.zarr' %ch)
            if os.path.exists(tmp_fused_path):
                shutil.rmtree(tmp_fused_path)

            with _utils.TemporarilyDisabledWidgets(self.all_widgets),\
                _utils.VisibleActivityDock(self.viewer),\
                _utils.TqdmCallback(tqdm_class=_utils.progress,
                                    desc='Fusing tiles of channel %s' %ch, bar_format=" "):
                
                mfused.to_zarr(tmp_fused_path)

            mfused = msi_utils.multiscale_spatial_image_from_zarr(tmp_fused_path, chunks={})

            fused_ch_layer_tuple = viewer_utils.create_image_layer_tuples_from_msim(
                mfused,
                colormap=None,
                name_prefix='fused',
            )[0]

            fused_layer = self.viewer.add_image(fused_ch_layer_tuple[0], **fused_ch_layer_tuple[1])
        
            self.fused_layers.append(fused_layer)


    def reset(self):
            
        self.msims = {}
        self.params = dict()
        self.reg_ch_picker.choices = ()
        self.visualization_type_rbuttons.value = CHOICE_METADATA
        self.times_slider.min, self.times_slider.max = (-1, 0)
        self.times_slider.value = (-1, 0)
        self.input_layers = []
        self.fused_layers = []
        self._last_applied_tp = None


    def load_metadata(self):
        
        reference_sim = msi_utils.get_sim_from_msim(self.msims[self.input_layers[0].name])
        
        # assume dims are the same for all layers
        if 't' in reference_sim.dims:
            self.times_slider.min = -1
            self.times_slider.max = len(reference_sim.coords['t']) - 1
            self.times_slider.value = self.times_slider.min, self.times_slider.max

        if 'c' in reference_sim.coords.keys():
            self.reg_ch_picker.choices = np.unique([
                _utils.get_str_unique_to_ch_from_sim_coords(msi_utils.get_sim_from_msim(msim).coords)
                for l_name, msim in self.msims.items()])
            self.reg_ch_picker.value = self.reg_ch_picker.choices[0]

        for w in self.reg_config_widgets + [self.button_stitch, self.button_fuse]:
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

        self.load_layers(self.viewer.layers.selection)


    def load_layers(self, layers):

        self.reset()
        self.viewer.layers.unlink_layers()

        layers = [l for l in layers if isinstance(l, (Image, Labels))]

        if len(layers) == 0:
            notifications.notification_manager.receive_info(
                'No compatible layers selected or available.'
            )
            return

        self.input_layers = layers
        self.layers_selection.choices = sorted([l.name for l in layers])

        # load in layers as msims
        for l in self.input_layers:
            msim = viewer_utils.image_layer_to_msim(l, self.viewer)
            
            if 'c' in msim['scale0/image'].dims:
                notifications.notification_manager.receive_info(
                    "Layer '%s' has more than one channel. Consider splitting the stack (right click on layer -> 'Split Stack')." %l.name
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

        if len(layers) / number_of_channels > 1:
            self.link_view_layers(layers)

        # if loaded layer changes, update msim
        for l in self.input_layers:
            l.events.connect(self.watch_layer_changes)

        self.load_metadata()


    def watch_layer_changes(self, event):
        """
        Watch user-initiated layer transform changes and update stored parameters.

        - Pre-registration or showing Original: do nothing here; transforms are
          captured at registration/fusion time via _capture_layer_transforms_to_msims.
        - Post-registration, showing Registered: live-update affine_registered for
          the current timepoint only, leaving other timepoints unchanged.
        """
        if self._updating_viewer:
            return

        if event.type not in ['affine', 'scale', 'translate']:
            return

        l = event.source
        if l.name not in self.msims:
            return

        # Post-registration + showing Registered: live update current tp
        if (self.visualization_type_rbuttons.enabled and
                self.visualization_type_rbuttons.value == CHOICE_REGISTERED):
            self._update_registered_param_for_current_tp(l)


    def link_channel_layers(self, layers, attributes=('contrast_limits', 'visible')):
        """
        Link the following attributes of channel layers:
          - contrast_limits
          - visible
        """

        # link channel layers
        from napari.experimental import link_layers

        sims = {l.name: msi_utils.get_sim_from_msim(self.msims[l.name])
                for l in layers}

        channels = [_utils.get_str_unique_to_ch_from_sim_coords(sim.coords) for sim in sims.values()]
        for ch in channels:
            ch_layers = list(_utils.filter_layers(layers, sims, ch=ch))

            if len(ch_layers):
                link_layers(ch_layers, attributes)


    def link_view_layers(self, layers, attributes=('affine', 'scale', 'translate', 'rotate')):
        """
        Link the following attributes of layers that share the same view:
          - affine
          - scale
          - translate
          - rotate
        """

        # link tile layers
        from napari.experimental import link_layers

        sims = {l.name: msi_utils.get_sim_from_msim(self.msims[l.name])
                for l in layers}

        views = [_utils.get_str_unique_to_view_from_layer_name(l.name) for l in layers]
        for view in views:
            view_layers = list(_utils.filter_layers(layers, sims, view=view))

            if len(view_layers):
                link_layers(view_layers, attributes)


    def __del__(self):

        print('Deleting napari-stitcher widget')

        # clean up callbacks
        self.viewer.dims.events.current_step.disconnect(self.update_viewer_transformations)

        for l in self.viewer.layers:
            if l.name in self.layers_selection.choices:
                l.events.disconnect(self.watch_layer_changes)


if __name__ == "__main__":

    import napari
    from multiview_stitcher.sample_data import get_mosaic_sample_data_path

    filename = get_mosaic_sample_data_path()

    viewer = napari.Viewer()
    
    wdg = StitcherQWidget(viewer)
    viewer.window.add_dock_widget(wdg)

