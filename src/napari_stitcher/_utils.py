import os

import numpy as np
from mvregfus import mv_utils, io_utils
from mvregfus.image_array import ImageArray

from dask import delayed, compute
import dask.array as da
from tqdm.dask import TqdmCallback

from napari.utils import progress


def load_tiles(view_dict: dict,
               channels: int,
               times: list,
               max_project: bool = True,
               ) -> dict:
    """
    Return: dict of delayed dask arrays
    """

    # load views
    view_ims = {ch: {t: {vdv['view']: 
                            da.from_delayed(
                                delayed(io_utils.read_tile_from_multitile_czi)
                                   (vdv['filename'],
                                    vdv['view'],
                                    ch,
                                    time_index=t,
                                    max_project=max_project,
                                    origin=vdv['origin'],
                                    spacing=vdv['spacing'],
                                    ),
                                shape=tuple(vdv['shape']),
                                dtype=np.uint16,
                            )
                        for vdv in view_dict.values()}
                    for t in times}
                for ch in channels}

    return view_ims


def load_tiles_from_layers(
                            layers: list,
                            view_dict: dict,
                            channels: int,
                            times: list,
                            source_identifier: tuple = None,
                            ) -> dict:
    
    """
    Return: dict of delayed dask arrays
      - directly from layer data
      - format: nested dict of channels, times, views (outer -> inner)
    """
    
    view_ims = {ch: {t: {vdv['view']: 
                            # delayed(lambda x, origin, spacing:
                            #         image_array.ImageArray(x, origin=origin, spacing=spacing))(
                                get_layer_from_source_identifier_view_and_ch(
                                    layers=layers,
                                    source_identifier=source_identifier,
                                    view=vdv['view'],
                                    ch=ch
                                ).data[t]
                                # origin=vdv['origin'],
                                # spacing=vdv['spacing'])
                        for vdv in view_dict.values()}
                    for t in times}
                for ch in channels}

    return view_ims


class TemporarilyDisabledWidgets(object):
    """
    Conext manager to temporarily disable widgets during long computation
    """
    def __init__(self, widgets):
        self.widgets = widgets
        self.enabled_states = {w: True if w.enabled else False for w in widgets}
    def __enter__(self):
        for w in self.widgets:
            w.enabled = False
    def __exit__(self, type, value, traceback):
        for w in self.widgets:
            w.enabled = self.enabled_states[w]


class VisibleActivityDock(object):
    """
    Conext manager to temporarily disable widgets during long computation
    """
    def __init__(self, viewer):
        self.viewer = viewer
    def __enter__(self):
        self.viewer.window._status_bar._toggle_activity_dock(True)
    def __exit__(self, type, value, traceback):
        self.viewer.window._status_bar._toggle_activity_dock(False)


def compute_dask_object(dask_object,
                        viewer,
                        widgets_to_disable=None,
                        message="Registering tiles",
                        scheduler='threading',
                        ):
    """
    Compute dask object. While doing so:
     - show progress bar
     - disable widgets temporarily
    """
    with TemporarilyDisabledWidgets(widgets_to_disable),\
         VisibleActivityDock(viewer),\
         TqdmCallback(tqdm_class=progress, desc=message, bar_format=" "):
        result = compute(dask_object, scheduler=scheduler)[0]

    return result


def add_metadata_to_tiles(viewims, view_dict):

    channels = list(viewims.keys())
    times = list(viewims[channels[0]].keys())

    viewims =   {ch:
                    {t: {vdv['view']:
                            delayed(lambda x, origin, spacing:
                                    ImageArray(x, origin=origin, spacing=spacing))(
                        
                                        viewims[ch][t][vdv['view']],
                                        vdv['origin'],
                                        vdv['spacing'])
                        for vdv in view_dict.values()}
                    for t in times}
                for ch in channels}

    return viewims


# get source file path from open layers
def get_source_path_from_viewer(viewer):
    for l in viewer.layers:
        if l.source.path is not None and l.source.path.endswith('.czi'):
            return l.source.path
    return None


def source_identifier_to_str(source_identifier):
    return f"File: {os.path.basename(source_identifier['filename'])} (Sample: {source_identifier['sample_index']})"


def str_to_source_identifier(string):
    # use regex to extract filename and sample index
    # regex to match filename from e.g. 'File: /home/.../sample_1.czi (Sample: 1)'
    filename = re.search(r'File: (.*) \(Sample: \d+\)', string).group(1)
    sample_index = int(re.search(r'File: .*\ \(Sample: (\d+)\)', string).group(1))

    return {'filename': filename, 'sample_index': sample_index}


def layer_was_loaded_by_own_reader(layer):
    if 'napari_stitcher_reader_function' in layer.metadata and\
        layer.metadata['napari_stitcher_reader_function'] == 'read_mosaic_czi':
        return True
    else:
        False


def layer_coincides_with_source_identifier(layer, source_identifier):
    if layer.source.path == source_identifier['filename'] and\
        layer.metadata['sample_index'] == source_identifier['sample_index']:
        return True
    else:
        return False


def get_list_of_source_identifiers_from_layers(layers):

    source_identifiers = []
    for l in layers:
        if layer_was_loaded_by_own_reader(l):
            source_identifier = {'filename': l.source.path,
                                 'sample_index': l.metadata['sample_index']}
            source_identifiers.append(source_identifier)

    return source_identifiers


def get_layer_name_from_view_and_ch(view=0, ch=0):
    return 'tile_%03d' %view + '_ch_%03d' %ch


def get_view_from_layer(layer):
    return layer.metadata['view']


import re
def get_ch_from_layer(layer):

    # regex to match ch from e.g. 'view_008_ch_002_ [0]'
    return int(re.search(r'_ch_(\d+)', layer.name).group(1))


def get_layers_from_source_identifier_and_view(layers, source_identifier, view):
    for l in layers:
        if layer_was_loaded_by_own_reader(l) and\
            layer_coincides_with_source_identifier(l, source_identifier) and\
                get_view_from_layer(l) == view:
            yield l


def get_layer_from_source_identifier_view_and_ch(layers, source_identifier, view, ch):

    view_layers = get_layers_from_source_identifier_and_view(layers, source_identifier, view)
    for l in view_layers:
        if get_ch_from_layer(l) == ch:
            return l


def params_to_napari_affine(params, stack_props, view_stack_props):

    """
    y = Ax+c
    y=sy*yp+oy
    x=sx*xp+ox
    sy*yp+oy = A(sx*xp+ox)+c
    yp = syi * A*sx*xp + syi  *A*ox +syi*(c-oy)
    A' = syi * A * sx
    c' = syi  *A*ox +syi*(c-oy)
    """

    p = mv_utils.params_to_matrix(params)

    ndim = len(stack_props['spacing'])

    sx = np.diag(list((stack_props['spacing'])))
    sy = np.diag(list((view_stack_props['spacing'])))
    syi = np.linalg.inv(sy)
    p[:ndim, ndim] = np.dot(syi, np.dot(p[:ndim, :ndim], stack_props['origin'])) \
                + np.dot(syi, (p[:ndim, ndim] - view_stack_props['origin']))
    p[:ndim, :ndim] = np.dot(syi, np.dot(p[:ndim, :ndim], sx))
    p = np.linalg.inv(p)

    return p
