import numpy as np
import xarray as xr

from dask import delayed, compute
import dask.array as da
from tqdm.dask import TqdmCallback

from napari.utils import progress


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


def get_str_unique_to_view_from_layer_name(layer_name):
    return layer_name.split(' :: ')[0]


def get_str_unique_to_ch_from_xim_coords(layer_coords):
    return str(layer_coords['c'].values)


def get_view_from_layer(layer):
    return layer.metadata['view']


def filter_layers(layers, xims, view=None, ch=None):
    for l in layers:
        if view is not None and get_str_unique_to_view_from_layer_name(l.name) != view: continue
        if ch is not None and get_str_unique_to_ch_from_xim_coords(xims[l.name].coords) != ch: continue
        yield l
