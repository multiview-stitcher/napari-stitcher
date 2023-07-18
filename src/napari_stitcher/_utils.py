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


def get_str_unique_to_view_from_layer_name(layer_name):
    return layer_name.split(' :: ')[0]


def get_str_unique_to_ch_from_xim_coords(layer_coords):
    return str(layer_coords['C'].values)


def get_view_from_layer(layer):
    return layer.metadata['view']


def filter_layers(layers, xims, view=None, ch=None):
    for l in layers:
        if view is not None and get_str_unique_to_view_from_layer_name(l.name) != view: continue
        # if ch is not None and get_str_unique_to_ch_from_layer_name(l.name) != ch: continue
        if ch is not None and get_str_unique_to_ch_from_xim_coords(xims[l.name].coords) != ch: continue
        yield l


def duplicate_channel_xims(xims):

    xims_ch_duplicated = [
        xr.concat([xim] * 2, dim='C')\
        .assign_coords(C=[
            xim.coords['C'].data[0],
            xim.coords['C'].data[0] + '_2']
        ) for xim in xims]
    
    return xims_ch_duplicated


def shift_to_matrix(shift):
    ndim = len(shift)
    M = np.concatenate([shift, [1]], axis=0)
    M = np.concatenate([np.eye(ndim + 1)[:,:ndim], M[:,None]], axis=1)
    return M
