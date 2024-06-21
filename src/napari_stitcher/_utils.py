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


def get_str_unique_to_ch_from_layer_name(layer_name):
    return layer_name.split(' :: ')[1]


def get_str_unique_to_ch_from_sim_coords(layer_coords):
    return str(layer_coords['c'].values)


def get_view_from_layer(layer):
    return layer.metadata['view']


def filter_layers(layers, sims, view=None, ch=None):
    for l in layers:
        if view is not None and get_str_unique_to_view_from_layer_name(l.name) != view: continue
        if ch is not None and get_str_unique_to_ch_from_sim_coords(sims[l.name].coords) != ch: continue
        yield l


def get_tile_indices(mosaic_arr='rows first', n_col=1, n_row=1, n_tiles=1):
    """
    Return list of tiles' indices following a mosaic arrangement
    mosaic_arr='rows first','columns first','snake by rows','snake by columns'
    n_col: number of columns
    n_row: number of rows
    n_tiles: number of tiles (can be lower than n_col x n_row)
    Returns a list of tuples (row index, column index)
    """

    # generate all possible indices 
    ind_list = []  # indices list
    if mosaic_arr == 'rows first': 
        for i in range(n_row):
            for j in range(n_col):
                ind_list.append((i,j))
    
    elif mosaic_arr == 'columns first': 
        for j in range(n_col):
            for i in range(n_row):
                ind_list.append((i,j))
                
    elif mosaic_arr == 'snake by rows': 
        for i in range(n_row):
            if i%2 == 0:  # even row: normal order
                for j in range(n_col):
                    ind_list.append((i,j))
            else:  # odd row: reversed order
                for j in range(n_col-1, -1, -1):
                    ind_list.append((i,j))
                    
    elif mosaic_arr == 'snake by columns': 
        for j in range(n_col):
            if j%2 == 0:  # even col: normal order
                for i in range(n_row):
                    ind_list.append((i,j))
            else:  # odd col: reversed order
                for i in range(n_row-1, -1, -1):
                    ind_list.append((i,j))

    # select only the indices for the existing tiles
    return ind_list[:n_tiles]