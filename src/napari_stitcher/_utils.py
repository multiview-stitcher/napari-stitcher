import warnings

import numpy as np
import xarray as xr

from dask import delayed, compute
import dask.array as da
from tqdm.dask import TqdmCallback

from multiview_stitcher import ngff_utils, msi_utils

from napari.utils import progress


NAPARI_OME_ZARR_PLUGIN = 'napari-ome-zarr'


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
    # 'c' may be a scalar (0-d) or a length-1 dimensional coordinate;
    # .flat[0] extracts the single string value in both cases
    return str(layer_coords['c'].values.flat[0])


def is_napari_ome_zarr_layer(layer):
    source = getattr(layer, 'source', None)
    reader_plugin = getattr(source, 'reader_plugin', None)

    return (
        isinstance(reader_plugin, str)
        and (
            reader_plugin == NAPARI_OME_ZARR_PLUGIN
            or reader_plugin.startswith(f'{NAPARI_OME_ZARR_PLUGIN}.')
        )
    )


def get_layer_source_path(layer):
    source = getattr(layer, 'source', None)
    return getattr(source, 'path', None)


def is_multiscale_dask_layer(layer):
    if not getattr(layer, 'multiscale', False):
        return False

    try:
        return all(isinstance(ldata, da.Array) for ldata in layer.data)
    except TypeError:
        return False


def get_channel_name_from_layer_name(layer):
    # Extract channel name from layer name (format: "tile :: channel").
    # Layers without this format (e.g. plain user images) share 'default_channel'
    # so they are treated as a single-channel dataset and can be registered together.
    if ' :: ' in layer.name:
        return get_str_unique_to_ch_from_layer_name(layer.name)

    return 'default_channel'


def get_napari_ome_zarr_channel_name(layer):
    if not is_napari_ome_zarr_layer(layer):
        return None

    # napari-ome-zarr uses names from OMERO metadata, sometimes prefixed as
    # "image: channel". The underlying msim uses the channel label itself.
    return layer.name.split(': ')[-1] if ': ' in layer.name else layer.name


def get_layer_c_coord(layer, ldata=None):
    if isinstance(ldata, xr.DataArray) and 'c' in ldata.coords:
        return get_str_unique_to_ch_from_sim_coords(ldata.coords)

    ch_name = get_channel_name_from_layer_name(layer)
    if ch_name != 'default_channel':
        return ch_name

    napari_ome_zarr_ch_name = get_napari_ome_zarr_channel_name(layer)
    if napari_ome_zarr_ch_name is not None:
        return napari_ome_zarr_ch_name

    return ch_name


def get_ome_zarr_image_dims(zarr_path):
    import ngff_zarr

    return tuple(ngff_zarr.from_ngff_zarr(zarr_path).images[0].dims)


def _as_scalar_coord(value):
    value_array = np.asarray(value)
    if value_array.size == 1:
        return value_array.reshape(()).item()

    return value


def _coord_matches(coord, candidate):
    if candidate is None:
        return False

    coord = _as_scalar_coord(coord)
    candidate = _as_scalar_coord(candidate)

    return coord == candidate or str(coord) == str(candidate)


def _get_msim_c_coords(msim):
    sim = msi_utils.get_sim_from_msim(msim)
    if 'c' not in sim.coords:
        return []

    c_values = np.asarray(sim.coords['c'].values).reshape(-1)
    if c_values.size == 1:
        return [get_str_unique_to_ch_from_sim_coords(sim.coords)]

    return [_as_scalar_coord(coord) for coord in c_values]


def _candidate_channel_names(layer, c_coord):
    candidates = [c_coord, getattr(layer, 'name', None)]

    name = getattr(layer, 'name', '')
    if ' :: ' in name:
        candidates.append(get_str_unique_to_ch_from_layer_name(name))
    if ': ' in name:
        candidates.append(name.split(': ')[-1])
    elif ':' in name:
        candidates.append(name.split(':')[-1].strip())

    return candidates


def _channel_index_from_viewer_order(layer, viewer, n_channels):
    source_path = get_layer_source_path(layer)
    source = getattr(layer, 'source', None)
    reader_plugin = getattr(source, 'reader_plugin', None)

    if viewer is None or source_path is None:
        return None

    try:
        layer_shape = tuple(layer.data[0].shape)
    except (TypeError, IndexError):
        layer_shape = None

    matching_layers = []
    for other in viewer.layers:
        other_source = getattr(other, 'source', None)
        if (
            getattr(other_source, 'path', None) != source_path
            or getattr(other_source, 'reader_plugin', None) != reader_plugin
            or not is_multiscale_dask_layer(other)
        ):
            continue

        try:
            other_shape = tuple(other.data[0].shape)
        except (TypeError, IndexError):
            other_shape = None

        if layer_shape is None or other_shape == layer_shape:
            matching_layers.append(other)

    if len(matching_layers) != n_channels:
        return None

    for ilayer, candidate_layer in enumerate(matching_layers):
        if candidate_layer is layer:
            return ilayer

    return None


def _get_ome_zarr_channel_coord(layer, viewer, msim, c_coord):
    c_coords = _get_msim_c_coords(msim)
    if not c_coords:
        return None

    for candidate in _candidate_channel_names(layer, c_coord):
        for coord in c_coords:
            if _coord_matches(coord, candidate):
                return coord

    channel_index = _channel_index_from_viewer_order(
        layer, viewer, n_channels=len(c_coords))
    if channel_index is not None:
        return c_coords[channel_index]

    if len(c_coords) == 1:
        return c_coords[0]

    return None


def set_msim_c_coord(msim, c_coord):
    if 'c' not in msi_utils.get_dims(msim):
        msim = msi_utils.ensure_dim(msim, 'c')

    for scale_key in msi_utils.get_sorted_scale_keys(msim):
        msim[scale_key]['image'].coords['c'] = [c_coord]

    return msim


def set_msim_spatial_coords_from_layer(msim, layer, sdims):
    if not sdims:
        return msim

    layer_scale = np.asarray(layer.scale[-len(sdims):], dtype=float)
    layer_translation = np.asarray(layer.translate[-len(sdims):], dtype=float)

    scale_keys = msi_utils.get_sorted_scale_keys(msim)
    finest_sim = msim[scale_keys[0]]['image']
    finest_spatial_shape = np.array(
        [finest_sim.sizes[dim] for dim in sdims], dtype=float)

    sims = []
    for scale_key in scale_keys:

        sim = msi_utils.get_sim_from_msim(msim, scale=scale_key)
        # sim = msim[scale_key]['image']
        level_spatial_shape = np.array(
            [sim.sizes[dim] for dim in sdims], dtype=float)
        level_scale_factors = finest_spatial_shape / level_spatial_shape

        coords = {}
        for idim, dim in enumerate(sdims):
            spacing = layer_scale[idim] * level_scale_factors[idim]
            origin = layer_translation[idim]
            coords[dim] = origin + spacing * np.arange(
                sim.sizes[dim], dtype=float)

        sim = sim.assign_coords(coords)
        sims.append(sim)

    msim = msi_utils.get_msim_from_sims(sims)

    return msim


def try_load_msim_from_napari_ome_zarr(layer, viewer, dims, sdims, c_coord):
    if (
        not is_napari_ome_zarr_layer(layer)
        or not is_multiscale_dask_layer(layer)
    ):
        return None

    zarr_path = get_layer_source_path(layer)
    if zarr_path is None:
        return None

    try:
        ome_zarr_dims = get_ome_zarr_image_dims(zarr_path)
        msim = ngff_utils.read_msim_from_ome_zarr(zarr_path)

        if 'c' in ome_zarr_dims and len(ome_zarr_dims) != len(dims):
            ome_zarr_c_coord = _get_ome_zarr_channel_coord(
                layer, viewer, msim, c_coord)
            if ome_zarr_c_coord is None:
                raise ValueError(
                    f'Could not determine channel coordinate for layer '
                    f'{layer.name!r}.')

            msim = msi_utils.multiscale_sel_coords(
                msim, {'c': [ome_zarr_c_coord]})

        msim_sdims = [
            dim for dim in msi_utils.get_spatial_dims(msim) if dim in sdims]
        if len(msim_sdims) != len(msi_utils.get_spatial_dims(msim)):
            msim_sdims = msi_utils.get_spatial_dims(msim)

        msim = set_msim_c_coord(msim, c_coord)
        msim = set_msim_spatial_coords_from_layer(msim, layer, msim_sdims)

        return msim

    except Exception as e:
        warnings.warn(
            'Could not load napari-ome-zarr layer from the underlying '
            f'OME-Zarr directly. Falling back to napari layer data. ({e})',
            RuntimeWarning,
        )
        return None


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
