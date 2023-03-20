"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/guides.html?#readers
"""
import numpy as np

from mvregfus.image_array import ImageArray
from mvregfus import io_utils, mv_utils

import dask.array as da
from dask import delayed

from aicspylibczi import CziFile

from napari_stitcher import _utils, _file_utils

import time


def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        path = path[0]

    # if we know we cannot read the file, we immediately return None.
    # otherwise we return the *function* that can read ``path``.
    if path.endswith(".czi"):
        return read_mosaic_czi
    else:
        return None


def read_mosaic_czi(path, sample_index=None):
    """
    
    Read in tiles as layers.
    
    Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer. Both "meta", and "layer_type" are optional. napari will
        default to layer_type=="image" if not provided
    """
    # handle both a string and a list of strings
    paths = [path] if isinstance(path, str) else path

    max_project = False
    # dims = io_utils.get_dims_from_multitile_czi(paths[0])
    czi = CziFile(paths[0])

    if czi.pixel_type == 'gray8':
        input_dtype = 'uint8'
    else:
        input_dtype = 'uint16'

    # determine number of samples
    dims = czi.get_dims_shape()
    n_samples = 0
    for dim in dims:
        n_samples = dim['S'][1]

    # ask for sample_index when several are available
    if sample_index is None:
        if n_samples == 1:
            sample_index = 0
        else:
            from magicgui.widgets import request_values
            sample_index = request_values(
                sample_index=dict(annotation=int,
                            label="Which sample should be loaded?",
                            options={'min': 0, 'max': n_samples - 1}),
                )['sample_index']
        
    dims = _file_utils.get_dims_from_multitile_czi(paths[0], sample_index=sample_index)

    view_dict = io_utils.build_view_dict_from_multitile_czi(paths[0], max_project=max_project, S=sample_index)
    views = np.array([view for view in sorted(view_dict.keys())])

    if max_project or int(dims['Z'][1] <= 1):
        ndim = 2
    else:
        ndim = 3

    channels = range(dims['C'][0], dims['C'][1])
    times = range(dims['T'][0], dims['T'][1])

    view_das = []
    for vdv in view_dict.values():
        view_das.append(
                da.stack([
                    da.stack([
                        da.from_delayed(delayed(
                            # io_utils.read_tile_from_multitile_czi
                            get_tile_from_multitile_czi
                            )
                                (vdv['filename'],
                                    vdv['view'],
                                    ch,
                                    time_index=t,
                                    sample_index=sample_index,
                                    max_project=max_project,
                                    origin=vdv['origin'],
                                    spacing=vdv['spacing'],
                                    ),
                        shape=tuple(vdv['shape']),
                        dtype=np.dtype(input_dtype),
                        )
                    for ch in channels])
                for t in times])
        )

    # if ndim == 2:
    #     view_das = [view_da[:, :, None] for view_da in view_das]

    # set target stack properties to those of first view
    stack_props = view_dict[0]
    view_stack_props = view_dict

    # assume identity transf parameters
    transf_params = [mv_utils.matrix_to_params(np.eye(ndim+1)) for i in range(len(views))]
    ps = [_utils.params_to_napari_affine(p, stack_props, view_stack_props[iview])
            for iview, p in enumerate(transf_params)]  

    layer_type = "image"
    file_id = time.time()

    return [(view_das[iview], # (T, C, (Z,) Y, X)
            {
             'contrast_limits': [[0,255]] * len(channels),
             'name': [_utils.get_layer_name_from_view_and_ch(iview, ch)
                        for ch in channels],
             'colormap': 'gray_r',
             'colormap': ['red', 'green'][iview%2],
             'gamma': 0.6,
             'channel_axis': 1,
             'affine': ps[iview],
             'cache': False,
             'metadata': {
                          'napari_stitcher_reader_function': 'read_mosaic_czi',
                          'load_id': file_id,
                          'stack_props': stack_props,
                          'view_dict': view_dict[iview],
                          'view': iview,
                          'ndim': ndim,
                        #   'source_file': path,
                        #   'parameter_type': 'metadata',
                          'sample_index': sample_index,
                          'times': times,
                          'dtype': input_dtype,
                        #   'axis_labels': 'TCZYX',
                          },
             'blending': 'additive',
             },
            layer_type)
                for iview, view in enumerate(views)][:]


def get_tile_from_multitile_czi(filename,
                                 tile_index, channel_index=0, time_index=0, sample_index=0,
                                 origin=None, spacing=None,
                                 max_project=True,
                                 ):
    """
    Use czifile to read images (as there's a bug in aicspylibczi20221013, namely that
    neighboring tiles are included (prestitching?) in a given read out tile).
    """
    czifileFile = CziFile(filename)

    tile = czifileFile.read_image(M=tile_index,
                                  S=sample_index,
                                  T=time_index,
                                  C=channel_index)[0].squeeze()

    if max_project and tile.ndim == 3:
        tile = tile.max(axis=0)

    if origin is None:
        origin = [0.] * tile.ndim

    if spacing is None:
        spacing = [1.] * tile.ndim

    tile = ImageArray(tile, origin=origin, spacing=spacing)

    return tile


if __name__ == "__main__":
    # tmp = czi_reader_function("/Users/malbert/software/napari-stitcher/image-datasets/arthur_20220621_premovie_dish2-max.czi")
    # fn = "/Users/malbert/software/napari-stitcher/image-datasets/yu_220829_WT_quail_st4+_x40_zoom0_5_5x5_488ZO1-568Sox2-647Tbra-max.czi"
    fn = "/Users/malbert/software/napari-stitcher/image-datasets/04_stretch-01_AcquisitionBlock2_pt2.czi"
    # fn = "/Users/malbert/software/napari-stitcher/image-datasets/arthur_20220621_premovie_dish2-max.czi"
    tmp = read_mosaic_czi(fn)

    io_utils.build_view_dict_from_multitile_czi(fn, max_project=False, S=0)

    ar = tmp[0][0].compute()