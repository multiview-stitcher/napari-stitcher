"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/guides.html?#readers
"""
import numpy as np
import xarray as xr

from mvregfus.image_array import ImageArray
from mvregfus import io_utils, mv_utils

import dask.array as da
from dask import delayed

from aicspylibczi import CziFile
from aicsimageio import AICSImage

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
    

def read_mosaic_czi_into_list_of_spatial_xarrays(path, scene_index=None):
    """
    Read CZI mosaic dataset into xarray containing all information needed for stitching.
    Could eventually be based on https://github.com/spatial-image/spatial-image.
    Use list instead of dict to make sure xarray metadata (coordinates + perhaps attrs)
    are self explanatory for downstream code (and don't depend e.g. on view/tile numbering).
    """

    aicsim = AICSImage(path, reconstruct_mosaic=False)

    if len(aicsim.scenes) > 1:
        from magicgui.widgets import request_values
        scene_index = request_values(
            scene_index=dict(annotation=int,
                        label="Which scene should be loaded?",
                        options={'min': 0, 'max': len(aicsim.scenes) - 1}),
            )['scene_index']
        aicsim.set_scene(scene_index)
    else:
        scene_index = 0

    xim =  aicsim.get_xarray_dask_stack()
    xim = xim.sel(I=scene_index)

    for axis in ['Z', 'T']:
        if axis in xim.dims and len(xim.coords[axis]) < 2:
            xim = xim.sel({axis: 0}, drop=True)

    views = range(len(xim.coords['M']))
    
    pixel_sizes = aicsim.physical_pixel_sizes._asdict()

    spatial_axes = [axis for axis in ['Z','Y','X'] if axis in xim.dims]

    view_xims = []
    for view in views:

        view_xim = xim.sel(M=view)

        tile_mosaic_position = aicsim.get_mosaic_tile_position(view)
        origin = {mosaic_axis: tile_mosaic_position[ima]
                  for ima, mosaic_axis in enumerate(['Y', 'X'])}
        if 'Z' in spatial_axes:
            origin['Z'] = 0
        
        spacing = {axis: pixel_sizes[axis] for axis in spatial_axes}

        for axis in spatial_axes:
            view_xim.assign_coords({axis: view_xim.coords[axis] + origin[axis]})
            view_xim.assign_coords({axis: view_xim.coords[axis] * spacing[axis]})

        view_xim.attrs.update(dict(
            spacing = spacing,
            origin = origin,
            scene_index=scene_index,
            spatial_axes=spatial_axes,
        ))

        view_xims.append(view_xim)

    return view_xims


def read_mosaic_czi(path, scene_index=None):
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

    view_xims = read_mosaic_czi_into_list_of_spatial_xarrays(paths[0], scene_index=scene_index)
        
    layer_type = "image"

    out_layers = []
    for iview, view_xim in enumerate(view_xims):

        for ch_coord in view_xim.coords['C']:

            view_ch_xim = view_xim.sel(C=ch_coord)

            out_layers.append(
                (view_ch_xim,
                {
                'contrast_limits': [np.iinfo(view_ch_xim.dtype).min,
                                    np.iinfo(view_ch_xim.dtype).max],
                'name': _utils.get_layer_name_from_view_and_ch_name(iview,
                                                                    str(ch_coord.data)),
                'colormap': 'gray_r',
                'colormap': ['red', 'green'][iview%2],
                'gamma': 0.6,
                'scale': [view_ch_xim.attrs['spacing'][ax] for ax in view_ch_xim.attrs['spatial_axes']],
                'translate': [view_ch_xim.attrs['origin'][ax] * view_ch_xim.attrs['spacing'][ax]
                            for ax in view_ch_xim.attrs['spatial_axes']],
                'cache': True,
                'metadata': {
                            'napari_stitcher_reader_function': 'read_mosaic_czi',
                            'scene_index': scene_index,
                            'channel_name': str(ch_coord.data),
                            },
                'blending': 'additive',
                },
                layer_type)
            )

    return out_layers

    # for view in views:

    #     aff = xr.DataArray(
    #         data=np.diag([pixel_sizes[axis] for axis in spatial_axes] + [1]),
    #         dims=['x_out', 'x_in'],
    #         coords=dict(
    #             x_in=affine_coords,
    #             x_out=affine_coords,
    #         )
    #     )

    #     tile_mosaic_position = aicsim.get_mosaic_tile_position(view)
    #     for ima, mosaic_axis in enumerate(['Y', 'X']):

    #         aff[
    #             dict(x_in=np.where(aff.coords['x_in'].data=='offset')[0][0],
    #                  x_out=np.where(aff.coords['x_out'].data==mosaic_axis)[0][0])
    #                  ] = tile_mosaic_position[ima] * pixel_sizes[mosaic_axis]

    #     affine_transforms[view] = aff

    # world_transforms = dict()
    # from napari.utils.transforms import Affine
    # world_transforms = {view: Affine(affine_matrix=affine_transforms[view].data)
    #                     for view in views}


    # layer_type = "image"
    # file_id = time.time()

    # out_layers = []
    # for view in views:
    #     for ch_coord in xim.coords['C']:

    #         layer_xim = xim.sel(M=view, C=ch_coord)
    #         layer_xim.attrs.update(dict(
    #             spacing={pixel_sizes[axis] for axis in spatial_axes},
    #             origin={},
    #         )
    #         )

    #         # import pdb; pdb.set_trace()

    #         out_layers.append(
    #             (layer_xim,
    #         {
    #          'contrast_limits': [np.iinfo(xim.dtype).min,
    #                              np.iinfo(xim.dtype).max],
    #          'name': _utils.get_layer_name_from_view_and_ch_name(view,
    #                                                              str(ch_coord.data)),
    #          'colormap': 'gray_r',
    #          'colormap': ['red', 'green'][view%2],
    #          'gamma': 0.6,
    #          'affine': world_transforms[view].expand_dims(
    #             layer_xim.get_axis_num([axis for axis in layer_xim.dims
    #                               if axis not in spatial_axes])),
    #          'cache': False,
    #          'metadata': {
    #                       'napari_stitcher_reader_function': 'read_mosaic_czi',
    #                       'load_id': file_id,
    #                     #   'stack_props': stack_props,
    #                     #   'view_dict': view_dict[view],
    #                     #   'view': view,
    #                     #   'ndim': ndim,
    #                     #   'source_file': path,
    #                     #   'parameter_type': 'metadata',
    #                       'scene_index': scene_index,
    #                     #   'times': times,
    #                     #   'dtype': input_dtype,
    #                     #   'axis_labels': 'TCZYX',
    #                       'channel_name': str(ch_coord.data),
    #                       'data2world': affine_transforms[view],
    #                       },
    #          'blending': 'additive',
    #          },
    #         layer_type)
    #         )

    # return out_layers


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

    import napari
    # from napari_stitcher import StitcherQWidget

    # filename = "/Users/malbert/software/napari-stitcher/image-datasets/04_stretch-01_AcquisitionBlock2_pt2.czi"
    # filename = "/Users/malbert/software/napari-stitcher/image-datasets/yu_220829_WT_quail_st6_x10_zoom0.7_1x3_488ZO1-568Sox2-647Tbra.czi"
    filename = "/Users/malbert/software/napari-stitcher/image-datasets/arthur_20220621_premovie_dish2-max.czi"
    # filename = "/Users/malbert/software/napari-stitcher/image-datasets/MAX_LSM900.czi"

    viewer = napari.Viewer()
    
    # viewer.open("/Users/malbert/software/napari-stitcher/image-datasets/arthur_20220609_WT_emb2_5X_part1_max.czi")

    # wdg = StitcherQWidget(viewer)
    # viewer.window.add_dock_widget(wdg)

    viewer.open(filename)

    napari.run()