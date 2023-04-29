"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/guides.html?#readers
"""
import numpy as np
import xarray as xr
import networkx as nx

from mvregfus.image_array import ImageArray
from mvregfus import io_utils, mv_utils

import dask.array as da
from dask import delayed

from aicspylibczi import CziFile
from aicsimageio import AICSImage

from napari_stitcher import _mv_graph, _spatial_image_utils


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

    if len(aicsim.scenes) > 1 and scene_index is None:
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

    spatial_dims = [axis for axis in ['Z','Y','X'] if axis in xim.dims]

    view_xims = []
    for view in views:

        view_xim = xim.sel(M=view)

        tile_mosaic_position = aicsim.get_mosaic_tile_position(view)
        origin_values = {mosaic_axis: tile_mosaic_position[ima] * pixel_sizes[mosaic_axis]
                  for ima, mosaic_axis in enumerate(['Y', 'X'])}
        
        if 'Z' in spatial_dims:
            origin_values['Z'] = 0

        origin = xr.DataArray([origin_values[dim] for dim in spatial_dims],
                              dims=['dim'],
                              coords={'dim': spatial_dims})
        
        # spacing = xr.DataArray([pixel_sizes[dim] for dim in spatial_dims],
        #                       dims=['dim'],
        #                       coords={'dim': spatial_dims})
        
        for dim in spatial_dims:
            view_xim = view_xim.assign_coords({dim: view_xim.coords[dim] + origin.loc[dim]})

        view_xim.attrs.update(dict(
            # spacing = spacing,
            # origin = origin,
            scene_index=scene_index,
            spatial_dims=spatial_dims,
            source=path,
        ))

        view_xim.name = str(view)

        view_xims.append(view_xim)

    return view_xims


def create_image_layer_tuple_from_spatial_xim(xim,
                                              colormap='gray_r',
                                              name_prefix=None,
                                              ):

    """
    Note:
    - xarray.DataArray can have coordinates for dimensions that are not listed in xim.dims (?)
    - useful for channel names
    """

    ch_name = str(xim.coords['C'].data)

    if name_prefix is None:
        name = ch_name
    else:
        name = ' :: '.join([name_prefix, ch_name])

    metadata = \
        {
        'napari_stitcher_reader_function': 'read_mosaic_czi',
        # 'channel_name': ch_name,
        }
    
    # metadata['xr_attrs'] = xim.attrs.copy()

    spatial_dims = _spatial_image_utils.get_spatial_dims_from_xim(xim)
    origin = _spatial_image_utils.get_origin_from_xim(xim)
    spacing = _spatial_image_utils.get_spacing_from_xim(xim)

    kwargs = \
        {
        'contrast_limits': [np.iinfo(xim.dtype).min,
                            np.iinfo(xim.dtype).max],
        # 'contrast_limits': [np.iinfo(xim.dtype).min,
        #                     30],
        'name': name,
        'colormap': colormap,
        'gamma': 0.6,
        # 'scale': [
        #     # xim.attrs['spacing'].loc[dim]
        #     xim.coords[dim][1] - xim.coords[dim][0]
        #           for dim in xim.attrs['spatial_dims']],
        # 'translate': [
        #     # xim.attrs['origin'].loc[dim]
        #     xim.coords[dim][0]
        #             for dim in xim.attrs['spatial_dims']],
        # 'affine': _spatial_image_utils.get_data_to_world_matrix_from_spatial_image(xim),
        'translate': [origin[dim] for dim in spatial_dims],
        'scale': [spacing[dim] for dim in spatial_dims],
        'cache': True,
        'blending': 'additive',
        'metadata': metadata,
        }

    return (xim, kwargs, 'image')


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

    # get colors from graph analysis
    mv_graph = _mv_graph.build_view_adjacency_graph_from_xims(view_xims, expand=True)
    colors = nx.coloring.greedy_color(mv_graph)
    
    # import pdb; pdb.set_trace()

    cmaps = ['red', 'green', 'blue', 'gray']
    cmaps = {iview: cmaps[color_index % len(cmaps)]
             for iview, color_index in colors.items()}
        
    out_layers = [
        create_image_layer_tuple_from_spatial_xim(
                    view_xim.sel(C=ch_coord),
                    cmaps[iview],
                    name_prefix='tile_%03d' %iview)
            for iview, view_xim in enumerate(view_xims)
        for ch_coord in view_xim.coords['C']
        ]

    return out_layers


if __name__ == "__main__":

    import napari
    # from napari_stitcher import StitcherQWidget

    # filename = "/Users/malbert/software/napari-stitcher/image-datasets/04_stretch-01_AcquisitionBlock2_pt2.czi"
    # filename = "/Users/malbert/software/napari-stitcher/image-datasets/yu_220829_WT_quail_st6_x10_zoom0.7_1x3_488ZO1-568Sox2-647Tbra.czi"
    # filename = "/Users/malbert/software/napari-stitcher/image-datasets/arthur_20220621_premovie_dish2-max.czi"
    # filename = "/Users/malbert/software/napari-stitcher/image-datasets/MAX_LSM900.czi"
    # filename = "/Users/malbert/software/napari-stitcher/image-datasets/mosaic_test.czi"
    # filename = "/Users/malbert/software/napari-stitcher/image-datasets/arthur_20210216_highres_TR2.czi"
    filename = "/Users/malbert/software/napari-stitcher/image-datasets/arthur_20230223_02_before_ablation-02_20X_max.czi"

    # xims = read_mosaic_czi_into_list_of_spatial_xarrays(filename)

    viewer = napari.Viewer()
    
    # viewer.open("/Users/malbert/software/napari-stitcher/image-datasets/arthur_20220609_WT_emb2_5X_part1_max.czi")

    # wdg = StitcherQWidget(viewer)
    # viewer.window.add_dock_widget(wdg)

    viewer.open(filename)

    # napari.run()