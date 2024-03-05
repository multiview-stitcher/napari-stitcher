"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/guides.html?#readers
"""

from multiview_stitcher import msi_utils
from multiview_stitcher.io import read_mosaic_image_into_list_of_spatial_xarrays,\
    METADATA_TRANSFORM_KEY

from napari_stitcher import viewer_utils


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
        return read_mosaic
    else:
        return None
    

def read_mosaic(path, scene_index=None):
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

    sims = read_mosaic_image_into_list_of_spatial_xarrays(paths[0], scene_index=scene_index)

    msims = [msi_utils.get_msim_from_sim(sim, scale_factors=[])
             for sim in sims]

    out_layers = viewer_utils.create_image_layer_tuples_from_msims(
        msims,
        transform_key=METADATA_TRANSFORM_KEY,
        data_as_array=False)

    return out_layers


if __name__ == "__main__":

    from multiview_stitcher.sample_data import get_mosaic_sample_data_path

    filename = get_mosaic_sample_data_path()

    sims = read_mosaic_image_into_list_of_spatial_xarrays(filename)

    from multiview_stitcher import msi_utils

    msim = msi_utils.get_msim_from_sim(sims[0], scale_factors=[])

    msim_sel = msi_utils.multiscale_sel_coords(msim, {'c':'EGFP'})