"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/guides.html?#readers
"""
import numpy as np


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
        return czi_reader_function
    else:
        return None


from mvregfus import io_utils, mv_utils
from napari_stitcher import _utils
import dask.array as da
from dask import delayed
import time
def czi_reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.

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

    max_project = True
    dims = io_utils.get_dims_from_multitile_czi(paths[0])
    view_dict = io_utils.build_view_dict_from_multitile_czi(paths[0], max_project=max_project)
    views = np.array([view for view in sorted(view_dict.keys())])
    pairs = mv_utils.get_registration_pairs_from_view_dict(view_dict)
    ndim = [2, 3][int(dims['Z'][1] > 1)]

    print(dims)

    channels = range(dims['C'][0], dims['C'][1])
    times = range(dims['T'][0], dims['T'][1])

    # view_das = []
    # for vdv in view_dict.values():
    #     view_das.append(da.from_delayed([[
    #         delayed(io_utils.read_tile_from_multitile_czi)
    #                     (vdv['filename'],
    #                         vdv['view'],
    #                         ch,
    #                         time_index=t,
    #                         max_project=max_project,
    #                         origin=vdv['origin'],
    #                         spacing=vdv['spacing'],
    #                         )
    #         for ch in channels]
    #             for t in times], shape=(len(times), len(channels)) + tuple(vdv['shape'])))

    view_das = []
    for vdv in view_dict.values():
        view_das.append(
                da.stack([
                    da.stack([
                        da.from_delayed(delayed(io_utils.read_tile_from_multitile_czi)
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
                    for ch in channels])
                for t in times])
        )

    # set target stack properties to those of first view
    stack_props = view_dict[0]
    view_stack_props = view_dict

    # assume identity transf parameters
    transf_params = [mv_utils.matrix_to_params(np.eye(ndim+1)) for i in range(len(views))]

    # get affine parameters
    ps = []
    for iview in range(len(views)):

        p = mv_utils.params_to_matrix(transf_params[iview])

        """
        y = Ax+c
        y=sy*yp+oy
        x=sx*xp+ox
        sy*yp+oy = A(sx*xp+ox)+c
        yp = syi * A*sx*xp + syi  *A*ox +syi*(c-oy)
        A' = syi * A * sx
        c' = syi  *A*ox +syi*(c-oy)
        """
        sx = np.diag(list((stack_props['spacing'])))
        sy = np.diag(list((view_stack_props[iview]['spacing'])))
        syi = np.linalg.inv(sy)
        p[:ndim, ndim] = np.dot(syi, np.dot(p[:ndim, :ndim], stack_props['origin'])) \
                   + np.dot(syi, (p[:ndim, ndim] - view_stack_props[iview]['origin']))
        p[:ndim, :ndim] = np.dot(syi, np.dot(p[:ndim, :ndim], sx))
        p = np.linalg.inv(p)

        ps.append(p)

    layer_type = "image"
    file_id = time.time()

    return [(view_das[iview],
            {
             'contrast_limits': [[0, 100], [0,255]],
             'name': 'view_%s' %view,
             'colormap': 'gray_r',
             'gamma': 0.6,
             'channel_axis': 1,
             'affine': ps[iview],
             'cache': False,
             'metadata': {'load_id': file_id,
                          'view_dict': view_dict[iview]},
             },
            layer_type)
                for iview, view in enumerate(views)][:]

    # construct dask array

    # # load all files into array
    # arrays = [np.load(_path) for _path in paths]
    # # stack arrays into single array
    # data = np.squeeze(np.stack(arrays))

    # # optional kwargs for the corresponding viewer.add_* method
    # add_kwargs = {}

    # layer_type = "image"  # optional, default is "image"
    # return [(data, add_kwargs, layer_type)]

if __name__ == "__main__":
    # tmp = czi_reader_function("/Users/malbert/software/napari-stitcher/image-datasets/arthur_20220621_premovie_dish2-max.czi")
    # fn = "/Users/malbert/software/napari-stitcher/image-datasets/yu_220829_WT_quail_st4+_x40_zoom0_5_5x5_488ZO1-568Sox2-647Tbra-max.czi"
    fn = "/Users/malbert/software/napari-stitcher/image-datasets/arthur_20220621_premovie_dish2-max.czi"
    tmp = czi_reader_function(fn)
