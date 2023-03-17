import os

import numpy as np
from mvregfus import mv_utils, io_utils
from mvregfus.image_array import ImageArray

from dask import delayed
import dask.array as da


def load_tiles(view_dict: dict,
               channels: int,
               times: list,
               max_project: bool = True,
               ) -> dict:
    """
    Return: dict of delayed dask arrays
    """

    # # load views
    # view_ims = {ch: {t: {vdv['view']: delayed(io_utils.read_tile_from_multitile_czi)
    #                                (vdv['filename'],
    #                                 vdv['view'],
    #                                 ch,
    #                                 time_index=t,
    #                                 max_project=max_project,
    #                                 origin=vdv['origin'],
    #                                 spacing=vdv['spacing'],
    #                                 )
    #                     for vdv in view_dict.values()}
    #                 for t in times}
    #             for ch in channels}

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
        # if 'source_file' in l.metadata and l.metadata['source_file'].endswith('.czi'):
        #     return l.metadata['source_file']
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


# def get_list_of_source_paths_from_layers(layers):

#     sources = []
#     for l in layers:
#         if l.source.path is not None and l.source.path.endswith('.czi'):
#             sources.append(l.source.path)

#     return sources


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


# def get_layer_from_view_and_ch(viewer, view, ch):
#     # improve: use regexp to match view and channel from e.g. 'view_008_ch_002'
#     candidates = [l for l in viewer.layers
#         if l.name == get_layer_name_from_view_and_ch(view, ch)]
#     # candidates = [l for l in viewer.layers if l.name.startswith('view_%s' %view)\
#     #                 and (l.name.endswith(f' [{ch}]')
#     #                 or (ch==0 and '[' not in l.name and l.name.endswith('view_%s' %view)))]
#     if not len(candidates):
#         return None
#     else:
#         return candidates[0]


# def transmit_params_to_viewer(viewer, params, channels, times, views):

#     for ch in channels:
#         for t in times:
#             for view in views:
#                 l = get_layer_from_view_and_ch(viewer, view, ch)

#                 if l is not None:
#                     l.params = params[t][view]

#     for l in viewer.layers:

#         if l.source.path is not None and l.source.path.endswith('.czi'):
#             l.params = params


# def transmit_params_to_layer(viewer, params, ch, t, view, stack_props, view_stack_props):
#     l = get_layer_from_view_and_ch(viewer, view, ch)
#     l.affine = params_to_napari_affine(params[t][view], stack_props, view_stack_props)
#     return


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


# def visualize_tiles():



# def transform_tiles(viewims: dict,
#                     ps: dict,
#                     reg_channel: int,
#                     ) -> dict:
#     """
#     Transform tiles in a view_dict.

#     Use dask.distributed in combination with dask.delayed to do so.

#     Return: dict of transformed images for each tp and view
#     """

#     ndim = len(viewims[0][0][0]['spacing'])
#     views = sorted(viewims[0][0].keys())

#     # load views
#     # view_reg_ims = load_tiles(view_dict, times, [reg_channel], max_project)
#     view_reg_ims = viewims

#     # transform views
#     view_reg_ims = {t: {vdv['view']: delayed(multiview.transform_stack)
#                                    (view_reg_ims[reg_channel][t][vdv['view']],
#                                     ps[t][vdv['view']],
#                                     ndim,
#                                     )
#                 for vdv in view_dict.values()}
#             for t in times}

#     return view_reg_ims

