import numpy as np
from mvregfus import mv_utils, io_utils, multiview

from dask import delayed


def apply_recursive_dict(func, d):
    res = {}
    if isinstance(d, dict):
        for k, v in d.items():
            res[k] = apply_recursive_dict(func, v)
    else:
        return func(d)
    return res


def load_tiles(view_dict: dict,
               channels: int,
               times: list,
               max_project: bool = True,
               ) -> dict:
    """
    Return: dict of delayed dask arrays
    """

    # load views
    view_ims = {ch: {t: {vdv['view']: delayed(io_utils.read_tile_from_multitile_czi)
                                   (vdv['filename'],
                                    vdv['view'],
                                    ch,
                                    time_index=t,
                                    max_project=max_project,
                                    origin=vdv['origin'],
                                    spacing=vdv['spacing'],
                                    )
                        for vdv in view_dict.values()}
                    for t in times}
                for ch in channels}

    return view_ims


def register_tiles(
                   viewims: dict,
                   pairs: list,
                   reg_channel: int,
                   times: list,
                   registration_binning = None,
                   ref_view_index = 0,
                   ) -> dict:
    """
    Register tiles in a view_dict.

    Use dask.distributed in combination with dask.delayed to do so.

    Return: dict of transform parameters for each tp and view
    """


    # ndim = len(view_dict[list(view_dict.keys())[0]]['spacing'])
    input_channels = sorted(viewims.keys())
    input_times = sorted(viewims[input_channels[0]].keys())
    views = sorted(viewims[input_channels[0]][input_times[0]].keys())

    # load views
    # view_reg_ims = load_tiles(view_dict, times, [reg_channel], max_project)
    view_reg_ims = viewims

    if registration_binning is not None:

        view_reg_ims = apply_recursive_dict(
            lambda x: delayed(mv_utils.bin_stack)(x, registration_binning),
            view_reg_ims)

    # perform pairwise registrations
    pair_ps = {t: {(view1, view2):
                        delayed(multiview.register_linear_elastix)
                                 (view_reg_ims[reg_channel][t][view1],
                                  view_reg_ims[reg_channel][t][view2],
                                  -1, #degree
                                  None,
                                  '',
                                  f'{view1}_{view2}',
                                  None
                                 )
                for view1, view2 in pairs}
            for t in times}

    # get final transform parameters
    ps = {t: delayed(multiview.get_params_from_pairs)(
                                views[ref_view_index],
                                pairs,
                                [pair_ps[t][(v1,v2)] for v1, v2 in pairs],
                                None, # time_alignment_params
                                True, # consider_reg_quality
                                [view_reg_ims[reg_channel][t][view] for view in views],
                                {view: iview for iview, view in enumerate(views)}
                                )
                # for vdv in view_dict.values()}
            for t in times}

    ps = {t: delayed(lambda x: {view: x[iview] for iview, view in enumerate(views)})
                (ps[t])
            for t in times}

    return ps

# get source file path from open layers
def get_source_path_from_viewer(viewer):
    for l in viewer.layers:
        if 'source_file' in l.metadata and l.metadata['source_file'].endswith('.czi'):
            return l.metadata['source_file']
        # if l.source.path is not None and l.source.path.endswith('.czi'):
        #     return l.source.path
    return None


def get_layer_name_from_view_and_ch(view=0, ch=0):
    return 'tile_%03d' %view + '_ch_%03d' %ch


def get_layer_from_view_and_ch(viewer, view, ch):
    # improve: use regexp to match view and channel from e.g. 'view_008_ch_002'
    candidates = [l for l in viewer.layers
        if l.name == get_layer_name_from_view_and_ch(view, ch)]
    # candidates = [l for l in viewer.layers if l.name.startswith('view_%s' %view)\
    #                 and (l.name.endswith(f' [{ch}]')
    #                 or (ch==0 and '[' not in l.name and l.name.endswith('view_%s' %view)))]
    if not len(candidates):
        return None
    else:
        return candidates[0]


def get_view_and_ch_from_layer_name(name):

    view = int(name.split('_')[1])
    ch = int(name.split('_')[-1])

    return view, ch


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
