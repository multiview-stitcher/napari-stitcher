import numpy as np
from mvregfus import mv_utils, io_utils, multiview

from dask import delayed
import dask.array as da


def apply_recursive_dict(func, d):
    res = {}
    if isinstance(d, dict):
        for k, v in d.items():
            res[k] = apply_recursive_dict(func, v)
    else:
        return func(d)
    return res


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
