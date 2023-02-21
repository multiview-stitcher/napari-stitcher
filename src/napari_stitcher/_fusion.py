import numpy as np

import dask.array as da
from dask import delayed

from mvregfus.mv_utils import get_sigmoidal_border_weights_ndim_only_one
from mvregfus.multiview import calc_stack_properties_from_views_and_params
from mvregfus.mv_utils import params_to_matrix

from dask_image import ndinterp


def combine_stack_props(stack_props_list):

    combined_stack_props = {}
    combined_stack_props['origin'] = np.min([sp['origin'] for sp in stack_props_list], axis=0)
    combined_stack_props['spacing'] = np.min([sp['spacing'] for sp in stack_props_list], axis=0)
    combined_stack_props['size'] = np.max([np.ceil((sp['origin'] + sp['size'] * sp['spacing']\
                                    - combined_stack_props['origin']) / combined_stack_props['spacing'])
                                    for sp in stack_props_list], axis=0)
    # import pdb; pdb.set_trace()
    return combined_stack_props

def fuse_tiles(viewims: dict,
               params: dict,
               view_dict: dict,
):

    input_channels = sorted(viewims.keys())
    input_times = sorted(viewims[input_channels[0]].keys())
    views = sorted(viewims[input_channels[0]][input_times[0]].keys())

    for view in views:
        view_dict[view]['size'] = view_dict[view]['shape']

    field_stack_props = [delayed(calc_stack_properties_from_views_and_params)(
                [view_dict[view] for view in views],
                params[t],
                view_dict[views[0]]['spacing'],
                mode='union',
            )
        for t in input_times]

    fusion_stack_props = delayed(combine_stack_props)(field_stack_props)

    fused_da = \
        da.stack([
            da.stack([
                da.from_delayed(delayed(fuse_field)(
                        viewims[ch][t],
                        params[t],
                        view_dict,
                        fusion_stack_props,
                        ),
                shape=tuple(viewims[ch][t][views[0]].shape),
                dtype=np.uint16,
                )
            for t in input_times])
        for ch in input_channels])

    return fused_da, fusion_stack_props, field_stack_props


def fuse_field(field_ims, params, view_dict, out_stack_props):
    """
    fuse tiles from single timepoint and channel
    """

    # return field_ims[0]

    views = sorted(field_ims.keys())

    ndim = field_ims[0].ndim
    field_ims_t = []
    for view in views:

        p = np.array(params[view])
        matrix = p[:ndim*ndim].reshape(ndim,ndim)
        offset = p[ndim*ndim:]

        # spacing matrices
        Sx = np.diag(out_stack_props['spacing'])
        Sy = np.diag(view_dict[view]['spacing'])

        matrix_prime = np.dot(np.linalg.inv(Sy), np.dot(matrix, Sx))
        offset_prime = np.dot(np.linalg.inv(Sy),
            offset - view_dict[view]['origin'] + np.dot(matrix, out_stack_props['origin']))

        field_ims_t.append(ndinterp.affine_transform(
                                            field_ims[view] + 1, # add 1 so that 0 indicates no data
                                            matrix=matrix_prime,
                                            offset=offset_prime,
                                            order=1,
                                            output_shape=tuple(out_stack_props['size']),
                                        #  output_chunks=tuple([100 for _ in out_stack_props['size']]),
                                            output_chunks=None,
                                            mode='constant',
                                            )
        )

    field_ims_t = da.stack(field_ims_t)

    field_ims_t = field_ims_t.rechunk((len(views), ) + field_ims_t[0].shape)

    field_weights = da.map_blocks(get_sigmoidal_border_weights_ndim_only_one,
        field_ims_t,
        dtype=np.float32,
        **{'width': 10},
        )
    # field_weights = da.map_blocks(get_sigmoidal_border_weights_ndim_mask,
    #     field_ims_t,
    #     dtype=np.float32,
    #     **{'width': 20},
    #     )

    # field_weights = da.map_overlap(get_sigmoidal_border_weights_ndim_only_one,
    #     field_ims_t,
    #     dtype=np.float32,
    #     depth={idim: ([0] + [10] * ndim)[idim] for idim in range(ndim + 1)},
    #     trim=True,
    #     boundary='none',
    #     )

    wsum = da.sum(field_weights, axis=0)
    wsum[wsum==0] = 1

    fused_field = da.sum(field_ims_t * field_weights, axis=0)
    fused_field /= wsum

    fused_field = fused_field - 1  # subtract 1 because of earlier addition

    return fused_field
