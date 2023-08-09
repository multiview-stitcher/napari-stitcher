import numpy as np
import xarray as xr
import transformations as tfs
import copy

import spatial_image as si


SPATIAL_DIMS = ['z', 'y', 'x']


def assign_si_coords_from_params(xim, p=None):
    """
    Assume that matrix p (shape ndim+1) is given with dim order
    equal to those in im (should be Z, Y, X)
    """

    spatial_dims = [dim for dim in ['z', 'y', 'x'] if dim in xim.dims]
    ndim = len(spatial_dims)


    # if ndim==2 temporarily expand to three dims to use
    # transformations.py for decomposition
    if ndim == 2:
        M = np.eye(4)
        M[1:, 1:] = p
        p = M.copy()

    scale, shear, angles, translate, perspective = tfs.decompose_matrix(p)
    direction_matrix = tfs.euler_matrix(angles[0], angles[1], angles[2]) # use tfs.compose_matrix here for consistency

    if ndim == 2:
        scale = scale[1:]
        translate = translate[1:]
        direction_matrix = direction_matrix[1:, 1:]

    # assign new coords
    for idim, dim in enumerate(spatial_dims):
        coords = np.linspace(0, len(xim.coords[dim])-1, len(xim.coords[dim]))
        coords *= scale[idim]
        coords += translate[idim]
        xim.coords[dim] = coords

    xim.attrs['direction'] = direction_matrix

    return xim


def compose_params(origin, spacing):

    ndim = len(origin)

    if ndim == 2:
        origin = np.concatenate([[0.], origin])
        spacing = np.concatenate([[1.], spacing])
    

    M = tfs.compose_matrix(scale=spacing, translate=origin)

    if ndim == 2:
        M = M[1:, 1:]

    return M


def get_data_to_world_matrix_from_spatial_image(xim):

    spatial_dims = [dim for dim in ['z', 'y', 'x'] if dim in xim.dims]

    # ndim = len([dim for dim in xim.dims if dim in spatial_dims])
    # p = np.eye(ndim + 1)

    ndim = len(spatial_dims)
    p = np.eye(ndim + 1)

    scale, offset = dict(), dict()
    for _, dim in enumerate(spatial_dims):
        coords = xim.coords[dim]

        if len(coords) > 1:
            scale[dim] = coords[1] - coords[0]
        else:
            scale[dim] = 1

        offset[dim] = coords[0]

    S = np.diag([scale[dim] for dim in spatial_dims] + [1])
    T = np.eye(ndim + 1)
    T[:ndim, ndim] = [offset[dim] for dim in spatial_dims]

    # direction not implemented yet
    # p = np.matmul(T, np.matmul(S, xim.attrs['direction']))
    p = np.matmul(T, S)

    return p


def get_spatial_dims_from_xim(xim):
    return [dim for dim in ['z', 'y', 'x'] if dim in xim.dims]


def get_origin_from_xim(xim, asarray=False):

    spatial_dims = get_spatial_dims_from_xim(xim)
    origin = {dim: float(xim.coords[dim][0]) for dim in spatial_dims}

    if asarray:
        origin = np.array([origin[sd] for sd in spatial_dims])

    return origin


def get_shape_from_xim(xim, asarray=False):

    spatial_dims = get_spatial_dims_from_xim(xim)
    shape = {dim: len(xim.coords[dim]) for dim in spatial_dims}

    if asarray:
        shape = np.array([shape[sd] for sd in spatial_dims])

    return shape


def get_spacing_from_xim(xim, asarray=False):
    
    spatial_dims = get_spatial_dims_from_xim(xim)
    spacing = {dim: float(xim.coords[dim][1] - xim.coords[dim][0])
               if len(xim.coords[dim]) > 1 else 1.0
               for dim in spatial_dims}

    if asarray:
        spacing = np.array([spacing[sd] for sd in spatial_dims])

    return spacing


def ensure_time_dim(sim):

    if 't' in sim.dims:
        return sim
    else:
        xim = sim.expand_dims(['t'], axis=0)
    
    sim = get_sim_from_xim(xim)

    sim.attrs.update(copy.deepcopy(xim.attrs))
    
    return sim


def get_sim_from_xim(xim):

    spacing = get_spacing_from_xim(xim)
    origin = get_origin_from_xim(xim)

    sim = si.to_spatial_image(
        xim,
        dims = xim.dims,
        scale=spacing,
        translation=origin,
        t_coords=xim.coords['t'] if 't' in xim.dims else None,
        c_coords=xim.coords['c'] if 'c' in xim.dims else None,
    )

    sim.attrs.update(copy.deepcopy(xim.attrs))

    return sim


def get_ndim_from_xim(xim):
    return len(get_spatial_dims_from_xim(xim))


def get_affine_from_xim(xim, transform_key=None):

    # ndim = get_ndim_from_xim(xim)
    affine = xim.attrs['transforms'][transform_key]#.reshape((ndim + 1, ndim + 1))

    return affine


def get_tranform_keys_from_xim(xim):

    return list(xim.attrs['transforms'].keys())


def set_xim_affine(xim, xaffine, transform_key=None, base_transform_key=None):

    # xim = copy.deepcopy(xim)

    if 'transforms' not in xim.attrs.keys():
        xim.attrs['transforms'] = dict()

    if not base_transform_key is None:
        xaffine = matmul_xparams(
            xaffine,
            get_affine_from_xim(xim, transform_key=base_transform_key))

    xim.attrs['transforms'][transform_key] = xaffine

    return


def get_center_of_xim(xim, transform_key=None):
    
    ndim = get_ndim_from_xim(xim)

    center = np.array([xim.coords[dim][len(xim.coords[dim])//2]
                    for dim in get_spatial_dims_from_xim(xim)])
    
    if transform_key is not None:
        affine = get_affine_from_xim(xim, transform_key=transform_key)
        # affine = np.array(affine.sel(t=affine.coords['t'][0]))
        affine = np.array(affine)
        center = np.concatenate([center, np.ones(1)])
        center = np.matmul(affine, center)[:ndim]

    return center


def xim_sel_coords(xim, sel_dict):
    """
    Select coords from xim and its transform attributes
    """

    sxim = xim.copy(deep=True)
    sxim = sxim.sel(sel_dict)

    # sel transforms which are xr.Datasets in the mxim attributes
    for data_var in xim.attrs['transforms']:
        for k, v in sel_dict.items():
            if k in xim.attrs['transforms'][data_var].dims:
                sxim.attrs['transforms'][data_var] = sxim.attrs['transforms'][data_var].sel({k: v})

    return sxim


def identity_transform(ndim, t_coords=None):

    if t_coords is None:
        params = xr.DataArray(
            np.eye(ndim + 1),
            dims=['x_in', 'x_out'])
    else:
        params = xr.DataArray(
            len(t_coords) * [np.eye(ndim + 1)],
            dims=['t', 'x_in', 'x_out'],
            coords={'t': t_coords})

    return params


def affine_to_xr(affine, t_coords=None):

    if t_coords is None:
        params = xr.DataArray(
            affine,
            dims=['x_in', 'x_out'])
    else:
        params = xr.DataArray(
            len(t_coords) * [affine],
            dims=['t', 'x_in', 'x_out'],
            coords={'t': t_coords})

    return params


def matmul_xparams(xparams1, xparams2):
    return xr.apply_ufunc(np.matmul,
        xparams1,
        xparams2,
        input_core_dims=[['x_in', 'x_out']]*2,
        output_core_dims=[['x_in', 'x_out']],
        vectorize=True)


def invert_xparams(xparams):
    return xr.apply_ufunc(np.linalg.inv,
        xparams,
        input_core_dims=[['x_in', 'x_out']],
        output_core_dims=[['x_in', 'x_out']],
        vectorize=True)



# def get_spatial_image_from_array_and_params(im, p=None):
#     """
#     Assume that matrix p (shape ndim+1) is given with dim order
#     equal to those in im (should be Z, Y, X)
#     """

#     ndim = im.ndim

#     input_dims = ['z', 'y', 'x'][-ndim:]

#     # if p is None:
#     #     p = np.eye(ndim + 1)

#     # if ndim==2 temporarily expand to three dims to use
#     # transformations.py for decomposition
#     if ndim == 2:
#         M = np.eye(4)
#         M[1:, 1:] = p
#         p = M.copy()

#     scale, shear, angles, translate, perspective = tfs.decompose_matrix(p)
#     direction_matrix = tfs.euler_matrix(angles[0], angles[1], angles[2])

#     if ndim == 2:
#         scale = scale[1:]
#         translate = translate[1:]
#         direction_matrix = direction_matrix[1:, 1:]

#     shape = im.shape
#     xim = xr.DataArray(im,
#         {dim: scale[idim] * np.linspace(0, shape[idim]-1, shape[idim]) - translate[idim]
#             for idim, dim in enumerate(input_dims)},
#         attrs={'direction': direction_matrix}
#     )

#     return xim
