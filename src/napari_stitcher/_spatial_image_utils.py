import numpy as np
import xarray as xr
import transformations as tfs


def assign_si_coords_from_params(xim, p=None):
    """
    Assume that matrix p (shape ndim+1) is given with dim order
    equal to those in im (should be Z, Y, X)
    """

    spatial_dims = [dim for dim in ['Z', 'Y', 'X'] if dim in xim.dims]
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

    spatial_dims = [dim for dim in ['Z', 'Y', 'X'] if dim in xim.dims]

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
    return [dim for dim in ['Z', 'Y', 'X'] if dim in xim.dims]


def get_origin_from_xim(xim, asarray=False):

    spatial_dims = get_spatial_dims_from_xim(xim)
    origin = {dim: xim.coords[dim][0] for dim in spatial_dims}

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
    spacing = {dim: xim.coords[dim][1] - xim.coords[dim][0] for dim in spatial_dims}

    if asarray:
        spacing = np.array([spacing[sd] for sd in spatial_dims])

    return spacing


def ensure_time_dim(xim):
    if 'T' not in xim.dims:
        xim = xim.expand_dims(['T'])
    return xim


def get_ndim_from_xim(xim):
    return len(get_spatial_dims_from_xim(xim))


# def get_spatial_image_from_array_and_params(im, p=None):
#     """
#     Assume that matrix p (shape ndim+1) is given with dim order
#     equal to those in im (should be Z, Y, X)
#     """

#     ndim = im.ndim

#     input_dims = ['Z', 'Y', 'X'][-ndim:]

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