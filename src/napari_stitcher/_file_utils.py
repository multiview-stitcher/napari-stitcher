import numpy as np
from aicspylibczi import CziFile


def get_dims_from_multitile_czi(filename, sample_index=0):

    czi = CziFile(filename)
    dims_list = czi.get_dims_shape()

    # https://allencellmodeling.github.io/aicspylibczi/_modules/aicspylibczi/CziFile.html#CziFile.get_dims_shape
    for dims in dims_list:
        if sample_index in range(*dims['S']):
            return dims
    else:
        raise ValueError('sample_index out of range')


def build_view_dict_from_multitile_czi(filename, sample_index=0, max_project=True):

    # import pdb; pdb.set_trace()
    czi = CziFile(filename)
    dims = get_dims_from_multitile_czi(filename, sample_index=sample_index)

    ntiles = dims['M'][1]
    z_shape = dims['Z'][1]
    ndim = 2 if z_shape == 1 else 3

    # spacing = AICSImage(filename).physical_pixel_sizes
    # spacing = np.array([spacing.Y, spacing.X])
    spacing = np.array([1., 1.])
    shape = np.array([dims[dim_s][1] for dim_s in ['Z', 'Y', 'X']])

    # xmin, ymin = np.min([[b.y, b.x] for b in bbs], axis=0)
    bbs = [czi.get_mosaic_tile_bounding_box(M=itile, S=0, C=0, T=0, Z=0) for itile in range(ntiles)]

    if ndim == 3 and not max_project:
        spacing = np.append([1.], spacing, axis=0)
        origins = np.array([[0., b.y, b.x] for b in bbs]) * spacing# - np.array([xmin, ymin])
        
    else:
        # bbs = [czi.get_mosaic_tile_bounding_box(M=itile, S=0, C=0, T=0) for itile in range(ntiles)]
        origins = np.array([[b.y, b.x] for b in bbs]) * spacing# - np.array([xmin, ymin])
        shape = shape[1:]
        
    view_dict = {itile: {
                  'shape': shape,
                  'size': shape,
                  'origin': o,
                  'rotation': 0,
                  'spacing': spacing,
                  'filename': filename,
                  'view': itile,
                  'sample_index': sample_index,
                  }
            for itile, o in zip(range(ntiles), origins)}

    return view_dict