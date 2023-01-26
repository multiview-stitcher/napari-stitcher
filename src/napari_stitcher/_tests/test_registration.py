import pathlib
import numpy as np
import dask

from napari_stitcher import _utils
from mvregfus import mv_utils, io_utils


def test_something():
    pass

# def test_registration_workflow():

#     filename = pathlib.Path(_utils.__file__).parents[2]\
#             /'image-datasets/arthur_20220609_WT_emb2_5X_part1_max.czi'

#     dims = io_utils.get_dims_from_multitile_czi(filename)

#     view_dict = io_utils.build_view_dict_from_multitile_czi(
#                         filename,
#                         S=0,
#                         max_project=True,
#                         )

#     views = list(view_dict.keys())

#     pairs = mv_utils.get_registration_pairs_from_view_dict(view_dict)

#     times = range(dims['T'][0], min(dims['T'][1], dims['T'][0]+2))
#     reg_channel = dims['C'][0]

#     viewims = _utils.load_tiles(view_dict, times, [reg_channel], max_project=True)

#     ps = _utils.register_tiles(
#                           viewims,
#                           pairs,
#                           reg_channel = reg_channel,
#                           times = times,
#                           registration_binning=[2, 2],
#                           )

#     assert(type(ps[times[0], views[0]]), np.ndarray)

#     return


# def test_image_array():

#     from dask import delayed
#     from mvregfus.image_array import ImageArray

#     imar = ImageArray(np.random.rand(10,10))
#     res = delayed(lambda x: x)(imar).compute(scheduler='threads')

#     assert(ImageArray, type(res))


# if __name__ == '__main__':

#     # ps = test_registration_workflow()

#     filename = pathlib.Path(_utils.__file__).parents[2]\
#             /'image-datasets/arthur_20220609_WT_emb2_5X_part1_max.czi'

#     view_dict = io_utils.build_view_dict_from_multitile_czi(filename, S=0, max_project=True)
#     viewims = _utils.load_tiles(view_dict, [0], [0,1], max_project=True)
#     pairs = mv_utils.get_registration_pairs_from_view_dict(view_dict)

#     ps = _utils.register_tiles(
#                           viewims,
#                           pairs,
#                           reg_channel = 0,
#                           times = [0,1],
#                           registration_binning=[4, 4],
#                           )

#     import dask.array as da
#     from napari.utils import progress
#     import numpy as np
#     from tqdm.dask import TqdmCallback

#     with TqdmCallback(tqdm_class=progress, desc="description"):
#         psc = dask.compute(ps, scheduler='threading')[0]


    # test_image_array()
    # filename = pathlib.Path(_utils.__file__).parents[2]\
    #         /'image-datasets/arthur_20220609_WT_emb2_5X_part1_max.czi'

    # dims = io_utils.get_dims_from_multitile_czi(filename)

    # view_dict = io_utils.build_view_dict_from_multitile_czi(
    #                     filename,
    #                     S=0,
    #                     max_project=True,
    #                     )

    # views = list(view_dict.keys())

    # pairs = mv_utils.get_registration_pairs_from_view_dict(view_dict)

    # times = range(dims['T'][0], min(dims['T'][1], dims['T'][0]+2))
    # reg_channel = dims['C'][0]

    # ps = _utils.register_tiles(view_dict,
    #                       pairs,
    #                       reg_channel = reg_channel,
    #                       times = times,
    #                       max_project=True,
    #                       dask_client=None,
    #                       )

# from distributed.protocol import dask_serialize, dask_deserialize
# from typing import Tuple, Dict, List
# from dask import delayed
# import numpy as np
# from dask.distributed import Client
# Client()

# from mvregfus.image_array import ImageArray
# @dask_serialize.register(ImageArray)
# def serialize(ar: ImageArray) -> Tuple[Dict, List[bytes]]:
#     print('la')
#     header = {}
#     frames = [ar.name.encode()]
#     return header, frames

# @dask_deserialize.register(ImageArray)
# def deserialize(header: Dict, frames: List[bytes]) -> ImageArray:
#     return ImageArray(frames[0].decode())

# imar = ImageArray(np.random.rand(10,10))
# delayed(lambda x: x)(imar).compute(scheduler='distributed')



# from distributed.protocol import dask_serialize, dask_deserialize,\
#                                  serialize_bytes, deserialize_bytes
# from typing import Tuple, Dict, List
# from dask import delayed
# import numpy as np
# from dask.distributed import Client
# import distributed.protocol
# from mvregfus.image_array import ImageArray
# Client()

# @dask_serialize.register(ImageArray)
# def serialize(imar):
#     print('LALALALALLLALALLA')
#     meta = imar.get_meta_dict()
#     ar = np.array(imar)
#     header, frames = serialize_bytes([meta,ar])
#     return header, frames

# @dask_deserialize.register(ImageArray)
# def deserialize(header, frames):
#     [meta,ar] = deserialize_bytes(header,frames)
#     return ImageArray(ar, meta=meta)

# imar = ImageArray(np.random.rand(10,10))
# delayed(lambda x: x)(imar).compute(scheduler='distributed')


# from dask import delayed
# import numpy as np
# from dask.distributed import Client
# from dask.distributed import Client
# from mvregfus.image_array import ImageArray
# Client()

# from distributed.protocol import register_generic
# register_generic(ImageArray)

# imar = ImageArray(np.random.rand(10,10))
# delayed(lambda x: x)(imar).compute(scheduler='distributed')