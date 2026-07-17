import numpy as np

from napari_stitcher import _sample_data


def _level_as_array(layer_data, level):
    return layer_data[level].data[0].compute()


def test_multiscale_digits_sample_data():
    layer_tuples = _sample_data.multiscale_digits()

    assert len(layer_tuples) == 4

    expected_shapes = [(1, 500, 500), (1, 250, 250), (1, 125, 125)]
    expected_translations = [
        (0, 0),
        (0, 500),
        (500, 0),
        (500, 500),
    ]

    reference_levels = None
    for layer_data, kwargs, layer_type in layer_tuples:
        assert layer_type == "image"
        assert kwargs["multiscale"]
        assert kwargs["colormap"] == "gray"
        assert kwargs["contrast_limits"] == [0, 255]
        assert [tuple(level.shape) for level in layer_data] == expected_shapes

        tile_index = int(kwargs["name"].split("_tile_")[1].split(" ")[0])
        assert np.allclose(
            kwargs["translate"], expected_translations[tile_index]
        )

        levels = [_level_as_array(layer_data, level) for level in range(3)]
        if reference_levels is None:
            reference_levels = levels
        else:
            for level, reference_level in zip(levels, reference_levels):
                assert np.array_equal(level, reference_level)

    assert not np.array_equal(
        reference_levels[0][::2, ::2], reference_levels[1]
    )
    assert not np.array_equal(
        reference_levels[1][::2, ::2], reference_levels[2]
    )
