
# Loading images

`napari-stitcher` can stitch any image layers that are loaded into napari. You can load images into napari by dragging and dropping files into the viewer, or by using the `File > Open Files` menu.

!!! note "Multi-channel data"
    We recommend using the napari reader plugin `napari-aicsimageio` for loading individual image layers into napari. This is especially useful for multi-channel data, as it automatically names the layers according to the [naming convention](naming_convention.md) used by `napari-stitcher`.

Additionally, `napari-stitcher` supports reading mosaic files which contain multiple pre-positioned tiles. If more than one scene is present, you'll be prompted to select a scene to load. To use this feature, use the `napari-stitcher` reader plugin. Supported file formats include .czi and .lif.

Image files / napari layers to stitch can be 2D/3D(+time).

!!! note "Selecting a reader plugin"
    After dropping an image file onto napari or by using the `File > Open Files` menu, you can select the reader plugin to use.
