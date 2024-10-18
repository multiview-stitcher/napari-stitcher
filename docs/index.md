---
title: Overview
---

# napari-stitcher

A napari plugin for stitching large multi-positioning datasets in 2/3D+t using [`multiview-stitcher`](https://github.com/multiview-stitcher/multiview-stitcher).

![](images/napari-stitcher-loaded-mosaic-annotated.png)
<small>Image data by Arthur Michaut @ Jérôme Gros Lab @ Institut Pasteur.</small>

#### Quick guide:

1. Directly stitch napari layers: Use napari to load, visualize and [preposition](prearrangement.md) the tiles to be stitched.
2. When working with multi-channel data, stick to the following [naming convention](naming_convention.md): `{tile} :: {channel}`.
3. Load either all or just a subset of the layers into the plugin.
4. Choose registration options: registration channel, binning and more.
5. Stitching = registration (refining the positions, optional) + fusion (joining the tiles into a single image).
6. The registration result is shown in the viewer and the fused channels are added as new layers.

## Demo

<video controls>
<source src="https://github.com/multiview-stitcher/napari-stitcher/raw/refs/heads/main/docs/images/demo_3d.mp4" type="video/mp4">
</video>

<small>This demo uses the awesome [`napari-threedee`](https://github.com/napari-threedee/napari-threedee) for prepositioning the tiles. Image data: [BigStitcher](https://imagej.net/plugins/bigstitcher/).</small>