<!-- This file is a placeholder for customizing description of your plugin 
on the napari hub if you wish. The readme file will be used by default if
you wish not to do any customization for the napari hub listing.

If you need some help writing a good description, check out our 
[guide](https://github.com/chanzuckerberg/napari-hub/wiki/Writing-the-Perfect-Description-for-your-Plugin)
-->

# Description

This plugin stitches images (napari layers) in 2/3D+t. Images can either be (pre-)aligned manually or on a grid.

Here's it's [user guide](https://multiview-stitcher.github.io/napari-stitcher/) and [github repository](https://github.com/multiview-stitcher/napari-stitcher). napari-stitcher uses [`multiview-stitcher`](https://github.com/multiview-stitcher/multiview-stitcher) for registration and fusion.

## Overview

![](https://github.com/multiview-stitcher/napari-stitcher/raw/refs/heads/main/docs/images/napari-stitcher-loaded-mosaic-annotated.png)
<small>Image data by Arthur Michaut @ Jérôme Gros Lab @ Institut Pasteur.</small>

1. Directly stitch napari layers: Use napari to load, visualize and [preposition](prearrangement.md) the tiles to be stitched.
2. When working with multi-channel data, stick to the following [naming convention](naming_convention.md): `{tile} :: {channel}`.
3. Load either all or just a subset of the layers into the plugin.
4. Choose registration options: registration channel, binning and more.
5. Stitching = registration (refining the positions, optional) + fusion (joining the tiles into a single image).
6. The registration result is shown in the viewer and the fused channels are added as new layers.

## Demo

[Link](https://github.com/user-attachments/assets/8773e49f-af18-4ff3-ab2f-2a5f1b1cadf2) to video demo.

<video controls>
<source src="https://github.com/multiview-stitcher/napari-stitcher/raw/refs/heads/main/docs/images/demo_3d.mp4" type="video/mp4">
</video>

<small>This demo uses the awesome [`napari-threedee`](https://github.com/napari-threedee/napari-threedee) for prepositioning the tiles. Image data: [BigStitcher](https://imagej.net/plugins/bigstitcher/).</small>