.. napari-stitcher documentation master file, created by
   sphinx-quickstart on Mon Mar 27 16:43:03 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.  


**********
Quickstart
**********


===============
Program opening
===============

- Activate the environment by running “conda activate napari-stitcher” in Terminal (MacOS) or Anaconda Powershell (Windows)
- Start napari by running “napari”

==================================
File opening and data requirements
==================================

Napari-stitcher is designed to stitch together tiles of large mosaic images. The tiles should have a substantial overlap to allow the stitching algorithm to match structures in the overlap region. Images can be 2D or 3D(+time) and have one or several channels.
Two types of file formats are supported: .czi files and tif files. To load the files, you can either drag and drop the file in the napari window, or use the “File > Open” menu.

For .czi files, the plugin will automatically arrange the tiles together using the stage metadata. Note that if the .czi file contains several scenes, only one can be treated at a time.
For tif files, each file must correspond to one tile. 

==================
Napari interface 
==================

  .. image:: ../_static/nap-stitch_doc.001.jpeg
    :align: center

- All the opened images are displayed in the layer list on the bottom left (1). Tiles can be displayed or hidden by clicking on the eye icon on the bottom left panel. Tiles are displayed with a color code to help the overlap inspection: an alternation of red and green is used so the overlapping structure appears yellow.
- The display settings (contrast, etc.) can be set using the panel at the top left (2) 
- Napari-stitcher widgets ("Mosaic arrangement" and "Stitcher") are open on the right panel (3).

============================
Tiles initial configuration
============================
If you use tif files, the initial arrangement of the tiles can be set using two methods: 

- Regular grid images. Arange the tiles using the "Mosaic arrangement" widget by clicking on Plugins > napari-stitcher > Mosaic arrangement. Set the number of rows and columns, the overlap and the arrangement of the tiles. The order of the tiles in the layer list will be used to arrange the tiles.
- Manual arrangement. Manually displace tiles by dragging them using the Napari "transform" tool. You can toggle the transform option by pressing 2 on the kerboard (activate transform mode) and 1 (deactivate transform mode). Only the selected tile in the layer list will be transformed.

============================
Running the stitching plugin
============================

- Start the stitching widget by clicking on Plugins > napari-stitcher > Stitcher.
- Load the tiles you want to stitch together by selecting only a subset of the layers in the layer list and click on "Selected". Or load all the layers by clicking on "All".
- Define the subset of timepoints you want to use for the stitching by adjusting the “Timepoints” range slider. This is useful to quickly test the stitching in case of a large time series.
- Choose the channel used for calculating the registration with the “Reg channel” dropdown menu.
..
  - Choose between “Stitch” and “Stabilize” to either calculate the best stitching by matching structures in the overlap region, or just perform stabilization of every tile to remove any relative movements between tiles. 
- Click on "Register" to run the stitching optimization. Once the registration is done, you can toggle between the “Original” image and the “Registered” image. Inspect the result by zooming on the overlap region.
- If you are satisfied with the result, click on “Fuse” to fuse all the tiles together in a single image. If there are several channels, each channel will be fused in a separate layer added at the top of the layer list (1).
- Export the output by selecting the output layer(s), and clicking on “File > Save selected layers” (you can also use “Cmd+S” or “Ctrl+S”). If there are several channels, you can save them in a single multichannel file by selecting all output layers.

================
Extra parameters
================

- You can adjust more parameters by opening the "More" tab before running the registration. In case of large images, you can run the registration using binned images. Note that the output images will not be binned, only the data used to compute to run the registration will be binned. 
- If the registration leads to innacurate shifts between tiles, you can adjust the quality threshold. A higher value will lead to smaller shifts between tiles.