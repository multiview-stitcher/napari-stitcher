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
- Start the plugin in the menu plugins > napari-stitcher > Stitcher

============
File opening
============

- Drag and drop a .czi file or tif file
- The file can be 2D/3D(+time). If it contains several scenes, only one can be treated at a time.

==================
Parameters setting 
==================

  .. image:: ../_static/nap-stitch_doc.001.png
    :align: center

- The display settings (contrast, etc.) can be set using the panel at the top left (1)
- Tiles and channels can be displayed or hidden by clicking on the eye icon on the bottom left panel (2). Tiles are displayed with a color code to help the overlap inspection: an alternation of red and green is used so the overlapping structure appears yellow. 
- The plugin panel is at the top right (3).


============================
Running the stitching plugin
============================

- The timepoints used to run the stitching can be adjusted with the “Timepoints” range slider. This is useful to quickly test the stitching in case of a large time series.
- Choose the channel used for calculating the registration with the “Reg channel” slider.
- Choose between “Stitch” and “Stabilize” to either calculate the best stitching by matching structures in the overlap region, or just perform stabilization of every tile to remove any relative movements between tiles. 
- Once the stitching or stabilization is performed, you can toggle between the “Original” image and the “Registered” image. Inspect the result by zooming on the overlap region.
- If you are satisfied with the result, click on “Fuse” to fuse all the tiles together in a single image. 
- Export the output by slecting the output layer (4), and clicking on “File > Save selected layers” (you can also use “Cmd+S” or “Ctrl+S”). Use .tif format for the output. 