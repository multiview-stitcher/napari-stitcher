.. napari-stitcher documentation master file, created by
   sphinx-quickstart on Mon Mar 27 16:43:03 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.  


************
Installation
************


============================
napari-stitcher installation
============================

.. _installation:

with conda
==========

- If you haven't installed Python yet, install Python 3.9: download miniconda 3.9: https://docs.conda.io/en/latest/miniconda.html
- Open a Terminal (Mac & Linux) or open an Anaconda powershell (Windows)
- Create environment: run :code:`conda create -y -n napari-env -c conda-forge python=3.9`
- Activate environment (to be run every time you open a new terminal): run :code:`conda activate napari-env`
- Install napari: :code:`conda install -c conda-forge napari`

- To install Napari-Stitcher, currently: download (e.g. clone) napari-stitcher and install using: :code:`pip install -e napari-stitcher`



..Archive overview
  ================


============================
napari-stitcher dependencies
============================

- numpy<=1.23
- magicgui
- qtpy
- dask_image
- zarr
- tifffile
- aicsimageio
- aicspylibczi
- multiview-stitcher
