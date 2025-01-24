# NDVI Data Preparation Process

This README describes the procedure for extracting NDVI data from existing satellite missions and ensuring compatibility with the project's model. All codes and batch files used are included in the repository. Data is available upon request. Note that all `.bat` files should be run in the **OSGeo4W terminal** or an equivalent terminal with the built-in GDAL library. Be sure to adjust the directories in the batch files to match your local setup.

---

## NDVI Data Acquisition

This part involves acquiring the NDVI data for the Region of Interest (ROI) and calculating the NDVI values.

### Process Flowchart
![NDVI Data Acquisition](Flowchart/NDVI_acquisition.png)

### Steps:

1. **Download LANDSAT data**  
   - Download LANDSAT data for each year (1988–2021) from the [USGS website](https://earthexplorer.usgs.gov/) for the ROI.  
   - Only data from **January to April** of each year was considered, with tiles selected based on the **least amount of cloud coverage**.  
   - This assumption ensures that the region remains consistent during this short time period.  
   - Typically, 6–7 tiles per year were selected, depending on data availability.

2. **Calculate NDVI (Normalized Difference Vegetation Index)**  
   - NDVI is calculated differently for each Landsat mission due to differences in band configurations:
     - **Landsat 5 (1988–1999)**  
     - **Landsat 7 (2000–2017)**  
     - **Landsat 8 (2018–2021)**  
   - **Function/Notebook Used**:  
     - `NDVI.ipynb` calculates the NDVI for a given Landsat tile.

---

## NDVI Processing

This part includes reprojecting, merging, clipping, and transforming the NDVI data to ensure it aligns with the binary classification files.

### Processing Flowchart
![NDVI Processing](Flowchart/NDVI_processing.png)

### Steps:

1. **Reproject the tiles to WGS84**  
   - Reproject all individual yearly tiles to the EPSG:4326 coordinate system, as the ROI spans multiple local map projections.  
   - **Function/Notebook Used**:  
     - `reprojection_ndvi_new.bat` loops through all year folders, creates reprojection sub-folders, transforms `.tif` files to EPSG:4326, and saves them in the sub-folders.

2. **Merge tiles together per year**  
   - Merge the reprojected NDVI tiles into a single raster per year. Average sampling is applied in overlapping regions, and the resolution is resampled to 0.0005 arc-seconds for faster processing.  
   - **Function/Notebook Used**:  
     - `merge_all.bat` merges rasters, applies average sampling, resamples to 0.0005 arc-seconds, and saves outputs to corresponding year folders.

3. **Interpolate to fill no-data values**  
   - Landsat 5 data after 2004 suffers from stripe effects due to sensor failure. Missing data is interpolated using a moving average with 20 neighboring pixels in QGIS.  
   - **Output**: Corrected raster files.

4. **Slice tiles into smaller sections**  
   - Divide the merged yearly NDVI tiles into 30 smaller tiles, corresponding to water classification binary data (28 for training, 1 for validation, 1 for testing).  
   - **Function/Notebook Used**:  
     - `data_clip.bat` uses predefined boundaries to clip the NDVI tiles. Outputs are stored in a separate folder.

5. **Create grids and resample NDVI tiles**  
   - Generate grids using binary TIFF files and resample NDVI tiles to align with the grid.  
   - **Function/Notebook Used**:  
     - `Create_point_files.ipynb` generates pixel center points from binary TIFF files and saves them in `.txt` format (x, y coordinates).  
     - `resampling_NDVI.ipynb` resamples NDVI values for a given year and area to align with the binary file’s pixel layout.

6. **Transform the tiles to match binary files**  
   - Rotate and scale NDVI tiles to match the orientation of binary classification tiles (river flow oriented top to bottom).  
   - **Function/Notebook Used**:  
     - `transform_image.ipynb` performs the transformation using functions adapted from Antonio’s repository.
