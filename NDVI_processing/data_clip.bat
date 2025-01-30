@echo off
setlocal enabledelayedexpansion

:: Directories
set "raster_folder=G:\Landsat_river\NDVI_2017_2020\merged"
set "shapefile_folder=G:\Landsat_river\polygons_antonio"
set "output_folder=G:\Landsat_river\NDVI_2017_2020\antonio_clipped"

:: Ensure the output folder exists
if not exist "!output_folder!" mkdir "!output_folder!"

:: Loop through each .tif file
for %%R in ("!raster_folder!\*.tif") do (
    :: Extract the year from the raster file name (e.g., 1998 from 1998.tif)
    set "raster_file=%%~nR"
    set "year=%%~nR"

    :: Loop through each shapefile
    for %%S in ("!shapefile_folder!\*.shp") do (
        :: Extract the shapefile name without extension (e.g., training_r1 from training_r1.shp)
        set "shapefile_name=%%~nS"

        :: Construct the output file name
        set "output_file=!output_folder!\!year!_!shapefile_name!.tif"

        :: Perform the clipping operation
        echo Clipping !raster_file!.tif with !shapefile_name!.shp
        gdalwarp -cutline "%%S" -crop_to_cutline -dstnodata 0 -overwrite "%%R" "!output_file!"

        if exist "!output_file!" (
            echo Successfully created: !output_file!
        ) else (
            echo Failed to clip !raster_file!.tif with !shapefile_name!.shp
        )
    )
)

pause
