@echo off
setlocal enabledelayedexpansion

:: Specify the base folder containing year directories (G:\Landsat_river\Landsat_data)
set "base_folder=G:\Landsat_river\NDVI_2017_2020"

:: Loop through each year folder inside the base folder (e.g., 1991, 1992, 1993, etc.)
for /d %%Y in ("%base_folder%\*") do (
    :: Check if folder name is numeric (i.e., a year folder)
    set "folder_name=%%~nxY"
    set "invalid="
    for /f "delims=0123456789" %%A in ("!folder_name!") do set "invalid=1"
    if not defined invalid (
        echo Processing year folder: !folder_name!

        :: Set the reprojected folder path for the current year
        set "reprojected_folder=%%Y\NDVI_!folder_name!\reprojected"

        :: Extract the year from the folder name (this should be numeric)
        set "year=!folder_name!"

        :: Output merged raster file with year
        set "output_raster=!reprojected_folder!\merged_average_!year!.tif"

        :: Build a list of input rasters
        set "input_rasters="
        for %%f in ("!reprojected_folder!\*.tif") do (
            set "input_rasters=!input_rasters! "%%f""
        )

        :: Delete the existing output raster, if it exists
        if exist "!output_raster!" (
            del /f /q "!output_raster!"
            echo Existing output raster deleted: !output_raster!
        )

        :: Merge rasters with average resampling
        echo Merging rasters in folder: !reprojected_folder!
        gdalwarp -r average -tr 0.0005 0.0005 -srcnodata 0 -dstnodata 0 -tap -overwrite !input_rasters! "!output_raster!"

        if exist "!output_raster!" (
            echo Successfully created merged raster: !output_raster!
        ) else (
            echo Failed to merge rasters in !reprojected_folder!.
        )
    ) else (
        echo Skipping folder: !folder_name! (not a valid year folder)
    )
)

pause
