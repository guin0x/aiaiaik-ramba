@echo off
setlocal enabledelayedexpansion

:: Set the root folder
set "root_folder=G:\Landsat_river\NDVI_2017_2020"

:: Loop through all year-based folders
for /d %%A in ("%root_folder%\*") do (
    :: Get the year folder name
    set "folder_name=%%~nxA"

    :: Check if the folder name is numeric (indicating a year folder)
    set "year_check=!folder_name!"
    set "year_check=!year_check: =!"

    if "!year_check!"=="!year_check:~0,4!" (
        :: Locate the NDVI_year folder
        set "ndvi_folder=%%A\NDVI_!folder_name!"

        if exist "!ndvi_folder!" (
            echo Processing NDVI folder: !ndvi_folder!

            :: Create a reprojected folder inside the NDVI_year folder
            set "reprojected_folder=!ndvi_folder!\reprojected"
            if not exist "!reprojected_folder!" (
                echo Creating folder: !reprojected_folder!
                mkdir "!reprojected_folder!"
            )

            :: Loop through .tif files in the NDVI_year folder
            for %%f in ("!ndvi_folder!\*.tif") do (
                echo Reprojecting: %%f

                :: Run gdalwarp for each .tif file
                gdalwarp -t_srs EPSG:4326 "%%f" "!reprojected_folder!\%%~nf_gcs.tif"
            )
        )
    )
)

echo All files have been processed.
pause