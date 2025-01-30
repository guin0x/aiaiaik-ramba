@echo off
setlocal enabledelayedexpansion

:: Set the root directory
set "root_folder=G:\Landsat_river\NDVI_2017_2020"

:: Debugging: Print the root folder being used
echo Root folder: %root_folder%

:: Loop through all subfolders (only the year-based folders)
for /d %%A in ("%root_folder%\*") do (
    :: Get the folder name (just the name, not the full path)
    set "folder_name=%%~nxA"

    :: Debugging: Show the folder name being processed
    echo Checking folder: !folder_name!

    :: Check if the folder name is numeric and exactly 4 digits long (i.e., a year)
    set "year_check=!folder_name!"
    set "year_check=!year_check: =!"

    :: Only process folders that are numeric and exactly 4 characters long
    if "!year_check!"=="!year_check:~0,4!" (
        :: If it's a valid year folder, create the NDVI folder
        set "ndvi_folder=%%A\NDVI_!folder_name!"

        :: Check if the NDVI folder already exists
        if not exist "!ndvi_folder!" (
            echo Creating folder: !ndvi_folder!
            mkdir "!ndvi_folder!"
        ) else (
            echo Folder already exists: !ndvi_folder!
        )
    ) else (
        echo Skipping folder: !folder_name! (not a valid year folder)
    )
)

echo All folders have been processed.
pause
