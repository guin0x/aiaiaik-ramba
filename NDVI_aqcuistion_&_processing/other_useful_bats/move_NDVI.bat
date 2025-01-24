@echo off
setlocal enabledelayedexpansion

:: Set the root directory
set "root_folder=G:\Landsat_river\2017_to_2020"

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
        :: If it's a valid year folder, define the target NDVI folder
        set "ndvi_folder=%%A\NDVI_!folder_name!"

        :: Debugging: Show the NDVI folder being used
        echo Target NDVI folder: !ndvi_folder!

        :: Check if the NDVI folder exists, and create it if necessary
        if not exist "!ndvi_folder!" (
            echo Creating folder: !ndvi_folder!
            mkdir "!ndvi_folder!"
        )

        :: Find all .tif files that contain 'NDVI' in their names
        echo Searching for NDVI files in: %%A
        for /r %%B in ("%%A\*NDVI*.tif") do (
            echo Found file: %%B
            if exist "%%B" (
                :: Create a new file name by appending a counter or timestamp to avoid overwriting
                set "file_name=%%~nxB"
                set "new_file_name=!file_name!"

                :: Check if the file already exists in the target folder and append a counter if it does
                set "counter=1"
                :check_duplicate
                if exist "!ndvi_folder!\!new_file_name!" (
                    set /a counter+=1
                    set "new_file_name=!file_name!_!counter!"
                    goto check_duplicate
                )

                :: Log before moving the file
                echo Preparing to move: %%B to !ndvi_folder!\!new_file_name!

                :: Move the file to the target NDVI folder
                move /Y "%%B" "!ndvi_folder!\!new_file_name!"

                :: Confirm if the file has been moved successfully
                if exist "!ndvi_folder!\!new_file_name!" (
                    echo Successfully moved: %%B
                ) else (
                    echo Failed to move: %%B
                )
            ) else (
                echo File not found: %%B
            )
        )
    ) else (
        echo Skipping folder: !folder_name! (not a valid year folder)
    )
)

echo All files have been processed.
pause


