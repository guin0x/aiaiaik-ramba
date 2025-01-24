@echo off
setlocal enabledelayedexpansion

:: Set the root directory
set "root_folder=G:\Landsat_river\2017_to_2020"

:: Loop through all subfolders (recursively)
for /r "%root_folder%" %%A in (.) do (
    :: Extract the folder name
    set "folder_name=%%~nxA"

    :: Check if the folder name contains a date pattern
    echo "!folder_name!" | findstr /r "^[A-Z0-9_]*_[0-9]\{8\}_" >nul
    if not errorlevel 1 (
        :: Extract the date from the folder name (4th part after splitting by '_')
        for /f "tokens=4 delims=_" %%B in ("!folder_name!") do (
            set "date=%%B"
        )

        :: Navigate to the subfolder
        pushd "%%A"

        :: Find and fix NDVI files
        for %%F in (*NDVI*_*_*) do (
            set "file_name=%%~nF"
            set "file_ext=%%~xF"

            :: Check if the filename contains duplicate dates
            echo "!file_name!" | find "_!date!_!date!" >nul
            if not errorlevel 1 (
                :: Remove the duplicate date
                set "new_name=!file_name:_!date!_!date!=_!date!"
                ren "%%F" "!new_name!!file_ext!"
                echo Renamed "%%F" to "!new_name!!file_ext!"
            ) else (
                echo Skipping "%%F" - no duplicate date found.
            )
        )

        :: Return to the previous folder
        popd
    )
)

echo All files have been processed.
pause
pause