@echo off
REM ============================================================================
REM AI Image Auto-Tagger - Automated tagging without browser (TEMPLATE)
REM ============================================================================
REM 
REM Copy this file and configure your settings below:
REM

REM === REQUIRED: Path to your image folder ===
set IMAGE_FOLDER=C:\Path\To\Your\Images

REM === OPTIONAL SETTINGS (uncomment and modify as needed) ===

REM Process subdirectories? (True/False)
REM set RECURSIVE=False

REM Output destination ("Metadata" or "Text File")
REM set OUTPUT_TO=Metadata

REM Sort order ("Newest First", "Oldest First", "Name (A-Z)", "Name (Z-A)", "None")
REM set SORT_ORDER=Newest First

REM General tags threshold (0.0 to 1.0, lower = more tags)
REM set GENERAL_THRESH=0.35

REM Character tags threshold (0.0 to 1.0)
REM set CHARACTER_THRESH=0.85

REM Hide rating tags? (True/False)
REM set HIDE_RATING_TAGS=True

REM Put character tags first? (True/False)
REM set CHARACTER_TAGS_FIRST=False

REM Remove underscores from tags? (True/False)
REM set REMOVE_SEPARATOR=False

REM Overwrite existing metadata tags? (True/False)
REM set OVERWRITE_TAGS=False

REM Skip images that already have tags? (True/False)
REM set SKIP_IF_TAGGED=False

REM ============================================================================

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Build the command with optional parameters
set CMD=python auto_tag.py "%IMAGE_FOLDER%"

if defined RECURSIVE set CMD=%CMD% --recursive %RECURSIVE%
if defined OUTPUT_TO set CMD=%CMD% --output-to "%OUTPUT_TO%"
if defined SORT_ORDER set CMD=%CMD% --sort-order "%SORT_ORDER%"
if defined GENERAL_THRESH set CMD=%CMD% --general-thresh %GENERAL_THRESH%
if defined CHARACTER_THRESH set CMD=%CMD% --character-thresh %CHARACTER_THRESH%
if defined HIDE_RATING_TAGS set CMD=%CMD% --hide-rating-tags %HIDE_RATING_TAGS%
if defined CHARACTER_TAGS_FIRST set CMD=%CMD% --character-tags-first %CHARACTER_TAGS_FIRST%
if defined REMOVE_SEPARATOR set CMD=%CMD% --remove-separator %REMOVE_SEPARATOR%
if defined OVERWRITE_TAGS set CMD=%CMD% --overwrite-tags %OVERWRITE_TAGS%
if defined SKIP_IF_TAGGED set CMD=%CMD% --skip-if-tagged %SKIP_IF_TAGGED%

REM Run the automated tagging script
%CMD%

REM Deactivate virtual environment
call venv\Scripts\deactivate.bat

echo.
pause


