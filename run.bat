@echo off
rem Usage: run.bat [command...]   e.g. run.bat python -m src.train
rem data/ and models/ are always mounted; src/ and configs/ are mounted so code edits need no rebuild.
docker run --rm ^
  -v "%cd%\data:/app/data" ^
  -v "%cd%\models:/app/models" ^
  -v "%cd%\src:/app/src" ^
  -v "%cd%\configs:/app/configs" ^
  flood-mapping %*
