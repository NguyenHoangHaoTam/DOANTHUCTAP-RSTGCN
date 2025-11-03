@echo off
setlocal
REM activate venv nếu cần: call .venv\Scripts\activate
python scripts\convert_to_rstgcn.py --stations data\templates\stations.csv --edges data\templates\edges.csv --stops data\templates\stop_times.csv --slot 60 --outdir data\processed
echo.
echo === Done preprocessing ===
pause
