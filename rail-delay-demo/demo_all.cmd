@echo off
rem ====== Activate venv ======
call .\.venv\Scripts\activate.bat

echo === Convert CSV to NPY ===
python scripts\convert_to_stgcn.py --stations data\templates\stations.csv --edges data\templates\edges.csv --stops data\templates\stop_times.csv --slot 60 --outdir data\processed

echo === Baselines ===
python scripts\baseline_demo.py --data data\processed

echo === Plot (DGR) ===
python scripts\plot_baseline.py --data data\processed --station DGR

echo === STGCN ===
python scripts\stgcn_demo.py --data data\processed --window 4 --horizon 1 --epochs 30

echo === RSTGCN ===
python scripts\rstgcn_demo.py --data data\processed --window 4 --horizon 1 --hidden 32 --depth 3 --epochs 40

echo === DONE. Figures in .\figs; results printed above. ===
