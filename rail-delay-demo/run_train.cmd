@echo off
setlocal
REM activate venv nếu cần: call .venv\Scripts\activate
python -m scripts.train_rstgcn --data data\processed --window 4 --hidden 32 --epochs 30 --batch 32 --lr 1e-3 --outdir runs\rstgcn_mvp
echo.
echo === Done training ===
pause
