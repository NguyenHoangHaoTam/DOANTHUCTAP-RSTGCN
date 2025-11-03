# 1) scrape nhiều chuyến (ví dụ)
python -m scripts.bulk_scrape --urls "https://runningstatus.in/status/05511-on-20251102,https://runningstatus.in/status/64612" --date 2025-11-02 --outdir data/templates_all

# 2) convert -> npy
python -m scripts.convert_to_rstgcn --stations data/templates_all/stations.csv --edges data/templates_all/edges.csv --stops data/templates_all/stop_times.csv --slot 15 --outdir data/processed --st-id-col station_code --lat-col lat --lon-col lon

# 3) train (ví dụ dự báo headway: target=4)
python -m scripts.train_rstgcn --data data/processed --window 2 --target 4 --epochs 30 --batch 32 --lr 1e-3 --outdir runs/rstgcn_headway --metrics-csv runs/rstgcn_headway/metrics.csv

# 4) infer (xuất CSV dự báo trên tập val)
python -m scripts.infer_rstgcn --data data/processed --ckpt runs/rstgcn_headway/rstgcn_best.pt --out-csv runs/rstgcn_headway/val_predictions.csv --window 2 --target 4

# 5) plot
python -m scripts.plot_eval --metrics-csv runs/rstgcn_headway/metrics.csv --pred-csv runs/rstgcn_headway/val_predictions.csv --out1 runs/rstgcn_headway/mae_curve.png --out2 runs/rstgcn_headway/station_pred.png
