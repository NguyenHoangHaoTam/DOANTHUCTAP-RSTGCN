# 🚆 Rail Delay Prediction using RSTGCN

Dự án triển khai mô hình **RSTGCN** (Railway-centric Spatio-Temporal Graph Convolutional Network) để dự báo độ trễ tàu hỏa dựa trên bài báo gốc: [RSTGCN paper](https://arxiv.org/pdf/2510.01262).

## ✨ Điểm nhấn
- Thu thập dữ liệu tự động từ [runningstatus.in](https://runningstatus.in/) với cơ chế retry + rate limit.
- Pipeline xử lý dữ liệu đầy đủ: augment độ trễ, geocode, chuyển đổi tensor.
- Huấn luyện/inference mô hình RSTGCN + so sánh baseline (LSTM, GCN, mean).
- Bộ script phân tích kết quả và visualize bằng Streamlit app (bản đồ + biểu đồ tương tác).

## 🧱 Cấu trúc dự án

```
rail-delay-demo/
├── app.py                 
├── data/                  
│   
├── models/
│   └── rstgcn.py
├── scripts/
│   ├── analysis/           
│   ├── data_acquisition/   
│   ├── modeling/           
│   ├── preprocessing/     
│   └── workflows/          
├── requirements.txt
└── README.md              
```

### Nhóm script theo chức năng

| Nhóm | Mục đích | Entrypoint chính |
| --- | --- | --- |
| `data_acquisition/` | Lấy dữ liệu thô | `python -m scripts.data_acquisition.update_data`<br>`python -m scripts.data_acquisition.scrape_runningstatus` |
| `preprocessing/` | Làm sạch, augment, convert tensor | `augment_delays`, `geocode_stations`, `convert_to_rstgcn`, `check_data`, `merge_templates`, `verify_new_data` |
| `modeling/` | Dataset loader, train/infer, baseline | `train_rstgcn`, `infer_rstgcn`, `baseline_comparison`, `data_modules` |
| `analysis/` | Đánh giá & trực quan hóa | `analyze_dataset`, `analyze_results`, `plot_eval`, `plot_delay_distribution` |
| `workflows/` | Orchestrate nhiều bước | `python -m scripts.workflows.quick_demo` |

## ⚙️ Yêu cầu môi trường

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/Mac
pip install -r requirements.txt
```

Dependencies chính: `torch`, `pandas`, `requests`, `beautifulsoup4`, `geopy`, `streamlit`, `plotly`, `folium`.


### 2. Pipeline thủ công (chia theo chức năng)

#### A. Thu thập dữ liệu

```bash
# Lấy nhiều URL từ file (có thể dùng comment #)
python -m scripts.data_acquisition.update_data --urls-file urls.txt --auto-date

# Hoặc nhập trực tiếp list URL
python -m scripts.data_acquisition.update_data --interactive --date 2025-11-02

# Scrape nhanh nếu chỉ có vài link
python -m scripts.data_acquisition.scrape_runningstatus \
  --urls "https://runningstatus.in/status/05511-on-20251102,https://runningstatus.in/status/64612" \
  --date 2025-11-02 --outdir data/templates_all
```

#### B. Tiền xử lý / chuẩn hóa

```bash
# Tạo stop_times_augmented.csv từ dữ liệu mới
python -m scripts.preprocessing.augment_delays

# Bổ sung toạ độ chính xác (khuyến nghị)
python -m scripts.preprocessing.geocode_stations \
  --input-csv data/templates_all/stations.csv \
  --output-csv data/templates_all/stations.csv

# Chuyển sang tensor để train
python -m scripts.preprocessing.convert_to_rstgcn \
  --stations data/templates_all/stations.csv \
  --edges data/templates_all/edges.csv \
  --stops data/templates_all/stop_times_augmented.csv \
  --slot 50 \
  --outdir data/processed \
  --st-id-col station_code --lat-col lat --lon-col lon
```

#### C. Huấn luyện & suy luận

```bash
# Train model
python -m scripts.modeling.train_rstgcn \
  --data data/processed \
  --window 2 --target 4 --epochs 30 --batch 32 --lr 1e-3 \
  --outdir runs/rstgcn_headway \
  --metrics-csv runs/rstgcn_headway/metrics.csv

# Inference / export dự báo
python -m scripts.modeling.infer_rstgcn \
  --data data/processed \
  --ckpt runs/rstgcn_headway/rstgcn_best.pt \
  --out-csv runs/rstgcn_headway/val_predictions.csv \
  --window 2 --target 4
```

#### D. Đánh giá & phân tích

```bash
# Vẽ biểu đồ MAE & so sánh theo ga
python -m scripts.analysis.plot_eval \
  --metrics-csv runs/rstgcn_headway/metrics.csv \
  --pred-csv runs/rstgcn_headway/val_predictions.csv \
  --out1 runs/rstgcn_headway/mae_curve.png \
  --out2 runs/rstgcn_headway/station_pred.png

# Khai thác thống kê dataset
python -m scripts.analysis.analyze_dataset \
  --data data/processed \
  --outdir runs/dataset_analysis \
  --target 4

# So sánh baseline
python -m scripts.modeling.baseline_comparison \
  --data data/processed \
  --rstgcn-ckpt runs/rstgcn_headway/rstgcn_best.pt \
  --outdir runs/baseline_comparison

# Đào sâu kết quả dự báo
python -m scripts.analysis.analyze_results \
  --pred-csv runs/rstgcn_headway/val_predictions.csv \
  --outdir runs/results_analysis
```

#### E. Làm mới dữ liệu cho app

Sau khi hoàn tất bước preprocessing, restart Streamlit hoặc dùng nút **“🔄 Refresh dữ liệu (Clear cache)”** trong sidebar. Nếu train lại model, cập nhật `runs/<tên_run>/rstgcn_best.pt` và chạy lại inference trước khi mở app.

## 📥 Chi tiết update_data

`python -m scripts.data_acquisition.update_data` cung cấp 3 cách nhập URL (file, CLI option, interactive). Các tính năng chính:

- Merge thông minh với dữ liệu cũ, tránh trùng lặp.
- Tự bắt ngày từ pattern `on-YYYYMMDD`.
- Cho phép comment trong file `urls.txt`.

Ví dụ file `urls.txt`:

```
# Tuyến Delhi → Dehradun
https://runningstatus.in/status/05511-on-20251102
https://runningstatus.in/status/64612
```

## 🎨 Streamlit dashboard

```bash
streamlit run app.py
```

Các tab chính: Overview (MAE/RMSE/tổng số ga trễ), Predictions Chart, Map View (Folium + Plotly), Data Table, Detailed Analysis, Custom Forecast.


