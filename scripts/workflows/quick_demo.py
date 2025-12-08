import os
import subprocess
import sys
from pathlib import Path

def run_cmd(cmd, description):
    """Chạy command và hiển thị kết quả"""
    print(f"\n{'='*60}")
    print(f"📌 {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}")
    print()
    
    parts = cmd.split()
    if len(parts) > 0 and parts[0] == "python":
        if len(parts) > 1 and parts[1] == "-m":
            new_cmd = [sys.executable, "-m"] + parts[2:]
        else:
            new_cmd = [sys.executable] + parts[1:]
    else:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Thành công!")
            if result.stdout:
                print(result.stdout)
        else:
            print("❌ Lỗi!")
            if result.stderr:
                print(result.stderr)
            return False
        return True
    
    result = subprocess.run(new_cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Thành công!")
        if result.stdout:
            print(result.stdout)
    else:
        print("❌ Lỗi!")
        if result.stderr:
            print(result.stderr)
        return False
    
    return True

def main():
    """Chạy demo pipeline"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║     DEMO PIPELINE: DỰ BÁO ĐỘ TRỄ TÀU HỎA (RSTGCN)           ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    data_dir = Path("data/templates_all")
    has_data = (data_dir / "stop_times.csv").exists()
    
    if not has_data:
        print("\n📥 BƯỚC 1: Thu thập dữ liệu từ runningstatus.in")
        print("   (Sử dụng dữ liệu mẫu - bạn có thể thay đổi URLs)")
        
        sample_urls = "https://runningstatus.in/status/05511-on-20251102,https://runningstatus.in/status/64612"
        date = "2025-11-02"
        
        cmd = (
            "python -m scripts.data_acquisition.scrape_runningstatus "
            f'--urls "{sample_urls}" --date {date} --outdir data/templates_all'
        )
        
        if not run_cmd(cmd, "Scraping dữ liệu"):
            print("\n⚠️  Lỗi khi scrape. Bạn có thể bỏ qua bước này nếu đã có dữ liệu.")
            response = input("Tiếp tục? (y/n): ")
            if response.lower() != 'y':
                return
    else:
        print("\n✅ Đã có dữ liệu, bỏ qua bước scrape")
    
    print("\n📊 BƯỚC 2: Làm giàu dữ liệu (augment delays)")
    cmd = "python -m scripts.preprocessing.augment_delays"
    run_cmd(cmd, "Augment delays")
    
    print("\n🔄 BƯỚC 3: Chuyển đổi dữ liệu sang định dạng tensor")
    cmd = (
        "python -m scripts.preprocessing.convert_to_rstgcn "
        "--stations data/templates_all/stations.csv "
        "--edges data/templates_all/edges.csv "
        "--stops data/templates_all/stop_times_augmented.csv "
        "--slot 50 "
        "--outdir data/processed "
        "--st-id-col station_code "
        "--lat-col lat "
        "--lon-col lon"
    )
    if not run_cmd(cmd, "Convert to RSTGCN format"):
        print("\n❌ Lỗi khi convert. Dừng pipeline.")
        return
    
    print("\n🎓 BƯỚC 4: Huấn luyện mô hình RSTGCN")
    cmd = (
        "python -m scripts.modeling.train_rstgcn "
        "--data data/processed --window 2 --target 4 --epochs 20 "
        "--batch 32 --lr 1e-3 "
        "--outdir runs/rstgcn_demo "
        "--metrics-csv runs/rstgcn_demo/metrics.csv"
    )
    if not run_cmd(cmd, "Training RSTGCN"):
        print("\n⚠️  Lỗi khi training. Bạn có thể bỏ qua và dùng model đã train sẵn.")
        response = input("Tiếp tục với inference? (y/n): ")
        if response.lower() != 'y':
            return
    
    print("\n🔮 BƯỚC 5: Dự đoán trên tập validation")
    ckpt_path = "runs/rstgcn_demo/rstgcn_best.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "runs/rstgcn_headway/rstgcn_best.pt"
        print(f"   Sử dụng checkpoint có sẵn: {ckpt_path}")
    
    cmd = (
        "python -m scripts.modeling.infer_rstgcn "
        f"--data data/processed --ckpt {ckpt_path} "
        "--out-csv runs/rstgcn_demo/val_predictions.csv "
        "--window 2 --target 4"
    )
    run_cmd(cmd, "Inference")
    
    print("\n📈 BƯỚC 6: Tạo biểu đồ đánh giá")
    cmd = (
        "python -m scripts.analysis.plot_eval "
        "--metrics-csv runs/rstgcn_demo/metrics.csv "
        "--pred-csv runs/rstgcn_demo/val_predictions.csv "
        "--out1 runs/rstgcn_demo/mae_curve.png "
        "--out2 runs/rstgcn_demo/station_pred.png"
    )
    run_cmd(cmd, "Plot evaluation")
    
    print("\n📊 BƯỚC 7: Phân tích dataset")
    cmd = (
        "python -m scripts.analysis.analyze_dataset "
        "--data data/processed --outdir runs/dataset_analysis --target 4"
    )
    run_cmd(cmd, "Analyze dataset")
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                    ✅ HOÀN TẤT DEMO!                         ║
    ╚══════════════════════════════════════════════════════════════╝
    
    📁 Kết quả được lưu tại:
       - Model: runs/rstgcn_demo/rstgcn_best.pt
       - Predictions: runs/rstgcn_demo/val_predictions.csv
       - Metrics: runs/rstgcn_demo/metrics.csv
       - Plots: runs/rstgcn_demo/*.png
    
    🚀 Chạy Streamlit app để xem kết quả:
       streamlit run app.py
    """)

if __name__ == "__main__":
    main()

