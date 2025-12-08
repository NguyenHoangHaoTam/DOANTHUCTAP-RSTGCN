import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

def main(a):
    df = pd.read_csv(a.pred_csv)
    
    print("=" * 60)
    print("PREDICTION RESULTS ANALYSIS")
    print("=" * 60)
    
    mae = mean_absolute_error(df['y_true'], df['y_pred'])
    rmse = np.sqrt(mean_squared_error(df['y_true'], df['y_pred']))
    mape = np.mean(np.abs((df['y_true'] - df['y_pred']) / (df['y_true'] + 1e-6))) * 100
    
    print(f"\nOverall Metrics:")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    
    station_metrics = []
    for station in df['station_code'].unique():
        df_st = df[df['station_code'] == station]
        if len(df_st) == 0:
            continue
        
        st_mae = mean_absolute_error(df_st['y_true'], df_st['y_pred'])
        st_rmse = np.sqrt(mean_squared_error(df_st['y_true'], df_st['y_pred']))
        st_mape = np.mean(np.abs((df_st['y_true'] - df_st['y_pred']) / (df_st['y_true'] + 1e-6))) * 100
        
        station_metrics.append({
            'station_code': station,
            'n_samples': len(df_st),
            'mae': st_mae,
            'rmse': st_rmse,
            'mape': st_mape,
            'mean_true': df_st['y_true'].mean(),
            'mean_pred': df_st['y_pred'].mean()
        })
    
    df_station_metrics = pd.DataFrame(station_metrics)
    df_station_metrics = df_station_metrics.sort_values('mae', ascending=False)
    
    print(f"\nPer-Station Statistics:")
    print(f"  Best station (lowest MAE): {df_station_metrics.iloc[-1]['station_code']} (MAE={df_station_metrics.iloc[-1]['mae']:.2f})")
    print(f"  Worst station (highest MAE): {df_station_metrics.iloc[0]['station_code']} (MAE={df_station_metrics.iloc[0]['mae']:.2f})")
    print(f"  Average MAE per station: {df_station_metrics['mae'].mean():.2f}")
    print(f"  Std MAE per station: {df_station_metrics['mae'].std():.2f}")
    
    df['error'] = df['y_pred'] - df['y_true']
    df['abs_error'] = np.abs(df['error'])
    
    print(f"\nError Statistics:")
    print(f"  Mean error: {df['error'].mean():.4f}")
    print(f"  Std error: {df['error'].std():.4f}")
    print(f"  Mean absolute error: {df['abs_error'].mean():.4f}")
    print(f"  Median absolute error: {df['abs_error'].median():.4f}")
    
    os.makedirs(a.outdir, exist_ok=True)
    df_station_metrics.to_csv(os.path.join(a.outdir, 'per_station_metrics.csv'), index=False)
    print(f"\n[OK] Saved per-station metrics: {os.path.join(a.outdir, 'per_station_metrics.csv')}")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    ax.scatter(df['y_true'], df['y_pred'], alpha=0.5, s=10)
    min_val = min(df['y_true'].min(), df['y_pred'].min())
    max_val = max(df['y_true'].max(), df['y_pred'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    ax.set_xlabel('True Value')
    ax.set_ylabel('Predicted Value')
    ax.set_title('True vs Predicted')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.hist(df['error'], bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero error')
    ax.set_xlabel('Error (Predicted - True)')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.hist(df['abs_error'], bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Absolute Error')
    ax.set_ylabel('Frequency')
    ax.set_title('Absolute Error Distribution')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    top_n = min(20, len(df_station_metrics))
    top_stations = df_station_metrics.head(top_n)
    ax.barh(range(len(top_stations)), top_stations['mae'])
    ax.set_yticks(range(len(top_stations)))
    ax.set_yticklabels(top_stations['station_code'], fontsize=8)
    ax.set_xlabel('MAE')
    ax.set_title(f'Top {top_n} Stations by MAE (worst)')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(a.outdir, 'results_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {os.path.join(a.outdir, 'results_analysis.png')}")
    
    print(f"\n[OK] All analysis saved to: {a.outdir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Analyze prediction results")
    p.add_argument("--pred-csv", required=True, help="Path to predictions CSV file")
    p.add_argument("--outdir", default="runs/results_analysis", help="Output directory")
    main(p.parse_args())

