import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scripts.modeling.data_modules import load_processed

def main(a):
    X, A, D, F, meta = load_processed(a.data)
    T, N, Fdim = X.shape
    
    print("=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Time steps: {T}")
    print(f"Stations: {N}")
    print(f"Features: {Fdim}")
    print(f"Time range: {meta['t_start']} to {meta['t_end']}")
    print(f"Slot duration: {meta['slot_minutes']} minutes")
    print(f"Features: {meta['features']}")
    
    print("\n" + "=" * 60)
    print("FEATURE STATISTICS")
    print("=" * 60)
    
    stats = []
    for i, feat_name in enumerate(meta['features']):
        values = X[:, :, i].flatten()
        values = values[~np.isnan(values)]
        
        stats.append({
            'feature': feat_name,
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
            'q25': float(np.percentile(values, 25)),
            'q75': float(np.percentile(values, 75)),
            'missing_pct': float(np.isnan(X[:, :, i]).sum() / X[:, :, i].size * 100)
        })
        
        print(f"\n{feat_name}:")
        print(f"  Mean: {stats[-1]['mean']:.2f}")
        print(f"  Std: {stats[-1]['std']:.2f}")
        print(f"  Min: {stats[-1]['min']:.2f}")
        print(f"  Max: {stats[-1]['max']:.2f}")
        print(f"  Median: {stats[-1]['median']:.2f}")
        print(f"  Missing: {stats[-1]['missing_pct']:.2f}%")
    
    print("\n" + "=" * 60)
    print("GRAPH STATISTICS")
    print("=" * 60)
    num_edges = (A > 0).sum()
    print(f"Number of edges: {num_edges}")
    print(f"Edge density: {num_edges / (N * N) * 100:.2f}%")
    print(f"Average degree: {num_edges / N:.2f}")
    
    dist_values = D[D > 0]
    print(f"\nDistance statistics:")
    print(f"  Mean: {np.mean(dist_values):.2f} km")
    print(f"  Median: {np.median(dist_values):.2f} km")
    print(f"  Max: {np.max(dist_values):.2f} km")
    
    freq_values = F[F > 0]
    print(f"\nFrequency statistics:")
    print(f"  Mean: {np.mean(freq_values):.2f}")
    print(f"  Median: {np.median(freq_values):.2f}")
    print(f"  Max: {np.max(freq_values):.2f}")
    
    os.makedirs(a.outdir, exist_ok=True)
    
    output_stats = {
        'dataset': {
            'T': int(T),
            'N': int(N),
            'F': int(Fdim),
            'time_range': [meta['t_start'], meta['t_end']],
            'slot_minutes': meta['slot_minutes']
        },
        'features': stats,
        'graph': {
            'num_edges': int(num_edges),
            'edge_density': float(num_edges / (N * N) * 100),
            'avg_degree': float(num_edges / N)
        },
        'distance': {
            'mean': float(np.mean(dist_values)),
            'median': float(np.median(dist_values)),
            'max': float(np.max(dist_values))
        },
        'frequency': {
            'mean': float(np.mean(freq_values)),
            'median': float(np.median(freq_values)),
            'max': float(np.max(freq_values))
        }
    }
    
    with open(os.path.join(a.outdir, 'dataset_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(output_stats, f, indent=2)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for i, feat_name in enumerate(meta['features']):
        ax = axes[i // 3, i % 3]
        values = X[:, :, i].flatten()
        values = values[~np.isnan(values)]
        ax.hist(values, bins=50, edgecolor='black', alpha=0.7)
        ax.set_title(f'{feat_name} Distribution')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(a.outdir, 'feature_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[OK] Saved: {os.path.join(a.outdir, 'feature_distributions.png')}")
    
    target_idx = a.target if hasattr(a, 'target') else 0
    target_name = meta['features'][target_idx]
    
    ts_mean = np.nanmean(X[:, :, target_idx], axis=1)
    ts_std = np.nanstd(X[:, :, target_idx], axis=1)
    
    plt.figure(figsize=(12, 6))
    plt.plot(ts_mean, label='Mean', linewidth=2)
    plt.fill_between(range(len(ts_mean)), 
                     ts_mean - ts_std, 
                     ts_mean + ts_std, 
                     alpha=0.3, label='±1 Std')
    plt.title(f'{target_name} Time Series (averaged across stations)')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(a.outdir, f'{target_name}_timeseries.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {os.path.join(a.outdir, f'{target_name}_timeseries.png')}")
    
    print(f"\n[OK] All statistics saved to: {a.outdir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Analyze dataset statistics")
    p.add_argument("--data", default="data/processed", help="Path to processed data directory")
    p.add_argument("--outdir", default="runs/dataset_analysis", help="Output directory")
    p.add_argument("--target", type=int, default=0, help="Target feature index for time series plot")
    main(p.parse_args())

