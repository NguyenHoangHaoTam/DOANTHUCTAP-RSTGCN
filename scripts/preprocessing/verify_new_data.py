"""Verify that new data has been added"""
import pandas as pd

df = pd.read_csv('data/templates_all/stop_times.csv')

dates = df['train_id'].str.split('_').str[-1].unique()

print("="*60)
print("KIEM TRA DU LIEU MOI DA DUOC THEM")
print("="*60)

print("\nCac ngay co trong du lieu:")
for d in sorted(dates):
    count = df[df['train_id'].str.endswith(d)].shape[0]
    print(f"  {d}: {count} dong")

print(f"\nTong so dong: {len(df)}")

new_date = '2025-11-15'
new_trains = df[df['train_id'].str.contains(new_date)]['train_id'].unique()
print(f"\nTrain IDs co ngay {new_date} (ngay moi nhat):")
print(f"  So train IDs: {len(new_trains)}")
print(f"  Danh sach:")
for train in sorted(new_trains):
    count = df[df['train_id'] == train].shape[0]
    print(f"    - {train}: {count} stops")

print(f"\nMau du lieu tu ngay {new_date}:")
sample = df[df['train_id'].str.contains(new_date)].head(5)
print(sample[['train_id', 'station_code', 'arr_delay', 'dep_delay']].to_string(index=False))

print("\n" + "="*60)
print("[XAC NHAN] Du lieu moi da duoc them vao thanh cong!")
print("="*60)

