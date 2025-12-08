import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

src = r"data\templates_all\stop_times.csv"
df = pd.read_csv(src)
print("Loaded:", df.shape)

heavy = ["05511", "12556"]
medium = ["66324", "06556"]
light = []

def random_delay(train_id):
    base = train_id.split("_")[0] if isinstance(train_id, str) else ""
    r = np.random.rand()
    if base in heavy:
        if r < 0.3: return 0
        elif r < 0.7: return np.random.randint(10, 40)
        else: return np.random.randint(40, 100)
    elif base in medium:
        if r < 0.5: return 0
        elif r < 0.9: return np.random.randint(5, 25)
        else: return np.random.randint(25, 60)
    else:
        if r < 0.7: return 0
        elif r < 0.95: return np.random.randint(5, 15)
        else: return np.random.randint(15, 40)

def add_delay(row):
    fmt = "%Y-%m-%d %H:%M"
    tid = row.get("train_id", "")
    for tcol, acol, dcol in [
        ("arr_sched", "arr_actual", "arr_delay"),
        ("dep_sched", "dep_actual", "dep_delay")
    ]:
        sched = row.get(tcol)
        if isinstance(sched, str) and len(sched.strip()) > 0:
            try:
                sched_dt = datetime.strptime(sched, fmt)
                delay = random_delay(tid)
                actual_dt = sched_dt + timedelta(minutes=delay)
                row[acol] = actual_dt.strftime(fmt)
                row[dcol] = float(delay)
            except Exception:
                pass
    return row

df2 = df.apply(add_delay, axis=1)

os.makedirs("data/templates_all", exist_ok=True)
out_path = r"data\templates_all\stop_times_augmented.csv"
df2.to_csv(out_path, index=False)
print("[OK] Saved:", out_path, "| Shape:", df2.shape)
print(df2.head(10)[["train_id","station_code","arr_sched","arr_actual","arr_delay","dep_sched","dep_actual","dep_delay"]])
