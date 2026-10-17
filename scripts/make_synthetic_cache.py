"""Write a tiny synthetic stand-in for the DopplerWild cache so the Kaggle stage scripts can be smoke-tested locally.

Synthetic data is used ONLY to exercise code paths (shapes, file formats, no crashes). No result in this repository comes from it.
Usage: python scripts/make_synthetic_cache.py OUT_DIR
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sensewhere import data  # noqa: E402

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
F, T = 64, 90
atomic = data.base_hierarchy().leaves
rows, X = [], []
subj = 0
for si, site in enumerate("ABCD"):
    for tr in range(30):  # 30 tracks per site, one subject each
        subj += 1
        a = atomic[tr % len(atomic)]
        for w in range(6):
            x = rng.normal(15 + si, 2.0, (F, T)).astype(np.float32)
            cls = data.LOAD_CLASSES.index(data.ATOMIC_TO_LOAD[a])
            x[20 + 8 * cls : 26 + 8 * cls, :] += 12
            X.append(x.astype(np.float16))
            rows.append(dict(task="motion", subject=subj, location=site, file_name=f"2025052{si}1000{tr:02d}", t_start=0.25 * w,
                             label_atomic=a, label_load=data.ATOMIC_TO_LOAD[a], fold_cs=tr % 3, occlusion_s=0.0, neighbours=0.0))
np.save(out / "dw_windows_raw.npy", np.stack(X))
pd.DataFrame(rows).to_csv(out / "dw_meta.csv", index=False)
ul = []
Xu = []
for k in range(120):
    site = "ABCD"[k % 4] if k % 5 else None  # a fifth of the recordings have no known site
    x = rng.normal(15, 2.0, (F, T)).astype(np.float32)
    x[20:26] += 12
    Xu.append(x.astype(np.float16))
    ul.append(dict(file_name=2025052000000 + k // 2, track=str(k // 2), date=20250520, site=site, split="train", t_start=0.0))
np.save(out / "ul_windows_raw.npy", np.stack(Xu))
pd.DataFrame(ul).to_csv(out / "ul_meta.csv", index=False)
print("wrote", out, len(rows), "labeled windows,", len(ul), "unlabeled windows")
