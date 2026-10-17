"""Build 2 (CPU): unlabeled-track windows for self-supervised pretraining, with the recording date and the site that date
maps to (sites are collection days; dates without labeled data have no known site)."""
import glob, os, shutil, sys, time, collections
f = glob.glob('/kaggle/input/**/preprocess.py', recursive=True)
os.makedirs('/tmp/pkg/sensewhere', exist_ok=True)
for p in glob.glob(os.path.dirname(f[0]) + '/*.py'):
    shutil.copy(p, '/tmp/pkg/sensewhere')
sys.path.insert(0, '/tmp/pkg')
import numpy as np, pandas as pd
from sensewhere import preprocess as pp

OUT = '/kaggle/working'
cfg = pp.DopplerConfig(suppress_bins=2, normalise='none')  # identical to the 'raw' labeled cache
STRIDE_S = 0.5
readme = glob.glob('/kaggle/input/**/DopplerWild/README.md', recursive=True)[0]
root = os.path.dirname(readme)
fs = os.path.join(root, 'fold_splits')
ul = pd.read_csv(os.path.join(fs, 'DopplerWild_unlabeled_tracklist.csv'))
lab = pd.concat([pd.read_csv(os.path.join(fs, n))[['file_name', 'location']] for n in
                 ['motionstate_1s_fold3.csv', 'singlehand_constrained_1s_fold3.csv', 'velocity_regression_1s_folds3.csv']]).drop_duplicates()
lab['date'] = (lab.file_name.astype(np.int64) // 1000000).astype(int)
date2site = lab.groupby('date').location.agg(lambda s: s.mode().iloc[0] if s.nunique() == 1 else None).to_dict()
print('date -> site', date2site, flush=True)
rows, mats, qc = [], [], collections.Counter()
t0 = time.time()
for i, r in ul.iterrows():
    p = os.path.join(root, 'unlabeled_tracks_Doppler', f"{r.file_name}_track_{r.track_id}.npz")
    if not os.path.exists(p):
        qc['missing_track'] += 1
        continue
    tr = np.load(p)['uD']
    date = int(int(r.file_name) // 1000000)
    n_win = int((tr.shape[1] / cfg.fps - cfg.window_s) / STRIDE_S) + 1
    for k in range(max(n_win, 0)):
        w = pp.slice_window(tr, k * STRIDE_S, cfg)
        if w is None:
            qc['out_of_track'] += 1
            continue
        x, why = pp.process_doppler_window(w, cfg)
        qc[why] += 1
        if x is None:
            continue
        mats.append(x.astype(np.float16))
        rows.append(dict(file_name=int(r.file_name), track=str(r.track_id), date=date, site=date2site.get(date), split=r.split, t_start=k * STRIDE_S))
    if i % 300 == 0:
        print(i, len(rows), round(time.time() - t0), 's', flush=True)
X = np.stack(mats)
meta = pd.DataFrame(rows)
np.save(f'{OUT}/ul_windows_raw.npy', X)
meta.to_csv(f'{OUT}/ul_meta.csv', index=False)
print('X', X.shape, dict(qc))
print(meta.site.value_counts(dropna=False).to_string())
print(meta.groupby('date').size().to_string())
