"""Kaggle script: build cached DopplerWild windows for several preprocessing variants, with a QC report."""
import glob, json, os, sys, time, collections
import numpy as np, pandas as pd

import shutil
_f = glob.glob('/kaggle/input/**/preprocess.py', recursive=True)
os.makedirs('/tmp/pkg/sensewhere', exist_ok=True)
for _p in glob.glob(os.path.dirname(_f[0]) + '/*.py'):
    shutil.copy(_p, '/tmp/pkg/sensewhere')
sys.path.insert(0, '/tmp/pkg')
from sensewhere import preprocess as pp

OUT = '/kaggle/working'
VARIANTS = {
    'base': dict(suppress_bins=2, normalise='window'),
    'nosup': dict(suppress_bins=0, normalise='window'),
    'robust': dict(suppress_bins=2, normalise='robust'),
    'raw': dict(suppress_bins=2, normalise='none'),
}
cfgs = {k: pp.DopplerConfig(**v) for k, v in VARIANTS.items()}
cfg0 = cfgs['base']

readme = glob.glob('/kaggle/input/**/DopplerWild/README.md', recursive=True)[0]
root = os.path.dirname(readme)
trk_dir = os.path.join(root, 'labeled_tracks_Doppler')
tasks = {'motion': 'motionstate_1s_fold3.csv', 'single': 'singlehand_constrained_1s_fold3.csv'}

rows = []
mats = {k: [] for k in cfgs}
qc = collections.Counter()
t0 = time.time()
for task, fn in tasks.items():
    df = pd.read_csv(os.path.join(root, 'fold_splits', fn))
    for (fname, gid), g in df.groupby(['file_name', 'global_id']):
        p = os.path.join(trk_dir, f'{fname}_track_{gid}.npz')
        if not os.path.exists(p):
            qc[(task, 'all', 'missing_track')] += len(g)
            continue
        track = np.load(p)['uD']
        for _, r in g.iterrows():
            loc = r['location']
            if r['occlusion_seconds'] > cfg0.max_occlusion_s:
                qc[(task, loc, 'occluded')] += 1
                continue
            if r['duration'] < 0.99:
                qc[(task, loc, 'short_window')] += 1
                continue
            # t_start_rel is measured from the start of the track file (t_end_rel fits inside the track for 100% of rows;
            # t_start, measured from the recording, does not)
            w = pp.slice_window(track, float(r['t_start_rel']), cfg0)
            if w is None:
                qc[(task, loc, 'out_of_track')] += 1
                continue
            outs = {k: pp.process_doppler_window(w, c) for k, c in cfgs.items()}
            reasons = {why for _, why in outs.values()}
            if reasons != {'ok'}:
                qc[(task, loc, '|'.join(sorted(reasons)))] += 1
                continue
            qc[(task, loc, 'ok')] += 1
            for k, (x, _) in outs.items():
                mats[k].append(x.astype(np.float16))
            rows.append(dict(task=task, subject=int(gid), location=loc, file_name=str(fname), t_start=float(r['t_start_rel']),
                             label_atomic=r['activity_atomic'], label_load=r['activity_load'], fold_cs=int(r['fold']),
                             occlusion_s=float(r['occlusion_seconds']), neighbours=float(r['close_person_avg_count_5m'])))
    print(task, 'done', len(rows), round(time.time() - t0), 's', flush=True)

meta = pd.DataFrame(rows)
for k in cfgs:
    X = np.stack(mats[k])
    assert len(X) == len(meta)
    np.save(f'{OUT}/dw_windows_{k}.npy', X)
meta.to_csv(f'{OUT}/dw_meta.csv', index=False)
qcs = pd.Series({'|'.join(map(str, k)): v for k, v in qc.items()}).rename('count')
qcs.to_csv(f'{OUT}/dw_qc_report.csv')
json.dump({k: v.__dict__ for k, v in cfgs.items()}, open(f'{OUT}/dw_config_used.json', 'w'), indent=1)
print('X', X.shape, X.dtype, 'meta', meta.shape)
print(meta.groupby(['task', 'location', 'label_load']).size().to_string())
print(qcs.to_string())

tot = sum(v for k, v in qc.items())
oot = sum(v for k, v in qc.items() if k[-1] == 'out_of_track')
assert oot / tot < 0.01, f'{oot}/{tot} windows outside their track: time indexing is wrong'
