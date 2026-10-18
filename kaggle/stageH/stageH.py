"""Stage G/H (GPU): base model with probabilities saved, for offline source-free recalibration, ensembles and conformal sets.
SEEDS is set by the `SEEDS = ...` line below (stage G: 0-2, exploratory design; stage H: 10-14, confirmatory)."""
import glob, os, shutil, sys, time
ROOT = os.environ.get('SW_INPUT', '/kaggle/input')
if os.environ.get('SW_PKG'):
    sys.path.insert(0, os.environ['SW_PKG'])
else:
    f = glob.glob(ROOT + '/**/preprocess.py', recursive=True)
    os.makedirs('/tmp/pkg/sensewhere', exist_ok=True)
    for p in glob.glob(os.path.dirname(f[0]) + '/*.py'):
        shutil.copy(p, '/tmp/pkg/sensewhere')
    sys.path.insert(0, '/tmp/pkg')
import numpy as np, pandas as pd, torch
from sensewhere import data, splits, train
from sensewhere import experiments as ex

OUT = os.environ.get('SW_OUT', '/kaggle/working')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SEEDS = [10, 11, 12, 13, 14]
TAG = 'H'
EPOCHS, LEVEL, K = 25, 'load', 3
if os.environ.get('SW_QUICK'):
    SEEDS, EPOCHS = [0], 2
def find(n): return glob.glob(ROOT + '/**/' + n, recursive=True)[0]
meta_all = pd.read_csv(find('dw_meta.csv'))
data.check_hierarchy_consistency(meta_all)
mo = (meta_all.task == 'motion').to_numpy()
meta = meta_all[mo].reset_index(drop=True)
Xraw = np.load(find('dw_windows_raw.npy'))[mo].astype(np.float32)
out, t0 = {}, time.time()
for seed in SEEDS:
    for sp in splits.leave_one_location_out(meta, seed=seed):
        site = sp.target
        cell = ex.make_cell(Xraw, meta, sp, LEVEL)
        m, pred, lg = ex.run_method(cell, LEVEL, 'flat', 'erm', seed, EPOCHS, DEV, input_norm=True, batch_size=128, lr=1e-3)
        _, lp_te = train.predict(m, cell.Xte, device=DEV)
        _, lp_va = train.predict(m, cell.Xva, device=DEV)
        keys = (meta.file_name.astype(str) + '|' + meta.subject.astype(str)).to_numpy()
        out[f'lp_te|{site}|{seed}'] = lp_te.astype(np.float16)
        out[f'lp_va|{site}|{seed}'] = lp_va.astype(np.float16)
        out[f'y_va|{site}|{seed}'] = cell.yva.astype(np.int8)
        out[f'track_va|{site}|{seed}'] = keys[sp.val_idx]
        out[f'y|{site}'] = cell.yte.astype(np.int8)
        out[f'subject|{site}'] = cell.subj_te
        out[f'track|{site}'] = cell.tracks_te
        print(TAG, seed, site, round(time.time() - t0), 's', flush=True)
        np.savez_compressed(f'{OUT}/stage{TAG}_probs.npz', **out)
print('done')
