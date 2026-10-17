"""Stage 6 (GPU): why does SHOT-style information maximisation hurt? Test the diversity term under label shift.
For each site/seed: train ERM, then adapt on (a) the natural target windows and (b) a class-balanced subsample of them,
with diversity weight in {0, 0.3, 1}. Always evaluate balanced accuracy on ALL target windows."""
import glob, os, shutil, sys, json, time
f = glob.glob('/kaggle/input/**/preprocess.py', recursive=True)
os.makedirs('/tmp/pkg/sensewhere', exist_ok=True)
for p in glob.glob(os.path.dirname(f[0]) + '/*.py'):
    shutil.copy(p, '/tmp/pkg/sensewhere')
sys.path.insert(0, '/tmp/pkg')
import numpy as np, pandas as pd, torch
from sensewhere import adapt, data, splits, train
from sensewhere import experiments as ex

OUT = '/kaggle/working'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SEEDS, EPOCHS, K = [0, 1, 2], 25, 3
def find(n): return glob.glob('/kaggle/input/**/' + n, recursive=True)[0]
meta_all = pd.read_csv(find('dw_meta.csv'))
mo = (meta_all.task == 'motion').to_numpy()
meta = meta_all[mo].reset_index(drop=True)
X = np.load(find('dw_windows_base.npy'))[mo]
rows, preds = [], {}
t0 = time.time()
for seed in SEEDS:
    for sp in splits.leave_one_location_out(meta, seed=seed):
        site = sp.target
        cell = ex.make_cell(X, meta, sp, 'load')
        m, pred0, _ = ex.run_method(cell, 'load', 'flat', 'erm', seed, EPOCHS, DEV, batch_size=128, lr=1e-3)
        prior = np.bincount(cell.yte, minlength=K) / len(cell.yte)
        rng = np.random.default_rng(seed)
        per = min(np.bincount(cell.yte, minlength=K))
        bal_idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(cell.yte == c), per, replace=False) for c in range(K)]))
        sets = {'natural': np.arange(len(cell.yte)), 'balanced': bal_idx}
        base = ex.score(cell.yte, pred0, K)['bal_acc']
        rows.append(dict(site=site, seed=seed, adapt_set='-', method='none', div=np.nan, bal_acc=base, max_prior=float(prior.max())))
        for sname, idx in sets.items():
            Xa = cell.Xte[idx]
            batches = [Xa[i:i + 128] for i in range(0, len(Xa), 128)]
            for name, fn in [('adabn', lambda: adapt.adabn(m, batches, DEV)), ('tent_e1', lambda: adapt.tent(m, batches, epochs=1, device=DEV))] + \
                            [(f'shot_div{d}', (lambda d=d: adapt.shot_im(m, batches, epochs=3, div_weight=d, device=DEV))) for d in (0.0, 0.3, 1.0)]:
                m2, tr = fn()
                p, _ = train.predict(m2, cell.Xte, device=DEV)
                rows.append(dict(site=site, seed=seed, adapt_set=sname, method=name, div=float(name.split('div')[1]) if 'div' in name else np.nan,
                                 bal_acc=ex.score(cell.yte, p, K)['bal_acc'], max_prior=float(prior.max()), collapse=ex.collapse_share(p, K)))
                preds[f'{site}|{sname}|{name}|{seed}'] = p.astype(np.int8)
        pd.DataFrame(rows).to_csv(f'{OUT}/stage6_rows.csv', index=False)
        print(seed, site, round(time.time() - t0), 's', flush=True)
print('done')
