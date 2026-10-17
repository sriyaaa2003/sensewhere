"""Stage 2 (GPU): domain generalisation, source-free adaptation and the preprocessing ablation, DopplerWild leave-one-location-out."""
import glob, os, shutil, sys, json, time
f = glob.glob('/kaggle/input/**/preprocess.py', recursive=True)
os.makedirs('/tmp/pkg/sensewhere', exist_ok=True)
for p in glob.glob(os.path.dirname(f[0]) + '/*.py'):
    shutil.copy(p, '/tmp/pkg/sensewhere')
sys.path.insert(0, '/tmp/pkg')
import numpy as np, pandas as pd, torch
from sensewhere import data, splits, train
from sensewhere import experiments as ex

OUT = '/kaggle/working'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SEEDS = [0, 1, 2]
EPOCHS = 25
LEVEL = 'load'
K = len(ex.LEVELS[LEVEL])
print('device', DEV, flush=True)
def find(n): return glob.glob('/kaggle/input/**/' + n, recursive=True)[0]
meta_all = pd.read_csv(find('dw_meta.csv'))
data.check_hierarchy_consistency(meta_all)
mo_idx = np.flatnonzero((meta_all.task == 'motion').to_numpy())
meta = meta_all.iloc[mo_idx].reset_index(drop=True)
def load_variant(v):
    X = np.load(find(f'dw_windows_{v}.npy'))
    return X[mo_idx]
XV = {v: load_variant(v) for v in ['base', 'nosup', 'robust', 'raw']}
GRID = {
    'erm':      [dict(lr=1e-3), dict(lr=3e-3)],
    'erm_aug':  [dict(lr=1e-3), dict(lr=3e-3)],
    'coral':    [dict(method_weight=0.1), dict(method_weight=1.0)],
    'dann':     [dict(method_weight=0.1), dict(method_weight=1.0)],
    'groupdro': [dict(dro_step=0.01), dict(dro_step=0.1)],
}
rows, preds, traces = [], {}, {}
t0 = time.time()
def log_row(**kw):
    rows.append(kw); pd.DataFrame(rows).to_csv(f'{OUT}/stage2_rows.csv', index=False)

for seed in SEEDS:
    for sp in splits.leave_one_location_out(meta, seed=seed):
        cell = ex.make_cell(XV['base'], meta, sp, LEVEL)
        site = sp.target
        preds[f'y|{site}'] = cell.yte.astype(np.int8)
        preds[f'subject|{site}'] = cell.subj_te
        sel_models = {}
        for method, grid in GRID.items():
            best = None
            for gi, kw in enumerate(grid):
                m, pred, lg = ex.run_method(cell, LEVEL, 'flat', method, seed, EPOCHS, DEV, batch_size=128, legacy_aug=True, **kw)
                val = max(r['val_bal_acc'] for r in lg)
                s = ex.score(cell.yte, pred, K)
                log_row(stage='dg', site=site, method=method, adapt='none', seed=seed, cfg=json.dumps(kw), val_bal_acc=val, epochs_run=len(lg), selected=False, collapse=ex.collapse_share(pred, K), **s)
                if best is None or val > best[0]:
                    best = (val, gi, m, pred)
            val, gi, m, pred = best
            rows[-(len(grid) - gi)]['selected'] = True  # flag chosen config by SOURCE validation only
            preds[f'dg|{site}|{method}|none|{seed}'] = pred.astype(np.int8)
            sel_models[method] = m
            print(seed, site, method, 'sel', gi, round(time.time() - t0), 's', flush=True)
        # source-free adaptation on the selected ERM and ERM+aug models, unlabeled target windows only
        for base_name in ['erm', 'erm_aug']:
            res = ex.adapt_rows(sel_models[base_name], cell, LEVEL, DEV)
            for name, (p, tr) in res.items():
                s = ex.score(cell.yte, p, K)
                log_row(stage='sfda', site=site, method=base_name, adapt=name, seed=seed, cfg='', val_bal_acc=np.nan, epochs_run=0, selected=True, collapse=ex.collapse_share(p, K), ent_first=float(tr[0]), ent_last=float(tr[-1]), **s)
                preds[f'sfda|{site}|{base_name}|{name}|{seed}'] = p.astype(np.int8)
        print(seed, site, 'adapted', round(time.time() - t0), 's', flush=True)
        # preprocessing ablation: ERM (lr 1e-3) on each variant
        for v in ['nosup', 'robust', 'raw']:
            c2 = ex.make_cell(XV[v], meta, sp, LEVEL)
            m, pred, lg = ex.run_method(c2, LEVEL, 'flat', 'erm', seed, EPOCHS, DEV, batch_size=128, lr=1e-3, legacy_aug=True)
            log_row(stage='prep', site=site, method='erm', adapt='none', seed=seed, cfg=v, val_bal_acc=max(r['val_bal_acc'] for r in lg), epochs_run=len(lg), selected=True, collapse=ex.collapse_share(pred, K), **ex.score(c2.yte, pred, K))
            preds[f'prep|{site}|{v}|{seed}'] = pred.astype(np.int8)
        np.savez_compressed(f'{OUT}/stage2_preds.npz', **preds)
    print('seed done', seed, round(time.time() - t0), 's', flush=True)
print('done', round(time.time() - t0), 's')
