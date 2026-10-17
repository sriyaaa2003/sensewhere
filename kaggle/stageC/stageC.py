"""Stage C (GPU): confirmatory run of the hypotheses in docs/CONFIRMATION.md on fresh seeds 3-7.
Arms: erm, erm_floor, erm_aug. BatchNorm-only SHOT on erm and erm_floor. Configs come from the stage-A nested selection."""
import glob, os, shutil, sys, json, time
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
from sensewhere import adapt, data, floor, splits, train
from sensewhere import experiments as ex

OUT = os.environ.get('SW_OUT', '/kaggle/working')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SEEDS, EPOCHS, LEVEL, K = [3, 4, 5, 6, 7], 25, 'load', 3
if os.environ.get('SW_QUICK'):
    SEEDS, EPOCHS = [3], 2
def find(n): return glob.glob(ROOT + '/**/' + n, recursive=True)[0]
meta_all = pd.read_csv(find('dw_meta.csv'))
mo = (meta_all.task == 'motion').to_numpy()
meta = meta_all[mo].reset_index(drop=True)
Xraw = np.load(find('dw_windows_raw.npy'))[mo]
prof = floor.site_profiles(Xraw, meta.location.to_numpy(), 10.0)
sel = pd.read_csv(find('stageA_selection.csv'))
sel = sel[sel.kind == 'select']
GRIDS = {'erm': [dict(lr=1e-3), dict(lr=3e-3)], 'erm_aug': [dict(lr=1e-3), dict(lr=3e-3)], 'erm_floor': [dict(lr=1e-3), dict(lr=3e-3)]}
FIXED = {'erm': dict(), 'erm_aug': dict(augment=True), 'erm_floor': dict(floor_transplant=True)}
def chosen(site, arm):
    r = sel[(sel.site == site) & (sel.arm == arm)]
    assert len(r) == 1, (site, arm)
    return GRIDS[arm][int(r.chosen.iloc[0])]
rows, preds = [], {}
t0 = time.time()
def log_row(**kw):
    rows.append(kw); pd.DataFrame(rows).to_csv(f'{OUT}/stageC_rows.csv', index=False)
def evaluate(site, arm, adapt_name, seed, model, cell):
    pred, lp = train.predict(model, cell.Xte, device=DEV)
    vote, _ = adapt.track_vote(lp, cell.tracks_te)
    for kind, p in (('window', pred), ('vote', vote)):
        log_row(stage='confirm', site=site, arm=arm, adapt=adapt_name, readout=kind, seed=seed, **ex.score(cell.yte, p, K))
        preds[f'v2|{site}|{arm}|{adapt_name}|{kind}|{seed}'] = p.astype(np.int8)
for seed in SEEDS:
    for sp in splits.leave_one_location_out(meta, seed=seed):
        site = sp.target
        cell = ex.make_cell(Xraw, meta, sp, LEVEL)
        preds[f'y|{site}'] = cell.yte.astype(np.int8); preds[f'subject|{site}'] = cell.subj_te
        for arm in ('erm', 'erm_floor', 'erm_aug'):
            m, pred, lg = ex.run_method(cell, LEVEL, 'flat', 'erm', seed, EPOCHS, DEV, input_norm=True, batch_size=128,
                                        profiles=prof if arm == 'erm_floor' else None, **FIXED[arm], **chosen(site, arm))
            evaluate(site, arm, 'none', seed, m, cell)
            if arm in ('erm', 'erm_floor'):
                m2, _ = adapt.shot(m, cell.Xte, params='bn', device=DEV)
                evaluate(site, arm, 'shot_bn', seed, m2, cell)
        print(seed, site, round(time.time() - t0), 's', flush=True)
        np.savez_compressed(f'{OUT}/stageC_preds.npz', **preds)
print('done')
