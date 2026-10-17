"""Stage D (GPU): exploratory sweep, v3 ideas on per-window models (docs/V3_PLAN.md). Seeds 0-2, four held-out sites.

Fixed for every arm so arms are comparable: plain ERM, lr 1e-3, 25 epochs, early stopping on source validation, windows
standardised inside the network. Readouts: single window and track vote. Representation (R), head (H), multitask (M)
and backbone (B) variants, each changing one thing relative to `base`.
"""
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
from sensewhere import adapt, data, reprs, splits, train
from sensewhere import experiments as ex

OUT = os.environ.get('SW_OUT', '/kaggle/working')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SEEDS, EPOCHS, LEVEL, K = [0, 1, 2], 25, 'load', 3
if os.environ.get('SW_QUICK'):
    SEEDS, EPOCHS = [0], 2
def find(n): return glob.glob(ROOT + '/**/' + n, recursive=True)[0]
meta_all = pd.read_csv(find('dw_meta.csv'))
data.check_hierarchy_consistency(meta_all)
mo = (meta_all.task == 'motion').to_numpy()
meta = meta_all[mo].reset_index(drop=True)
Xraw = np.load(find('dw_windows_raw.npy'))[mo].astype(np.float32)
t0 = time.time()
Xal = reprs.torso_align(Xraw)
cidx = reprs.context_indices_padded(meta, k=3)
print('aligned', Xal.shape, 'padded context share', round(reprs.padded_share(cidx), 3), round(time.time() - t0), 's', flush=True)
Xctx = reprs.build_context_windows(Xraw, cidx)
Xctx_al = reprs.build_context_windows(Xal, cidx)
XS = {'raw': Xraw, 'aligned': Xal, 'ctx3': Xctx, 'ctx3_aligned': Xctx_al}
y_atomic = ex.encode(meta, 'atomic')
ARMS = {
    'base':          dict(x='raw'),
    'aligned':       dict(x='aligned'),
    'stretch':       dict(x='raw', kw=dict(stretch=True)),
    'aligned_stretch': dict(x='aligned', kw=dict(stretch=True)),
    'ctx3':          dict(x='ctx3'),
    'ctx3_aligned':  dict(x='ctx3_aligned'),
    'hier_foot':     dict(x='raw', head='hier', hierarchy=data.HIER_FOOT_FIRST),
    'hier_walk':     dict(x='raw', head='hier', hierarchy=data.HIER_WALK_FIRST),
    'moe_e2k1':      dict(x='raw', head='moe', head_kw=dict(n_experts=2, top_k=1)),
    'moe_e4k2':      dict(x='raw', head='moe', head_kw=dict(n_experts=4, top_k=2)),
    'moe_e8k2':      dict(x='raw', head='moe', head_kw=dict(n_experts=8, top_k=2)),
    'moe_e8k1':      dict(x='raw', head='moe', head_kw=dict(n_experts=8, top_k=1)),
    'multitask_w05': dict(x='raw', kw=dict(aux_weight=0.5), aux=True),
    'multitask_w1':  dict(x='raw', kw=dict(aux_weight=1.0), aux=True),
    'resnet18':      dict(x='raw', backbone='resnet18'),
    'mobilenetv2':   dict(x='raw', backbone='mobilenetv2'),
}
rows, preds = [], {}
def log_row(**kw):
    rows.append(kw); pd.DataFrame(rows).to_csv(f'{OUT}/stageD_rows.csv', index=False)
def evaluate(site, arm, seed, model, cell, extra):
    pred, lp = train.predict(model, cell.Xte, device=DEV)
    vote, _ = adapt.track_vote(lp, cell.tracks_te)
    for kind, p in (('window', pred), ('vote', vote)):
        log_row(stage='v3', site=site, arm=arm, readout=kind, seed=seed, **extra, **ex.score(cell.yte, p, K))
        preds[f'v3|{site}|{arm}|none|{kind}|{seed}'] = p.astype(np.int8)

for seed in SEEDS:
    for sp in splits.leave_one_location_out(meta, seed=seed):
        site = sp.target
        base_cell = ex.make_cell(Xraw, meta, sp, LEVEL)
        preds[f'y|{site}'] = base_cell.yte.astype(np.int8); preds[f'subject|{site}'] = base_cell.subj_te; preds[f'track|{site}'] = base_cell.tracks_te
        for arm, spec in ARMS.items():
            cell = ex.make_cell(XS[spec['x']], meta, sp, LEVEL)
            kw = dict(spec.get('kw', {}))
            aux = dict(aux_y=y_atomic[sp.train_idx], aux_classes=12) if spec.get('aux') else {}
            m, pred, lg = ex.run_method(cell, LEVEL, spec.get('head', 'flat'), 'erm', seed, EPOCHS, DEV, input_norm=True, batch_size=128, lr=1e-3,
                                        hierarchy=spec.get('hierarchy'), head_kw=spec.get('head_kw'), backbone=spec.get('backbone', 'cnn'), **aux, **kw)
            evaluate(site, arm, seed, m, cell, dict(val_bal_acc=max(r['val_bal_acc'] for r in lg), epochs_run=len(lg)))
            print(seed, site, arm, round(time.time() - t0), 's', flush=True)
        np.savez_compressed(f'{OUT}/stageD_preds.npz', **preds)
print('done', round(time.time() - t0), 's')
