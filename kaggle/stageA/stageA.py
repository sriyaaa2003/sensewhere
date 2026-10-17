"""Stage A (GPU), v2 protocol. DopplerWild 3-class, leave-one-location-out, 3 seeds.

Fixes over v1: augmentation is an explicit switch, model selection uses nested leave-one-SOURCE-site-out (the target never
influences any choice), and all models standardise each window inside the network so noise-floor tools can work in dB.
Arms: ERM, +augmentation, +noise-floor transplant, CORAL, Group-DRO, site-profile calibration (two quantiles) and
masked-reconstruction pretraining on unlabeled tracks (source-site recordings only, or all recordings).
Source-free adaptation and track voting are applied on top.
"""
import glob, os, shutil, sys, json, time
ROOT = os.environ.get('SW_INPUT', '/kaggle/input')   # SW_* variables allow a local smoke test on synthetic data
if os.environ.get('SW_PKG'):
    sys.path.insert(0, os.environ['SW_PKG'])
else:
    f = glob.glob(ROOT + '/**/preprocess.py', recursive=True)
    os.makedirs('/tmp/pkg/sensewhere', exist_ok=True)
    for p in glob.glob(os.path.dirname(f[0]) + '/*.py'):
        shutil.copy(p, '/tmp/pkg/sensewhere')
    sys.path.insert(0, '/tmp/pkg')
import numpy as np, pandas as pd, torch
from sensewhere import adapt, data, floor, mae, models, splits, train
from sensewhere import experiments as ex

OUT = os.environ.get('SW_OUT', '/kaggle/working')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SEEDS, EPOCHS, LEVEL, K = [0, 1, 2], 25, 'load', 3
PRE_EPOCHS = 15
if os.environ.get('SW_QUICK'):
    SEEDS, EPOCHS, PRE_EPOCHS = [0], 2, 1
def find(n): return glob.glob(ROOT + '/**/' + n, recursive=True)[0]
meta_all = pd.read_csv(find('dw_meta.csv'))
data.check_hierarchy_consistency(meta_all)
mo = (meta_all.task == 'motion').to_numpy()
meta = meta_all[mo].reset_index(drop=True)
Xraw = np.load(find('dw_windows_raw.npy'))[mo]
sites_arr = meta.location.to_numpy()
ulm = pd.read_csv(find('ul_meta.csv'))
Xul = torch.from_numpy(np.load(find('ul_windows_raw.npy')).astype(np.float32)).unsqueeze(1)
print('labeled motion', Xraw.shape, 'unlabeled windows', tuple(Xul.shape), ulm.site.value_counts(dropna=False).to_dict(), flush=True)

# site profiles use windows only (no labels). q50 = typical level, q10 = noise floor
prof = {q: floor.site_profiles(Xraw, sites_arr, q) for q in (50.0, 10.0)}
X_sitecal = {q: floor.calibrate(Xraw, sites_arr, prof[q]) for q in (50.0, 10.0)}
transplant_profiles = prof[10.0]

GRID_LR = [dict(lr=1e-3), dict(lr=3e-3)]
ARMS = {
    # name: (X key, fixed kwargs, grid, needs profiles)
    'erm':            ('raw', dict(method='erm'), GRID_LR),
    'erm_aug':        ('raw', dict(method='erm', augment=True), GRID_LR),
    'erm_floor':      ('raw', dict(method='erm', floor_transplant=True), GRID_LR),
    'erm_aug_floor':  ('raw', dict(method='erm', augment=True, floor_transplant=True), GRID_LR),
    'coral':          ('raw', dict(method='coral', lr=1e-3), [dict(method_weight=0.1), dict(method_weight=1.0)]),
    'groupdro':       ('raw', dict(method='groupdro', lr=1e-3), [dict(dro_step=0.01), dict(dro_step=0.1)]),
    'sitecal_q50':    ('cal50', dict(method='erm'), GRID_LR),
    'sitecal_q10':    ('cal10', dict(method='erm'), GRID_LR),
}
XS = {'raw': Xraw, 'cal50': X_sitecal[50.0], 'cal10': X_sitecal[10.0]}
rows, preds, sel_log = [], {}, []
t0 = time.time()
def log_row(**kw):
    rows.append(kw); pd.DataFrame(rows).to_csv(f'{OUT}/stageA_rows.csv', index=False)

def evaluate(site, arm, adapt_name, seed, model, cell, sel_cfg, logp=None, extra=None):
    pred, lp = train.predict(model, cell.Xte, device=DEV)
    vote, _ = adapt.track_vote(lp, cell.tracks_te)
    for kind, p in (('window', pred), ('vote', vote)):
        s = ex.score(cell.yte, p, K)
        log_row(stage='v2', site=site, arm=arm, adapt=adapt_name, readout=kind, seed=seed, cfg=json.dumps(sel_cfg), collapse=ex.collapse_share(p, K), **(extra or {}), **s)
        preds[f'v2|{site}|{arm}|{adapt_name}|{kind}|{seed}'] = p.astype(np.int8)

pretrained = {}  # (site, kind) -> Backbone
def get_pretrained(site, kind):
    key = (site, kind)
    if key not in pretrained:
        if kind == 'src':
            keep = ulm.site.notna() & (ulm.site != site)
        else:
            keep = np.ones(len(ulm), bool)
        Xp = Xul[torch.from_numpy(np.asarray(keep))]
        bb = models.Backbone(in_ch=1, widths=(16, 32, 64, 64), feat_dim=64)
        losses = mae.pretrain(bb, Xp, epochs=PRE_EPOCHS, device=DEV, seed=0)
        pretrained[key] = bb
        sel_log.append(dict(kind='pretrain', site=site, ssl=kind, n=int(len(Xp)), first=losses[0], last=losses[-1]))
        print('pretrained', key, len(Xp), round(losses[0], 3), '->', round(losses[-1], 3), round(time.time() - t0), 's', flush=True)
    return pretrained[key]

chosen = {}
for seed in SEEDS:
    for sp in splits.leave_one_location_out(meta, seed=seed):
        site = sp.target
        base_cell = ex.make_cell(Xraw, meta, sp, LEVEL)
        preds[f'y|{site}'] = base_cell.yte.astype(np.int8)
        preds[f'subject|{site}'] = base_cell.subj_te
        preds[f'track|{site}'] = base_cell.tracks_te
        models_by_arm = {}
        for arm, (xk, fixed, grid) in ARMS.items():
            X = XS[xk]
            cell = ex.make_cell(X, meta, sp, LEVEL)
            if (site, arm) not in chosen:  # nested selection once per site/arm (seed 0 split), reused for later seeds
                gi, scores = ex.select_config(X, meta, sp, LEVEL, 'flat', fixed['method'],
                                              grid, seed=0, epochs=EPOCHS, device=DEV, site_profiles=transplant_profiles,
                                              input_norm=True, batch_size=128, **{k: v for k, v in fixed.items() if k != 'method'})
                chosen[(site, arm)] = gi
                sel_log.append(dict(kind='select', site=site, arm=arm, grid=json.dumps(grid), inner_scores=json.dumps(scores), chosen=gi))
            kw = grid[chosen[(site, arm)]]
            m, pred, lg = ex.run_method(cell, LEVEL, 'flat', fixed['method'], seed, EPOCHS, DEV, input_norm=True, batch_size=128,
                                        profiles=transplant_profiles if fixed.get('floor_transplant') else None,
                                        **{k: v for k, v in fixed.items() if k != 'method'}, **kw)
            evaluate(site, arm, 'none', seed, m, cell, kw, extra=dict(val_bal_acc=max(r['val_bal_acc'] for r in lg)))
            models_by_arm[arm] = (m, cell)
            print(seed, site, arm, round(time.time() - t0), 's', flush=True)
        # pretrained arms (fixed lr; pretraining excluded the target site for 'src')
        for kind in ('src', 'all'):
            bb = get_pretrained(site, kind)
            m, pred, lg = ex.run_method(base_cell, LEVEL, 'flat', 'erm', seed, EPOCHS, DEV, input_norm=True, batch_size=128, init=bb, lr=1e-3)
            evaluate(site, f'ssl_{kind}', 'none', seed, m, base_cell, dict(lr=1e-3), extra=dict(val_bal_acc=max(r['val_bal_acc'] for r in lg)))
            models_by_arm[f'ssl_{kind}'] = (m, base_cell)
        # source-free adaptation on three source models
        for arm in ('erm', 'erm_floor', 'ssl_src'):
            m, cell = models_by_arm[arm]
            Xt = cell.Xte
            batches = [Xt[i:i + 128] for i in range(0, len(Xt), 128)]
            variants = {
                'adabn': lambda: adapt.adabn(m, batches, DEV),
                'tent_e1': lambda: adapt.tent(m, batches, epochs=1, device=DEV),
                'shot_bn': lambda: adapt.shot(m, Xt, params='bn', device=DEV),
                'shot_all': lambda: adapt.shot(m, Xt, params='all', device=DEV),
                'tcst': lambda: adapt.tcst(m, Xt, cell.tracks_te, device=DEV),
            }
            for name, fn in variants.items():
                m2, tr = fn()
                evaluate(site, arm, name, seed, m2, cell, {}, extra=dict(ent_last=float(tr[-1]) if len(tr) else np.nan))
        print(seed, site, 'adapted', round(time.time() - t0), 's', flush=True)
        np.savez_compressed(f'{OUT}/stageA_preds.npz', **preds)
        pd.DataFrame(sel_log).to_csv(f'{OUT}/stageA_selection.csv', index=False)
print('done', round(time.time() - t0), 's')
