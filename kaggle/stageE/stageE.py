"""Stage E (GPU): exploratory sweep, v3 ideas that use the track structure (docs/V3_PLAN.md). Seeds 0-2, four held-out sites.

T: track-level multiple-instance models (attention pooling, GRU), optionally on torso-aligned or stretch-augmented windows.
P: track-contrastive pretraining on the unlabeled tracks (source-site recordings only, or all recordings), followed by
   window-level fine-tuning or by an attention model. Track models predict whole tracks, so their only readout is 'vote'.
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
from sensewhere import adapt, contrastive, data, models, reprs, splits, track, train
from sensewhere import experiments as ex

OUT = os.environ.get('SW_OUT', '/kaggle/working')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SEEDS, EPOCHS, LEVEL, K = [0, 1, 2], 25, 'load', 3
PRE_STEPS = 1500
if os.environ.get('SW_QUICK'):
    SEEDS, EPOCHS, PRE_STEPS = [0], 2, 20
def find(n): return glob.glob(ROOT + '/**/' + n, recursive=True)[0]
meta_all = pd.read_csv(find('dw_meta.csv'))
mo = (meta_all.task == 'motion').to_numpy()
meta = meta_all[mo].reset_index(drop=True)
Xraw = np.load(find('dw_windows_raw.npy'))[mo].astype(np.float32)
Xal = reprs.torso_align(Xraw)
ulm = pd.read_csv(find('ul_meta.csv'))
Xul = torch.from_numpy(np.load(find('ul_windows_raw.npy')).astype(np.float32)).unsqueeze(1)
ul_keys = (ulm.file_name.astype(str) + '|' + ulm.track.astype(str)).to_numpy()
XS = {'raw': Xraw, 'aligned': Xal}
tstart = meta.t_start.to_numpy(dtype=float)
rows, preds = [], {}
t0 = time.time()
def log_row(**kw):
    rows.append(kw); pd.DataFrame(rows).to_csv(f'{OUT}/stageE_rows.csv', index=False)
def stretch_fn(x):
    fac = torch.exp(torch.empty(len(x), device=x.device).uniform_(np.log(0.7), np.log(1.4)))
    return reprs.stretch_doppler(x, fac)

def track_index(keys, idx_or_t, y):
    return track.index_tracks(keys, idx_or_t, y)

def run_track_arm(cell, sp, pool, seed, init=None, aug=None):
    ti = track.index_tracks((meta.file_name.astype(str) + '|' + meta.subject.astype(str)).to_numpy()[sp.train_idx], tstart[sp.train_idx], cell.ytr)
    tiv = track.index_tracks((meta.file_name.astype(str) + '|' + meta.subject.astype(str)).to_numpy()[sp.val_idx], tstart[sp.val_idx], cell.yva)
    tit = track.index_tracks(cell.tracks_te, tstart[sp.target_idx], cell.yte)
    bb = models.Backbone(in_ch=1, widths=(16, 32, 64, 64), feat_dim=64)
    if init is not None:
        from sensewhere.mae import copy_encoder
        copy_encoder(init, bb)
    m = track.TrackModel(bb, K, pool=pool, input_norm=True)
    m, lg = track.fit_tracks(m, cell.Xtr, ti, cell.Xva, tiv, K, epochs=EPOCHS, seed=seed, device=DEV, augment_fn=aug)
    lp, win = track.predict_tracks(m, cell.Xte, tit, device=DEV)
    return win, max(r['val_bal_acc'] for r in lg)

pretrained = {}
def get_pre(site, kind, seed):
    key = (site, kind)
    if key not in pretrained:
        keep = (ulm.site.notna() & (ulm.site != site)).to_numpy() if kind == 'src' else np.ones(len(ulm), bool)
        bb = models.Backbone(in_ch=1, widths=(16, 32, 64, 64), feat_dim=64)
        losses = contrastive.pretrain_contrastive(bb, Xul[torch.from_numpy(keep)], ul_keys[keep], steps=PRE_STEPS, seed=0, device=DEV,
                                                  stretch=lambda x, g: reprs.stretch_doppler(x, torch.exp(torch.empty(len(x), device=x.device).uniform_(np.log(0.8), np.log(1.25)))))
        pretrained[key] = bb
        log_row(stage='pretrain', site=site, arm=f'contrastive_{kind}', n=int(keep.sum()), first=losses[0], last=losses[-1])
        print('pretrained', key, int(keep.sum()), round(losses[0], 3), '->', round(losses[-1], 3), round(time.time() - t0), 's', flush=True)
    return pretrained[key]

for seed in SEEDS:
    for sp in splits.leave_one_location_out(meta, seed=seed):
        site = sp.target
        cell = ex.make_cell(Xraw, meta, sp, LEVEL)
        cell_al = ex.make_cell(Xal, meta, sp, LEVEL)
        preds[f'y|{site}'] = cell.yte.astype(np.int8); preds[f'subject|{site}'] = cell.subj_te; preds[f'track|{site}'] = cell.tracks_te
        def rec(arm, win, val):
            log_row(stage='v3', site=site, arm=arm, readout='vote', seed=seed, val_bal_acc=val, **ex.score(cell.yte, win, K))
            preds[f'v3|{site}|{arm}|none|vote|{seed}'] = win.astype(np.int8)
        for arm, (c, pool, aug) in {'mil_attn': (cell, 'attn', None), 'mil_gru': (cell, 'gru', None), 'mil_attn_aligned': (cell_al, 'attn', None), 'mil_attn_stretch': (cell, 'attn', stretch_fn)}.items():
            win, val = run_track_arm(c, sp, pool, seed, aug=aug)
            rec(arm, win, val)
            print(seed, site, arm, round(time.time() - t0), 's', flush=True)
        for kind in ('src', 'all'):
            bb = get_pre(site, kind, seed)
            m, pred, lg = ex.run_method(cell, LEVEL, 'flat', 'erm', seed, EPOCHS, DEV, input_norm=True, batch_size=128, lr=1e-3, init=bb)
            pr, lp = train.predict(m, cell.Xte, device=DEV)
            vote, _ = adapt.track_vote(lp, cell.tracks_te)
            for rk, p in (('window', pr), ('vote', vote)):
                log_row(stage='v3', site=site, arm=f'con_{kind}_ft', readout=rk, seed=seed, val_bal_acc=max(r['val_bal_acc'] for r in lg), **ex.score(cell.yte, p, K))
                preds[f'v3|{site}|con_{kind}_ft|none|{rk}|{seed}'] = p.astype(np.int8)
        win, val = run_track_arm(cell, sp, 'attn', seed, init=get_pre(site, 'src', seed))
        rec('con_src_mil_attn', win, val)
        print(seed, site, 'pretrained arms', round(time.time() - t0), 's', flush=True)
        np.savez_compressed(f'{OUT}/stageE_preds.npz', **preds)
print('done', round(time.time() - t0), 's')
