"""Stage 4 (GPU): Widar 3.0 BVP replication. Session 20181130: 9 users, gestures 1-9, 5 torso locations (domains).
Base task = gestures 1-6; extension adds gestures 7-9. Flat and mixture-of-experts heads only (Widar has no hierarchy)."""
import glob, os, re, shutil, sys, json, time, collections
f = glob.glob('/kaggle/input/**/preprocess.py', recursive=True)
os.makedirs('/tmp/pkg/sensewhere', exist_ok=True)
for p in glob.glob(os.path.dirname(f[0]) + '/*.py'):
    shutil.copy(p, '/tmp/pkg/sensewhere')
sys.path.insert(0, '/tmp/pkg')
import numpy as np, pandas as pd, torch, scipy.io as sio
from sensewhere import preprocess as pp, splits, train
from sensewhere import experiments as ex

OUT = '/kaggle/working'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SESSION = '20181130'
BASE = [f'g{i}' for i in range(1, 7)]
NEW = [f'g{i}' for i in range(7, 10)]
SEEDS, EPOCHS, REPLAY = [0, 1, 2], 25, [0.0, 0.05, 0.2]
K = len(BASE)
IN_CH = 20

# ---- manifest + preprocessing with QC
rx = re.compile(r'BVP/(\d{8})-VS/([^/]+)/(?:(user\d+)/)?(user\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(.*)\.mat$')
mats = sorted(glob.glob('/kaggle/input/**/BVP/' + SESSION + '-VS/**/*.mat', recursive=True))
print('files in session', len(mats), flush=True)
cfg = pp.BVPConfig(n_frames=20, normalise='window')
rows, tens, qc = [], [], collections.Counter()
for p in mats:
    r = rx.search(p.replace(chr(92), '/'))
    if not r:
        qc['bad_name'] += 1; continue
    date, grp, _, user, g, loc, ori, rep, rest = r.groups()
    a = sio.loadmat(p)['velocity_spectrum_ro']
    x, why = pp.process_bvp(a, cfg)
    qc[why] += 1
    if x is None: continue
    tens.append(x.astype(np.float16))
    rows.append(dict(subject=user, location=f'L{loc}', ori=int(ori), rep=int(rep), label_load=f'g{g}', frames=int(a.shape[2])))
X = np.stack(tens); meta = pd.DataFrame(rows)
print('built', X.shape, dict(qc), meta.groupby('label_load').size().to_dict(), meta.groupby('location').size().to_dict(), flush=True)
pd.Series(dict(qc)).to_csv(f'{OUT}/widar_qc.csv')

base_mask = meta.label_load.isin(BASE).to_numpy()
mb = meta[base_mask].reset_index(drop=True); Xb = X[base_mask]
mn = meta[~base_mask].reset_index(drop=True); Xn = X[~base_mask]
to_t = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32))
GRID = {
    'erm':      [dict(lr=1e-3), dict(lr=3e-3)],
    'coral':    [dict(method_weight=0.1), dict(method_weight=1.0)],
    'dann':     [dict(method_weight=0.1), dict(method_weight=1.0)],
    'groupdro': [dict(dro_step=0.01), dict(dro_step=0.1)],
}
rows_out, preds = [], {}
t0 = time.time()
def log_row(**kw):
    rows_out.append(kw); pd.DataFrame(rows_out).to_csv(f'{OUT}/stage4_rows.csv', index=False)

for seed in SEEDS:
    for sp in splits.leave_one_domain_out_shared(mb, 'location', 'subject', val_fraction=0.22, seed=seed):
        site = sp.target
        cell = ex.make_cell(Xb, mb, sp, 'load', classes=BASE)
        preds[f'y|{site}'] = cell.yte.astype(np.int8); preds[f'subject|{site}'] = cell.subj_te
        sel = {}
        for method, grid in GRID.items():
            best = None
            for gi, kw in enumerate(grid):
                m, pred, lg = ex.run_method(cell, 'load', 'flat', method, seed, EPOCHS, DEV, in_ch=IN_CH, n_classes=K, batch_size=128, **kw)
                val = max(r['val_bal_acc'] for r in lg)
                log_row(stage='dg', site=site, method=method, adapt='none', head='flat', seed=seed, cfg=json.dumps(kw), val_bal_acc=val, selected=False, collapse=ex.collapse_share(pred, K), **ex.score(cell.yte, pred, K))
                if best is None or val > best[0]: best = (val, gi, m, pred)
            val, gi, m, pred = best
            rows_out[-(len(grid) - gi)]['selected'] = True
            preds[f'dg|{site}|{method}|none|{seed}'] = pred.astype(np.int8); sel[method] = m
        res = ex.adapt_rows(sel['erm'], cell, 'load')
        for name, (p, tr) in res.items():
            log_row(stage='sfda', site=site, method='erm', adapt=name, head='flat', seed=seed, cfg='', selected=True, collapse=ex.collapse_share(p, K), ent_first=float(tr[0]), ent_last=float(tr[-1]), **ex.score(cell.yte, p, K))
            preds[f'sfda|{site}|erm|{name}|{seed}'] = p.astype(np.int8)
        # heads + extension (new gestures 7-9 at source locations for training, target location for testing)
        is_site = (mn.location == site).to_numpy()
        va_users = set(mb.subject.iloc[sp.val_idx])
        is_val = (~is_site) & mn.subject.isin(va_users).to_numpy()
        is_tr = (~is_site) & (~is_val)
        yn = mn.label_load.map({g: i for i, g in enumerate(NEW)}).to_numpy()
        for head in ['flat', 'moe']:
            m, pred, lg = (sel['erm'], None, None) if head == 'flat' else ex.run_method(cell, 'load', head, 'erm', seed, EPOCHS, DEV, in_ch=IN_CH, n_classes=K, batch_size=128, lr=1e-3)
            pb = train.predict(m, cell.Xte, device=DEV)[0]
            s0 = ex.score(cell.yte, pb, K)
            log_row(stage='heads', site=site, method='erm', adapt='none', head=head, seed=seed, cfg='', selected=True, collapse=ex.collapse_share(pb, K), **s0)
            for rf in REPLAY:
                m2, lg2 = ex.extend_and_finetune(m, head, cell, to_t(Xn[is_tr]).contiguous(), yn[is_tr], to_t(Xn[is_val]).contiguous(), yn[is_val], len(NEW), rf, seed, epochs=12, device=DEV)
                p_old, _ = train.predict(m2, cell.Xte, device=DEV)
                p_new, _ = train.predict(m2, to_t(Xn[is_site]), device=DEV)
                y_join = np.concatenate([cell.yte, K + yn[is_site]]); p_join = np.concatenate([p_old, p_new])
                log_row(stage='extend', site=site, method='erm', adapt='none', head=head, seed=seed, cfg=str(rf), selected=True, replay=rf,
                        bal_acc=ex.score(cell.yte, p_old, K + len(NEW))['bal_acc'], base_old_bal_acc=s0['bal_acc'], new_recall=float(np.mean(p_new == K + yn[is_site])),
                        joint_bal_acc=ex.score(y_join, p_join, K + len(NEW))['bal_acc'], n=int(len(y_join)))
                preds[f'ext|{site}|{head}|{rf}|{seed}'] = p_join.astype(np.int8)
        print(seed, site, round(time.time() - t0), 's', flush=True)
        np.savez_compressed(f'{OUT}/stage4_preds.npz', **preds)
print('done', round(time.time() - t0), 's')
