"""Stage 5 (CPU): class extension at the motion-class level. Base = walking, dual-hand load, on wheels.
New class = single-hand load, placed under a 'load' node next to dual-hand load in the hierarchical head."""
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
torch.set_num_threads(4)
SEEDS, EPOCHS, REPLAY, HEADS = [0, 1, 2], 25, [0.0, 0.05, 0.2], ['flat', 'hier', 'moe']
LEVEL, K = 'load', 3
HIER = data.Hierarchy({"root": ["normal_walking", "load", "On Wheels"], "load": ["Dual-hand Load"]}, ["normal_walking", "Dual-hand Load", "On Wheels"])
def find(n): return glob.glob('/kaggle/input/**/' + n, recursive=True)[0]
meta_all = pd.read_csv(find('dw_meta.csv'))
Xall = np.load(find('dw_windows_base.npy'))
mo = (meta_all.task == 'motion').to_numpy()
meta, X = meta_all[mo].reset_index(drop=True), Xall[mo]
sgm = ((meta_all.task == 'single') & (meta_all.label_load == 'Single-hand Load')).to_numpy()
sg, Xsg = meta_all[sgm].reset_index(drop=True), Xall[sgm]
print('motion', len(meta), 'single-hand windows', len(sg), sg.groupby('location').size().to_dict(), flush=True)
to_t = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32)).unsqueeze(1)
rows, preds = [], {}
t0 = time.time()
def log_row(**kw):
    rows.append(kw); pd.DataFrame(rows).to_csv(f'{OUT}/stage5_rows.csv', index=False)
for seed in SEEDS:
    for sp in splits.leave_one_location_out(meta, seed=seed):
        site = sp.target
        cell = ex.make_cell(X, meta, sp, LEVEL)
        tr_sub, va_sub = set(meta.subject.iloc[sp.train_idx]), set(meta.subject.iloc[sp.val_idx])
        rng = np.random.default_rng(1000 + seed)
        extra = sorted(set(sg.subject) - tr_sub - va_sub - set(sg.subject[sg.location == site]))
        extra_val = set(rng.choice(extra, size=max(1, int(round(0.15 * len(extra)))), replace=False)) if extra else set()
        is_site = (sg.location == site).to_numpy()
        is_val = (~is_site) & sg.subject.isin(va_sub | extra_val).to_numpy()
        is_tr = (~is_site) & (~is_val)
        zeros = lambda n: np.zeros(n, dtype=int)
        preds[f'y_old|{site}'] = cell.yte.astype(np.int8); preds[f'subject_old|{site}'] = cell.subj_te
        for head in HEADS:
            m, pred, lg = ex.run_method(cell, LEVEL, head, 'erm', seed, EPOCHS, 'cpu', batch_size=128, lr=1e-3, hierarchy=HIER)
            s0 = ex.score(cell.yte, pred, K)
            log_row(stage='base', site=site, head=head, seed=seed, replay=np.nan, freeze=np.nan, bal_acc=s0['bal_acc'], macro_f1=s0['macro_f1'])
            preds[f'base|{site}|{head}|{seed}'] = pred.astype(np.int8)
            for freeze in [True, False]:
                for rf in REPLAY:
                    m2, lg2 = ex.extend_and_finetune(m, head, cell, to_t(Xsg[is_tr]), zeros(is_tr.sum()), to_t(Xsg[is_val]), zeros(is_val.sum()), 1, rf, seed,
                                                     epochs=12, device='cpu', parent='load', names=['Single-hand Load'], freeze_backbone=freeze)
                    p_old, _ = train.predict(m2, cell.Xte)
                    p_new, _ = train.predict(m2, to_t(Xsg[is_site]))
                    y_join = np.concatenate([cell.yte, np.full(is_site.sum(), 3)]); p_join = np.concatenate([p_old, p_new])
                    log_row(stage='extend', site=site, head=head, seed=seed, replay=rf, freeze=freeze, bal_acc=ex.score(cell.yte, p_old, 4)['bal_acc'], base_old_bal_acc=s0['bal_acc'],
                            new_recall=float(np.mean(p_new == 3)), joint_bal_acc=ex.score(y_join, p_join, 4)['bal_acc'], n_new=int(is_site.sum()),
                            trainable=int(sum(p.numel() for p in m2.parameters() if p.requires_grad)))
                    preds[f'ext|{site}|{head}|{rf}|{freeze}|{seed}'] = p_join.astype(np.int8)
            print(seed, site, head, round(time.time() - t0), 's', flush=True)
        np.savez_compressed(f'{OUT}/stage5_preds.npz', **preds)
print('done', round(time.time() - t0), 's')
