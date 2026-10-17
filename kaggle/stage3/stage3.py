"""Stage 3 (GPU): head designs on the 12 fine activities, then class extension with 7 new single-hand activities."""
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
EPOCHS = 30
LEVEL = 'atomic'
K = len(ex.LEVELS[LEVEL])
REPLAY = [0.0, 0.05, 0.2]
HEADS = ['flat', 'hier', 'moe']
def find(n): return glob.glob('/kaggle/input/**/' + n, recursive=True)[0]
meta_all = pd.read_csv(find('dw_meta.csv'))
data.check_hierarchy_consistency(meta_all)
Xall = np.load(find('dw_windows_base.npy'))
mo_idx = np.flatnonzero((meta_all.task == 'motion').to_numpy())
meta = meta_all.iloc[mo_idx].reset_index(drop=True)
X = Xall[mo_idx]
sg_mask = ((meta_all.task == 'single') & meta_all.label_atomic.isin(data.SINGLE_ATOMIC)).to_numpy()
sg = meta_all[sg_mask].reset_index(drop=True)
Xsg = Xall[sg_mask]
sg_y = sg.label_atomic.map({a: i for i, a in enumerate(data.SINGLE_ATOMIC)}).to_numpy()
print('motion', len(meta), 'single-hand new-class windows', len(sg), sg.groupby('location').size().to_dict(), flush=True)
to_t = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32)).unsqueeze(1)

def rollup(pred_atomic):
    leaves = ex.LEVELS['atomic']
    m = np.array([data.LOAD_CLASSES.index(data.ATOMIC_TO_LOAD[l]) for l in leaves])
    return m[pred_atomic]

rows, preds = [], {}
t0 = time.time()
def log_row(**kw):
    rows.append(kw); pd.DataFrame(rows).to_csv(f'{OUT}/stage3_rows.csv', index=False)

for seed in SEEDS:
    for sp in splits.leave_one_location_out(meta, seed=seed):
        site = sp.target
        cell = ex.make_cell(X, meta, sp, LEVEL)
        y_load_te = ex.encode(meta.iloc[sp.target_idx], 'load')
        # subject sets for the new-class windows: reuse the motion split, so no source-validation subject leaks into training
        tr_sub = set(meta.subject.iloc[sp.train_idx]); va_sub = set(meta.subject.iloc[sp.val_idx])
        rng = np.random.default_rng(1000 + seed)
        extra = sorted(set(sg.subject) - tr_sub - va_sub - set(sg.subject[sg.location == site]))
        extra_val = set(rng.choice(extra, size=max(1, int(round(0.15 * len(extra)))), replace=False)) if extra else set()
        is_site = (sg.location == site).to_numpy()
        is_val = (~is_site) & sg.subject.isin(va_sub | extra_val).to_numpy()
        is_tr = (~is_site) & (~is_val)
        for head in HEADS:
            m, pred, lg = ex.run_method(cell, LEVEL, head, 'erm', seed, EPOCHS, DEV, batch_size=128, lr=1e-3)
            s = ex.score(cell.yte, pred, K)
            s_load = ex.score(y_load_te, rollup(pred), 3)
            log_row(stage='heads', site=site, head=head, seed=seed, replay=np.nan, bal_acc=s['bal_acc'], macro_f1=s['macro_f1'], load_bal_acc=s_load['bal_acc'], n=s['n'],
                    params=int(sum(p.numel() for p in m.parameters())), epochs_run=len(lg))
            preds[f'heads|{site}|{head}|{seed}'] = pred.astype(np.int8)
            # ---- extension
            base_old_ba = s['bal_acc']
            for rf in REPLAY:
                m2, lg2 = ex.extend_and_finetune(m, head, cell, to_t(Xsg[is_tr]), sg_y[is_tr], to_t(Xsg[is_val]), sg_y[is_val], 7, rf, seed, epochs=12, device=DEV, parent='dual', names=list(data.SINGLE_ATOMIC))
                p_old, lp_old = train.predict(m2, cell.Xte, device=DEV)
                ba_old = ex.score(cell.yte, p_old, K + 7)['bal_acc']  # old windows scored against the 19-way head
                p_new, _ = train.predict(m2, to_t(Xsg[is_site]), device=DEV)
                new_rec = float(np.mean(p_new == K + sg_y[is_site])) if is_site.any() else np.nan
                # joint 19-class balanced accuracy on old + new target windows
                y_join = np.concatenate([cell.yte, K + sg_y[is_site]]); p_join = np.concatenate([p_old, p_new])
                s_join = ex.score(y_join, p_join, K + 7)
                log_row(stage='extend', site=site, head=head, seed=seed, replay=rf, bal_acc=ba_old, macro_f1=s_join['macro_f1'], load_bal_acc=s_join['bal_acc'], n=int(len(y_join)),
                        base_old_bal_acc=base_old_ba, new_recall=new_rec, n_new=int(is_site.sum()),
                        trainable=int(sum(p.numel() for p in m2.parameters() if p.requires_grad)), epochs_run=len(lg2))
                preds[f'ext|{site}|{head}|{rf}|{seed}'] = p_join.astype(np.int8)
            print(seed, site, head, round(time.time() - t0), 's', flush=True)
        preds[f'y_old|{site}'] = cell.yte.astype(np.int8); preds[f'subject_old|{site}'] = cell.subj_te
        np.savez_compressed(f'{OUT}/stage3_preds.npz', **preds)
print('done', round(time.time() - t0), 's')
