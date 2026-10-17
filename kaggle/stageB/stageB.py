"""Stage B (GPU), v2 class extension at the motion-class level.

Base task: walking / dual-hand load / on wheels. New class: single-hand load (placed under a 'load' node in the tree).
Methods: head expansion with replay budgets 0 / 5% / 20% (flat, tree, mixture of experts; backbone and its BatchNorm
statistics frozen, or fine-tuned) and prototype extension (nearest class mean; no replay, no boundary to learn, N-shot).
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
from sensewhere import data, proto, splits, train
from sensewhere import experiments as ex

OUT = os.environ.get('SW_OUT', '/kaggle/working')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SEEDS, EPOCHS, REPLAY, HEADS, SHOTS = [0, 1, 2, 3, 4], 25, [0.0, 0.05, 0.2], ['flat', 'hier', 'moe'], [5, 20, None]
if os.environ.get('SW_QUICK'):
    SEEDS, EPOCHS = [0], 2
LEVEL, K = 'load', 3
HIER = data.Hierarchy({"root": ["normal_walking", "load", "On Wheels"], "load": ["Dual-hand Load"]}, ["normal_walking", "Dual-hand Load", "On Wheels"])
def find(n): return glob.glob(ROOT + '/**/' + n, recursive=True)[0]
meta_all = pd.read_csv(find('dw_meta.csv'))
Xall = np.load(find('dw_windows_raw.npy'))
mo = (meta_all.task == 'motion').to_numpy()
meta, X = meta_all[mo].reset_index(drop=True), Xall[mo]
sgm = ((meta_all.task == 'single') & (meta_all.label_load == 'Single-hand Load')).to_numpy()
sg, Xsg = meta_all[sgm].reset_index(drop=True), Xall[sgm]
print('motion', len(meta), 'single-hand windows', len(sg), flush=True)
to_t = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32)).unsqueeze(1)
rows, preds = [], {}
t0 = time.time()
def log_row(**kw):
    rows.append(kw); pd.DataFrame(rows).to_csv(f'{OUT}/stageB_rows.csv', index=False)

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
        Xnew_tr, Xnew_va, Xnew_te = to_t(Xsg[is_tr]), to_t(Xsg[is_val]), to_t(Xsg[is_site])
        flat_model = None
        for head in HEADS:
            m, pred, lg = ex.run_method(cell, LEVEL, head, 'erm', seed, EPOCHS, DEV, input_norm=True, batch_size=128, lr=1e-3, hierarchy=HIER)
            s0 = ex.score(cell.yte, pred, K)
            log_row(stage='base', site=site, head=head, seed=seed, mode='-', replay=np.nan, shots=np.nan, bal_acc=s0['bal_acc'], macro_f1=s0['macro_f1'])
            if head == 'flat':
                flat_model = m
            for freeze in [True, False]:
                for rf in REPLAY:
                    m2, lg2 = ex.extend_and_finetune(m, head, cell, Xnew_tr, zeros(is_tr.sum()), Xnew_va, zeros(is_val.sum()), 1, rf, seed,
                                                     epochs=12, device=DEV, parent='load', names=['Single-hand Load'], freeze_backbone=freeze)
                    p_old, _ = train.predict(m2, cell.Xte, device=DEV)
                    p_new, _ = train.predict(m2, Xnew_te, device=DEV)
                    y_join = np.concatenate([cell.yte, np.full(is_site.sum(), 3)]); p_join = np.concatenate([p_old, p_new])
                    log_row(stage='extend', site=site, head=head, seed=seed, mode='frozen' if freeze else 'finetune', replay=rf, shots=np.nan,
                            bal_acc=ex.score(cell.yte, p_old, 4)['bal_acc'], base_old_bal_acc=s0['bal_acc'], new_recall=float(np.mean(p_new == 3)),
                            joint_bal_acc=ex.score(y_join, p_join, 4)['bal_acc'], n_new=int(is_site.sum()),
                            trainable=int(sum(p.numel() for p in m2.parameters() if p.requires_grad)))
                    preds[f'ext|{site}|{head}|{rf}|{"frozen" if freeze else "finetune"}|{seed}'] = p_join.astype(np.int8)
            print(seed, site, head, round(time.time() - t0), 's', flush=True)
        # prototype extension on the flat model's embedding: old classes keep ONE stored mean each (memory: 3 vectors)
        emb_tr = proto.embed(flat_model, cell.Xtr, device=DEV)
        means = proto.class_means(emb_tr, cell.ytr, K)
        emb_te = proto.embed(flat_model, cell.Xte, device=DEV)
        base_p = proto.ncm_predict(emb_te, means)
        ba_base = ex.score(cell.yte, base_p, K)['bal_acc']
        emb_new_tr = proto.embed(flat_model, Xnew_tr, device=DEV)
        emb_new_te = proto.embed(flat_model, Xnew_te, device=DEV)
        for shots in SHOTS:
            r2 = np.random.default_rng(seed)
            sel = np.arange(len(emb_new_tr)) if shots is None else r2.choice(len(emb_new_tr), size=min(shots, len(emb_new_tr)), replace=False)
            ext = proto.extend_means(means, emb_new_tr[sel])
            p_old = proto.ncm_predict(emb_te, ext)
            p_new = proto.ncm_predict(emb_new_te, ext)
            y_join = np.concatenate([cell.yte, np.full(len(p_new), 3)]); p_join = np.concatenate([p_old, p_new])
            log_row(stage='extend', site=site, head='ncm', seed=seed, mode='prototype', replay=0.0, shots=-1 if shots is None else shots,
                    bal_acc=ex.score(cell.yte, p_old, 4)['bal_acc'], base_old_bal_acc=ba_base, new_recall=float(np.mean(p_new == 3)),
                    joint_bal_acc=ex.score(y_join, p_join, 4)['bal_acc'], n_new=int(len(p_new)), trainable=0)
            preds[f'ext|{site}|ncm|{shots}|prototype|{seed}'] = p_join.astype(np.int8)
        log_row(stage='base', site=site, head='ncm', seed=seed, mode='-', replay=np.nan, shots=np.nan, bal_acc=ba_base, macro_f1=np.nan)
        np.savez_compressed(f'{OUT}/stageB_preds.npz', **preds)
print('done', round(time.time() - t0), 's')
