"""Stage 1: domain-shift analysis, majority and hand-feature gradient-boosting baselines (CPU)."""
import glob, os, shutil, sys, json, time
f = glob.glob('/kaggle/input/**/preprocess.py', recursive=True)
os.makedirs('/tmp/pkg/sensewhere', exist_ok=True)
for p in glob.glob(os.path.dirname(f[0]) + '/*.py'):
    shutil.copy(p, '/tmp/pkg/sensewhere')
sys.path.insert(0, '/tmp/pkg')
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sensewhere import data, features, splits
from sensewhere.experiments import encode, LEVELS, score

OUT = '/kaggle/working'
def find(n): return glob.glob('/kaggle/input/**/' + n, recursive=True)[0]
meta = pd.read_csv(find('dw_meta.csv'))
data.check_hierarchy_consistency(meta)
print(meta.groupby('task').size(), flush=True)
mo = meta[meta.task == 'motion'].reset_index(drop=True)
mo_idx = np.flatnonzero((meta.task == 'motion').to_numpy())

def feats(variant):
    X = np.load(find(f'dw_windows_{variant}.npy'), mmap_mode='r')
    F = []
    for i in mo_idx:
        x = np.asarray(X[i], dtype=np.float64)
        F.append(features.window_features(x - x.min()))
    names = list(F[0].keys())
    return np.array([[r[n] for n in names] for r in F], dtype=np.float32), names

t0 = time.time()
FE = {v: feats(v) for v in ['raw', 'base', 'robust']}
print('features', {v: FE[v][0].shape for v in FE}, round(time.time() - t0), 's', flush=True)

# 1. priors per site
pri = pd.crosstab(mo.location, mo.label_load); pri.to_csv(f'{OUT}/shift_priors.csv')
sub = mo.groupby(['location', 'label_load']).subject.nunique().unstack(); sub.to_csv(f'{OUT}/shift_subjects.csv')
print(pri, sub, sep='\n', flush=True)

# 2. can a classifier tell the sites apart?  subject-grouped CV, macro one-vs-rest AUC, by preprocessing variant and by class
rows = []
sites = sorted(mo.location.unique())
for variant, (F, names) in FE.items():
    for scope in ['all'] + data.LOAD_CLASSES:
        m = np.ones(len(mo), bool) if scope == 'all' else (mo.label_load == scope).to_numpy()
        Fm, ym, gm = F[m], mo.location[m].to_numpy(), mo.subject[m].to_numpy()
        if len(np.unique(ym)) < 4: continue
        proba = np.zeros((len(ym), len(sites)))
        for tr, te in GroupKFold(5).split(Fm, ym, gm):
            clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, random_state=0).fit(Fm[tr], ym[tr])
            proba[te][:, :] = 0
            pr = clf.predict_proba(Fm[te])
            for j, c in enumerate(clf.classes_): proba[te, sites.index(c)] = pr[:, j]
        aucs = {s: roc_auc_score((ym == s).astype(int), proba[:, sites.index(s)]) for s in sites}
        rows.append(dict(variant=variant, scope=scope, n=int(m.sum()), auc_macro=float(np.mean(list(aucs.values()))), **{f'auc_{s}': v for s, v in aucs.items()}))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(f'{OUT}/shift_site_auc.csv', index=False)

# 3. baselines under leave-one-location-out
F, names = FE['base']
sp_all = splits.leave_one_location_out(mo, seed=0)
out_rows, preds = [], {}
for level in ['load', 'atomic']:
    y = encode(mo, level); K = len(LEVELS[level])
    for sp in sp_all:
        te = sp.target_idx
        tr = np.concatenate([sp.train_idx, sp.val_idx])
        maj = np.bincount(y[tr], minlength=K).argmax()
        pm = np.full(len(te), maj)
        out_rows.append(dict(level=level, site=sp.target, method='majority', seed=0, **score(y[te], pm, K)))
        for seed in range(3):
            clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, class_weight='balanced', random_state=seed).fit(F[tr], y[tr])
            pg = clf.predict(F[te])
            out_rows.append(dict(level=level, site=sp.target, method='gbm_features', seed=seed, **score(y[te], pg, K)))
            preds[f'{level}|{sp.target}|gbm_features|{seed}'] = pg.astype(np.int8)
        preds[f'{level}|{sp.target}|y'] = y[te].astype(np.int8)
        preds[f'{level}|{sp.target}|subject'] = mo.subject.to_numpy()[te]
        print(level, sp.target, out_rows[-4:], flush=True)
pd.DataFrame(out_rows).to_csv(f'{OUT}/baseline_rows.csv', index=False)
np.savez_compressed(f'{OUT}/baseline_preds.npz', **preds)
print('done', round(time.time() - t0), 's')
