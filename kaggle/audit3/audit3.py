"""Audit 3 (CPU): (1) can unlabeled recordings be assigned to sites? (2) picture of raw spectrograms per site/class and
the effect of candidate background removal."""
import glob, os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT = '/kaggle/working'
readme = glob.glob('/kaggle/input/**/DopplerWild/README.md', recursive=True)[0]
root = os.path.dirname(readme)
fs = os.path.join(root, 'fold_splits')
ul = pd.read_csv(os.path.join(fs, 'DopplerWild_unlabeled_tracklist.csv'))
mo = pd.read_csv(os.path.join(fs, 'motionstate_1s_fold3.csv'))
sg = pd.read_csv(os.path.join(fs, 'singlehand_constrained_1s_fold3.csv'))
vr = pd.read_csv(os.path.join(fs, 'velocity_regression_1s_folds3.csv'))
lab = pd.concat([mo[['file_name', 'location']], sg[['file_name', 'location']], vr[['file_name', 'location']]]).drop_duplicates()
print('labeled recordings', lab.file_name.nunique(), 'with >1 location:', (lab.groupby('file_name').location.nunique() > 1).sum())
ul['file_name'] = ul.file_name.astype(np.int64)
lab['file_name'] = lab.file_name.astype(np.int64)
m = ul.merge(lab.drop_duplicates('file_name'), on='file_name', how='left')
print('unlabeled tracks', len(ul), 'recordings', ul.file_name.nunique(), 'split', ul.split.value_counts().to_dict())
print('tracks whose recording has a labeled site:', m.location.notna().sum(), m.location.value_counts(dropna=False).to_dict())
print('duration', ul.duration.describe().round(1).to_dict(), 'uD_length', ul.uD_length.describe().round(0).to_dict())
# recordings: date prefix -> site (sites are collection days?)
lab['date'] = (lab.file_name // 1000000).astype(int)
print('date -> sites (labeled):'); print(lab.groupby('date').location.agg(lambda s: sorted(set(s))).to_string())
ul['date'] = (ul.file_name // 1000000).astype(int)
print('unlabeled dates:'); print(ul.date.value_counts().sort_index().to_string())
date2site = lab.groupby('date').location.agg(lambda s: s.mode().iloc[0] if s.nunique() == 1 else 'multi')
ul['site_by_date'] = ul.date.map(date2site)
print('unlabeled tracks with site inferable by date:', ul.site_by_date.notna().sum(), ul.site_by_date.value_counts(dropna=False).to_dict())
ul.to_csv(OUT + '/unlabeled_sites.csv', index=False)
# pictures
trk_dir = os.path.join(root, 'labeled_tracks_Doppler')
rng = np.random.default_rng(0)
classes = ['normal_walking', 'Dual-hand Load', 'On Wheels']
fig, axes = plt.subplots(4, 9, figsize=(27, 11))
for si, site in enumerate(['A', 'B', 'C', 'D']):
    for ci, c in enumerate(classes):
        sub = mo[(mo.location == site) & (mo.activity_load == c) & (mo.duration > 0.99) & (mo.occlusion_seconds == 0)]
        r = sub.sample(1, random_state=si * 3 + ci).iloc[0]
        tr = np.load(os.path.join(trk_dir, f"{r.file_name}_track_{r.global_id}.npz"))['uD']
        a = int(round(r.t_start_rel * 90)); w = tr[:, a:a + 90]
        bg = np.percentile(tr, 30, axis=1)
        for k, (img, title) in enumerate([(w, 'raw dB'), (w - bg[:, None], 'minus track bg'), (tr[:, :], 'whole track')]):
            ax = axes[si, ci * 3 + k]
            ax.imshow(img, aspect='auto', origin='lower', cmap='viridis')
            ax.set_title(f'{site} {c[:10]} {title}', fontsize=8); ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.savefig(OUT + '/audit_spectrograms.png', dpi=70); plt.close()
# per-site background profiles from whole tracks
fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
prof = {}
for site in ['A', 'B', 'C', 'D']:
    files = mo[mo.location == site].drop_duplicates(['file_name', 'global_id']).sample(40, random_state=0)
    P = []
    for _, r in files.iterrows():
        tr = np.load(os.path.join(trk_dir, f"{r.file_name}_track_{r.global_id}.npz"))['uD']
        P.append(np.percentile(tr, 30, axis=1))
    P = np.array(P); prof[site] = P.mean(0).tolist()
    ax[0].plot(np.arange(256) - 128, P.mean(0), label=site)
    ax[1].plot(np.arange(256) - 128, P.std(0), label=site)
ax[0].set_title('mean per-bin 30th percentile (track background), dB'); ax[1].set_title('std across tracks'); ax[0].legend()
plt.tight_layout(); plt.savefig(OUT + '/audit_background.png', dpi=100); plt.close()
json.dump(prof, open(OUT + '/audit_bg_profiles.json', 'w'))
print('done')
