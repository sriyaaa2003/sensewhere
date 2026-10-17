"""Per-track viewing geometry from the dataset's own tracker output (position, velocity): approach angle, speed, range."""
import glob, os
import numpy as np, pandas as pd
OUT = '/kaggle/working'
readme = glob.glob('/kaggle/input/**/DopplerWild/README.md', recursive=True)[0]
fs = os.path.join(os.path.dirname(readme), 'fold_splits')
vr = pd.read_csv(os.path.join(fs, 'velocity_regression_1s_folds3.csv'))
mo = pd.read_csv(os.path.join(fs, 'motionstate_1s_fold3.csv'))
sg = pd.read_csv(os.path.join(fs, 'singlehand_constrained_1s_fold3.csv'))
lab = pd.concat([mo[['file_name', 'global_id', 'location', 'activity_load', 'activity_atomic']], sg[['file_name', 'global_id', 'location', 'activity_load', 'activity_atomic']]]).drop_duplicates(['file_name', 'global_id'])
vr['speed'] = vr.speed_mean.clip(lower=1e-3)
vr['lat_frac'] = vr.v_lateral.abs() / vr.speed          # 0 = straight along the line of sight, 1 = fully across
vr['rad_frac'] = vr.v_radial.abs() / vr.speed
vr['dist'] = np.hypot(vr.x_mean, vr.y_mean)
g = vr.groupby(['file_name', 'global_id']).agg(location=('location', 'first'), n=('speed', 'size'), speed=('speed', 'mean'), lat_frac=('lat_frac', 'mean'),
                                              rad_frac=('rad_frac', 'mean'), dist=('dist', 'mean'), x=('x_mean', 'mean'), y=('y_mean', 'mean')).reset_index()
g = g.merge(lab[['file_name', 'global_id', 'activity_load', 'activity_atomic']], on=['file_name', 'global_id'], how='left')
g.to_csv(OUT + '/track_geometry.csv', index=False)
print(len(g), 'tracks;', g.activity_load.notna().sum(), 'with a motion label')
print(g.groupby('location')[['speed', 'lat_frac', 'rad_frac', 'dist']].describe().T.to_string())
print(g.groupby(['location', 'activity_load'])[['speed', 'lat_frac', 'dist']].mean().round(2).to_string())
