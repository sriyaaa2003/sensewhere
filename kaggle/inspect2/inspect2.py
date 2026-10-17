import os, glob, json, re, collections, time
import numpy as np, pandas as pd
t0=time.time()
root='/kaggle/input'
W='/kaggle/working'
def log(*a): print(*a, flush=True)
# ---------------- DopplerWild
readme=[p for p in glob.glob(root+'/**/DopplerWild/README.md', recursive=True)]
log('README', readme)
if readme:
    txt=open(readme[0],encoding='utf-8',errors='ignore').read()
    log(txt[8800:])
dw_root=os.path.dirname(readme[0]) if readme else None
log('DW top', os.listdir(dw_root) if dw_root else None)
fs=os.path.join(dw_root,'fold_splits')
log('fold_splits', os.listdir(fs))
for f in sorted(os.listdir(fs)):
    if f.endswith('.csv'):
        df=pd.read_csv(os.path.join(fs,f), nrows=3 if 'frame_meta' in f else None)
        log('==',f,df.shape); log(list(df.columns))
m=pd.read_csv(os.path.join(fs,'motionstate_1s_fold3.csv'))
log('motionstate rows',len(m),'subjects',m.global_id.nunique(),'tracks',m.filename_id.nunique())
log(pd.crosstab(m.location,m.activity_load))
log(pd.crosstab(m.location,m.activity_atomic))
log(pd.crosstab(m.fold,m.location))
log(pd.crosstab(m.activity_atomic,m.activity_load))
log('subjects per location', m.groupby('location').global_id.nunique().to_dict())
log('subjects per location x load'); log(m.groupby(['location','activity_load']).global_id.nunique().unstack())
log('windows stride check', (m.t_end-m.t_start).describe().to_dict())
log('occlusion', m.occlusion_seconds.describe().to_dict())
log('close5m', m.close_person_avg_count_5m.describe().to_dict())
# track stats
trk=sorted(glob.glob(dw_root+'/labeled_tracks_Doppler/*.npz'))
log('labeled tracks', len(trk))
rows=[]
loc_of={}
for fn,loc in zip(m.file_name.astype(str),m.location): loc_of.setdefault(fn,loc)
for i,p in enumerate(trk):
    u=np.load(p)['uD']
    fn=os.path.basename(p).split('_track_')[0]
    rows.append(dict(file=os.path.basename(p),loc=loc_of.get(fn),T=u.shape[1],nbin=u.shape[0],mn=float(np.nanmin(u)),mx=float(np.nanmax(u)),mean=float(np.nanmean(u)),std=float(np.nanstd(u)),nan=int(np.isnan(u).sum()),inf=int(np.isinf(u).sum()),zero=float((u==0).mean()),neg=float((u<0).mean()),
        zero_bin_energy=float(np.abs(u[128]).mean()), p50=float(np.percentile(u,50)),p99=float(np.percentile(u,99))))
ts=pd.DataFrame(rows); ts.to_csv(W+'/dw_track_stats.csv',index=False)
log(ts.describe().T.to_string())
log(ts.groupby('loc')[['T','mean','std','p50','p99','zero','neg']].agg(['mean','std']).to_string())
# mean spectrum by location (domain shift fingerprint)
prof={}
for loc in sorted(ts['loc'].dropna().unique()):
    acc=np.zeros(256); n=0
    for p in ts[ts['loc']==loc].file.head(60):
        u=np.load(os.path.join(dw_root,'labeled_tracks_Doppler',p))['uD']
        acc+=np.log1p(np.abs(u)).mean(1); n+=1
    prof[loc]=(acc/n).tolist()
json.dump(prof,open(W+'/dw_loc_profile.json','w'))
m.to_csv(W+'/dw_motionstate.csv',index=False)
# ---------------- Widar BVP manifest
mats=sorted(glob.glob(root+'/**/BVP/**/*.mat', recursive=True))
log('widar mats',len(mats), time.time()-t0)
import scipy.io as sio
rx=re.compile(r'BVP/(\d{8})-VS/([^/]+)/(?:(user\d+)/)?(user\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(.*)\.mat$')
man=[]
bad=0
for i,p in enumerate(mats):
    q=p.replace(chr(92),'/'); r=rx.search(q)
    if not r: bad+=1; continue
    date,grp,_,user,g,loc,ori,rep,rest=r.groups()
    man.append(dict(path=q.split('BVP/')[-1],date=date,group=grp,user=user,gesture=int(g),loc=int(loc),ori=int(ori),rep=int(rep),rest=rest))
mf=pd.DataFrame(man); log('parsed',len(mf),'bad',bad)
full={q.replace(chr(92),'/').split('BVP/')[-1]:q for q in mats}
log(mf.rest.value_counts().head(8))
# shape + value checks on a sample
samp=mf.sample(min(2500,len(mf)),random_state=0)
shp=collections.Counter(); st=[]
for p in samp.path:
    try:
        d=sio.loadmat(full[p])
        a=d['velocity_spectrum_ro']; shp[a.shape]+=1
        st.append((a.shape[2], float(a.min()), float(a.max()), float(np.isnan(a).sum()), float((a.sum(axis=(0,1))>0).sum())))
    except Exception as e: shp['ERR '+str(e)[:40]]+=1
log('shapes',shp.most_common(12))
sa=np.array(st); log('T,min,max,nan,nonzero_frames describe'); log(pd.DataFrame(sa,columns=['T','min','max','nan','nzframes']).describe().to_string())
mf.to_csv(W+'/widar_manifest.csv',index=False)
for c in ['date','group','gesture','loc','ori','user']:
    log(c, mf[c].nunique(), mf[c].value_counts().sort_index().head(30).to_dict())
log(pd.crosstab(mf.date,mf.gesture).to_string())
log(pd.crosstab(mf.date,mf.loc).to_string())
log(pd.crosstab(mf.date,mf.ori).to_string())
log('dups on key', mf.duplicated(['date','user','gesture','loc','ori','rep']).sum())
log('done', time.time()-t0)
