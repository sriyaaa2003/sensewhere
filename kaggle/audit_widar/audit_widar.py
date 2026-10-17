import os, glob, json, re, collections, time
import numpy as np, pandas as pd
t0=time.time()
root='/kaggle/input'
W='/kaggle/working'
def log(*a): print(*a, flush=True)

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
