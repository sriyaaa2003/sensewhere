import os, glob, collections
import numpy as np
root='/kaggle/input'
for d in sorted(os.listdir(root)): print('DATASET',d)
# DopplerWild preview
for p in glob.glob(root+'/**/README.md', recursive=True):
    print('=====README',p); print(open(p,encoding='utf-8',errors='ignore').read()[:9000])
for p in glob.glob(root+'/**/labeled_samples.csv', recursive=True):
    print('=====CSV',p); print(open(p).read()[:2500])
npz=sorted(glob.glob(root+'/**/*.npz', recursive=True))
print('npz files',len(npz))
for p in npz[:3]:
    z=np.load(p, allow_pickle=True)
    print(os.path.basename(p), {k:(z[k].shape,str(z[k].dtype)) for k in z.files})
    for k in z.files:
        if z[k].size<10: print('  ',k,z[k])
# Widar BVP structure
mats=sorted(glob.glob(root+'/**/*.mat', recursive=True))
print('mat files',len(mats))
c=collections.Counter()
for p in mats:
    rel=p.split('BVP/')[-1].split('/')
    c[tuple(rel[:2])]+=1
for k,v in sorted(c.items()): print(k,v)
import scipy.io as sio
m=sio.loadmat(mats[0]); print(mats[0], {k:(getattr(v,'shape',None),str(getattr(v,'dtype',''))) for k,v in m.items() if not k.startswith('__')})
