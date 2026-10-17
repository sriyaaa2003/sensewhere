import glob, os, shutil, sys
def boot():
    f = glob.glob('/kaggle/input/**/preprocess.py', recursive=True)
    os.makedirs('/tmp/pkg/sensewhere', exist_ok=True)
    for p in glob.glob(os.path.dirname(f[0]) + '/*.py'):
        shutil.copy(p, '/tmp/pkg/sensewhere')
    sys.path.insert(0, '/tmp/pkg')
def find(name):
    r = glob.glob('/kaggle/input/**/' + name, recursive=True)
    if not r:
        raise FileNotFoundError(name)
    return r[0]
