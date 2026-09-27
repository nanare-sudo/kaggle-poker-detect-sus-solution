"""Fit Bothshort/RCLEAN from the reconstructed views with a shared seed42 Dataset.

Starts at raw-derived training views; saved weights are optional *post-fit*
comparison targets and are never read by the fit or feature construction.
"""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import argparse,ast,hashlib,json,re,time
import numpy as np
import lightgbm as lgb

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for b in iter(lambda:stream.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def main(a):
    root=a.root.resolve();source=a.inputs.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    names=json.loads((root/'assets/anchor/feature_names.json').read_text())
    parts=[np.load(source/f'{t:03d}.npz') for t in range(400)]
    key=np.concatenate([z['key'] for z in parts]);order=np.argsort(key,kind='stable')
    assert len(key)==len(np.unique(key))==137739
    destination=np.empty(len(key),np.int64);destination[order]=np.arange(len(key))
    y=np.concatenate([z['y'] for z in parts])[order];truth=np.tile((y==1).astype(int),2)
    assert (y==1).sum()==372 and (y==0).sum()==1488
    matrix=out/'train.npy'
    if not matrix.exists():
        X=np.lib.format.open_memmap(matrix,mode='w+',dtype=np.float32,shape=(2*len(key),len(names)))
        start=0
        for z in parts:
            ix=destination[start:start+len(z['key'])];start+=len(ix)
            for view,offset in [('first',0),('last',len(key))]:X[ix+offset]=z[f'{a.arm}_{view}']
        X.flush();del X
        np.savez(out/'identity.npz',key=key[order],y=y)
    identity=np.load(out/'identity.npz');np.testing.assert_array_equal(identity['key'],key[order]);np.testing.assert_array_equal(identity['y'],y)
    X=np.load(matrix,mmap_mode='r')
    tree=ast.parse((root/'src/model.py').read_text())
    defs=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PARAMS' for t in n.targets)]
    assert len(defs)==1;scope={};exec(compile(ast.Module(body=defs,type_ignores=[]),'model.PARAMS','exec'),scope)
    params=dict(scope['PARAMS'],n_jobs=2)
    seeds=[int(s) for s in a.seeds.split(',')];assert set(seeds)<=set(range(42,47))
    report=dict(arm=a.arm,rows=len(truth),pairs=len(key),confirmed_positive=int((y==1).sum()),confirmed_negative=int((y==0).sum()),
        unknown=int((y<0).sum()),unknown_target='PU zero approximation',matrix_sha256=sha(matrix),
        names=names,params=params,dataset_seed=42,rounds=500,seeds=seeds,lightgbm=lgb.__version__,numpy=np.__version__,results=[])
    (out/'CONTRACT.json').write_text(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))
    data=lgb.Dataset(X,truth,feature_name=names,params=dict(params,seed=42),free_raw_data=False);data.construct()
    start=time.time()
    for seed in seeds:
        path=out/f'seed{seed}.txt';assert not path.exists()
        model=lgb.train(dict(params,seed=seed),data,num_boost_round=500);model.save_model(str(path))
        result=dict(seed=seed,sha256=sha(path),seconds=time.time()-start)
        if a.compare:
            target=root/f'assets/{"bothshort" if a.arm=="original" else "rclean"}/seed{seed}.txt'
            normalize=lambda s:re.sub(r'^\[num_threads: \d+\]$','[num_threads: normalized]',s,flags=re.MULTILINE)
            result['exact_trees_and_parameters_except_thread_count']=normalize(path.read_text())==normalize(target.read_text())
            result['comparison_sha256']=sha(target)
        report['results'].append(result);(out/'REPRODUCTION.json').write_text(json.dumps(report,indent=2)+'\n')
        print('CURRENT FRESH FIT',a.arm,result,flush=True)
        if a.compare:assert result['exact_trees_and_parameters_except_thread_count']

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('arm',choices=['original','RCLEAN']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--seeds',default='42,43,44,45,46')
    p.add_argument('--compare',action='store_true');main(p.parse_args())
