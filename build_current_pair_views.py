"""Rebuild current Bothshort/RCLEAN training views from gameplay/action arrays.

The full-development exposure filter defines the pair population once; both
2000-hand views retain it. The saved final-training schema is explicit. This
does not claim to relearn that schema on outer held-out groups.
"""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['NPY_DISABLE_CPU_FEATURES']='X86_V4'
for n in ['POLICY_FIX','POLICY_V2','SURP_STRAT','ENGINE_WK','ENGINE_SZ','ENGINE_CF','AGG_QREPLACE','AGG_EPI']:
    os.environ[n]='0'
os.environ['AGG_QFRACS']=''
os.environ['POLICY_Q2']='1';os.environ['POLICY_Q2_SFX']='_q2nnnoply'
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
import pandas as pd
STATE=None

def init(root,schema):
    global STATE
    root=Path(root);schema=Path(schema)
    os.environ['POKER_CACHE']=str(root/'cache')
    os.environ['Q2_STATS_FILE']=str(root/'cache/q2_pstats.npz')
    os.environ['POKER_CHEN_ORDER_FILE']=str(root/'pipeline/chen_order_cpu.json')
    sys.path.insert(0,str(root/'src'))
    labels=pd.read_parquet(root/'cache/labels.parquet')
    key=labels.plo.to_numpy().astype(np.int64)*12000+labels.phi.to_numpy()
    assert pd.Index(key).is_unique
    STATE=dict(root=root,keep=np.load(schema/'base_keep.npy'),names=json.loads((schema/'feature_names.json').read_text()),
        labels=dict(zip(key,labels.label.to_numpy())))

def work(table):
    import engine as E
    import aggregate as A
    import derive
    t=time.time();s=STATE;root=s['root']
    hs=np.flatnonzero((E.A('hand_table')==table)&(E.A('hand_phase')==0))
    hs=hs[np.argsort(E.A('hand_started_at')[hs],kind='stable')];assert len(hs)==3000
    E.POLICY_Q2=True;E.POLICY_Q2_SFX='_q2nnnoply'
    old={n:E.A(f'act_{n}_q2nnnoply') for n in ['surprise','paggr','pfold']}
    def raw(selected):
        k,h,to,x,names=E.table_pair_hands(selected)
        return A.aggregate_table(k,to,x,names)
    keys,full,names=raw(hs);keys=keys[full[:,names.index('n_shared')]>=57]
    assert pd.Index(keys).is_unique
    values=dict(key=keys,y=np.array([s['labels'].get(k,-1) for k in keys],np.int8),table=np.full(len(keys),table,np.int16))
    try:
        for arm in ['original','RCLEAN']:
            for name in old:E._ARR[f'act_{name}_q2nnnoply']=(old[name] if arm=='original' else np.load(root/f'cache/act_{name}_rclean.npy',mmap_mode='r'))
            for view,selected in [('first',hs[:2000]),('last',hs[-2000:])]:
                kk,F,fn=raw(selected);assert fn==names
                ix=pd.Index(kk).get_indexer(keys);present=ix>=0
                aligned=np.zeros((len(keys),len(names)),np.float32);aligned[present]=F[ix[present]]
                X,nm=derive.add_ratio_features(aligned[:,s['keep']],[names[j] for j in s['keep']])
                assert nm==s['names']
                values[f'{arm}_{view}']=np.nan_to_num(np.asarray(X,np.float32),nan=0,posinf=0,neginf=0)
                values[f'{arm}_{view}_present']=present
    finally:
        for name,arr in old.items():E._ARR[f'act_{name}_q2nnnoply']=arr
    print('CURRENT TRAINING VIEWS',table,len(keys),round(time.time()-t),flush=True)
    return table,values

def main(a):
    root=a.root.resolve();schema=(a.schema or root/'assets/anchor').resolve()
    tables=list(range(400)) if a.tables=='all' else [int(x) for x in a.tables.split(',')]
    a.out.mkdir(parents=True,exist_ok=True)
    contract=dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        schema={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [schema/'base_keep.npy',schema/'feature_names.json']},
        filter='n_shared>=57 in original complete 3000-hand Dev phase',views=[[0,2000],[1000,3000]],
        label_source='official confirmed pair labels; unknown=-1 retained',
        target='PU approximation unknown->0 only at fit time; same pair weight in each view',
        scope='Current raw-to-training-matrix bridge, frozen final schema; not new outer validation')
    path=a.out/'CONTRACT.json'
    if path.exists():assert json.loads(path.read_text())==contract
    else:path.write_text(json.dumps(contract,indent=2)+'\n')
    todo=[t for t in tables if not (a.out/f'{t:03d}.npz').exists()]
    if a.jobs==1:
        init(root,schema)
        for t in todo:
            table,values=work(t);np.savez_compressed(a.out/f'{table:03d}.npz',**values)
    else:
        from multiprocessing import get_context
        with get_context('spawn').Pool(a.jobs,initializer=init,initargs=(root,schema)) as pool:
            for table,values in pool.imap_unordered(work,todo):np.savez_compressed(a.out/f'{table:03d}.npz',**values)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--schema',type=Path)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--tables',default='0,22');p.add_argument('--jobs',type=int,default=1)
    a=p.parse_args();assert 1<=a.jobs<=3;main(a)
