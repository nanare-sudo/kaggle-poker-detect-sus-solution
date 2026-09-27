"""Current J50 pair components from an isolated gameplay cache and weights.

Keeps the historical anchor for the inherited evidence gate. No labels,
old predictions or old submission are inputs to work().
"""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['NPY_DISABLE_CPU_FEATURES']='X86_V4'
import argparse, json, sys, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
os.environ['POKER_CHEN_ORDER_FILE']=str(ROOT/'pipeline/chen_order_cpu.json')
sys.path.insert(0,str(ROOT/'src'))
import pair_inference as L
CURRENT=None


def init():
    global CURRENT
    import lightgbm as lgb
    L.init()
    recipe=L.STATE['spec'];names=recipe['tabicl_features192']
    assert len(names)==192 and names[:96]==recipe['tabicl_features']
    CURRENT=dict(names=names,models={arm:[lgb.Booster(model_file=str(ROOT/f'assets/{arm}/seed{s}.txt'))
                                        for s in range(42,47)] for arm in ['bothshort','rclean']})
    for models in CURRENT['models'].values():
        for model in models:assert model.feature_name()==L.STATE['anchor_names']


def work(table):
    import engine as E
    import aggregate as A
    import derive
    started=time.time()
    _,old=L.work(table)
    ht,hp,st=[E.A(n) for n in ['hand_table','hand_phase','hand_started_at']]
    hs=np.flatnonzero((ht==table)&(hp==1));hs=hs[np.argsort(st[hs],kind='stable')]
    E.POLICY_Q2=True;E.POLICY_Q2_SFX='_q2nnnoply'
    saved={stem:E.A(f'act_{stem}_q2nnnoply') for stem in ['surprise','paggr','pfold']}
    result=dict(old)
    try:
        for arm in ['bothshort','rclean']:
            for stem in saved:
                E._ARR[f'act_{stem}_q2nnnoply']=(saved[stem] if arm=='bothshort' else
                    np.load(ROOT/f'cache/act_{stem}_rclean.npy',mmap_mode='r'))
            k,h,to,x,names=E.table_pair_hands(hs)
            u,F,fn=A.aggregate_table(k,to,x,names)
            use=np.isin(u,L.STATE['want']);keys=u[use];F=F[use]
            np.testing.assert_array_equal(keys,old['key'])
            mat=np.nan_to_num(np.asarray(L.reduced(F,fn,'anchor'),np.float32),nan=0,posinf=0,neginf=0)
            result[arm+'_seeds']=np.stack([m.predict(mat,num_threads=1) for m in CURRENT['models'][arm]])
            if arm=='bothshort':
                full,columns=derive.add_ratio_features(F,fn)
                tab=np.nan_to_num(full[:,[columns.index(n) for n in CURRENT['names']]],nan=0,posinf=0,neginf=0).astype(np.float32)
                np.testing.assert_array_equal(tab[:,:96],old['tabicl'])
                result['tabicl192']=tab
    finally:
        for stem,value in saved.items():E._ARR[f'act_{stem}_q2nnnoply']=value
    print('CURRENT PAIR COMPONENTS',table,len(old['key']),round(time.time()-started),flush=True)
    return table,result


def main(args):
    out=ROOT/'current_pairs';out.mkdir(exist_ok=True)
    tables=list(range(400)) if args.tables=='all' else [int(x) for x in args.tables.split(',')]
    assert all(0<=t<400 for t in tables)
    if args.jobs==1:
        init()
        for table in tables:
            path=out/f'{table:03d}.npz';assert not path.exists()
            _,values=work(table);np.savez_compressed(path,**values)
    else:
        from multiprocessing import get_context
        with get_context('spawn').Pool(args.jobs,initializer=init) as pool:
            for table,values in pool.imap_unordered(work,tables):
                path=out/f'{table:03d}.npz';assert not path.exists();np.savez_compressed(path,**values)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--tables',default='0,22');parser.add_argument('--jobs',type=int,default=1)
    args=parser.parse_args();assert 1<=args.jobs<=4;main(args)
