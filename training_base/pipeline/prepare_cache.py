"""Portable raw-gameplay cache stages using fitted action model assets.

Run with a solution root containing src/ and assets/. No historical feature,
prediction, router or submission file is consulted.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

STATE=None


def init(root,stage):
    global STATE
    sys.path.insert(0,str(root/'src'))
    os.environ['POKER_CACHE']=str(root/'cache')
    os.environ['Q2_STATS_FILE']=str(root/'cache/q2_pstats.npz')
    os.environ['POLICY_FIX']='1' if stage=='policy' else '0'
    os.environ['POLICY_V2']='1' if stage=='policy' else '0'
    STATE=dict(root=root,stage=stage)
    if stage=='policy':
        import lightgbm as lgb
        STATE['models']=[lgb.Booster(model_file=str(root/'assets'/name),params={'num_threads':1})
            for name in ['policy_classic.txt','policy_q2.txt']]
    elif stage=='math':
        import equity_apply
        equity_apply._init()


def work(table):
    if STATE['stage']=='math':
        import equity_apply,expect
        return table,equity_apply._work(table),expect._work(table)
    import action_context as Q
    import engine as E
    A=E.A;ht,hp,st=[A(n) for n in ['hand_table','hand_phase','hand_started_at']]
    result=[]
    for phase in [0,1]:
        hands=np.flatnonzero((ht==table)&(hp==phase));hands=hands[np.argsort(st[hands],kind='stable')]
        X,cls,idx=Q.features(hands);X[:,10]=A('hand_n_board')[A('act_hidx')[idx]]
        parts=[]
        for model in STATE['models']:
            p=model.predict(X[:,:model.num_feature()],num_threads=1)
            parts.append(np.column_stack([-np.log(np.maximum(p[np.arange(len(p)),cls],1e-9)),p[:,0],p[:,1]]).astype(np.float32))
        result.append((idx,parts))
    return table,result


def main(a):
    t=time.time();root=a.root.resolve();sys.path.insert(0,str(root/'src'))
    os.environ['POKER_CACHE']=str(root/'cache');os.environ['Q2_STATS_FILE']=str(root/'cache/q2_pstats.npz')
    os.environ['POLICY_FIX']='1' if a.stage=='policy' else '0'
    os.environ['POLICY_V2']='1' if a.stage=='policy' else '0'
    if a.stage=='core':
        assert a.data is not None
        from raw_cache import build
        build(a.data.resolve(),root/'cache');return
    cache=root/'cache';assert cache.exists()
    n=len(np.load(cache/'act_hidx.npy',mmap_mode='r'))
    if a.stage=='math':
        import equity
        equity.build_preflop_table(n_deals=24000000)
        target=cache/'act_equity.npy';assert not target.exists()
        values=np.lib.format.open_memmap(target,mode='w+',dtype=np.float32,shape=(n,));values[:]=np.nan
        total=None
    else:
        import action_context as Q
        if not Q.STATS.exists():Q.build_stats()
        targets={s:[cache/f'{name}{s}.npy' for name in ['act_surprise','act_paggr','act_pfold']] for s in ['', '_q2']}
        assert not any(p.exists() for paths in targets.values() for p in paths)
        values={s:[np.lib.format.open_memmap(p,mode='w+',dtype=np.float32,shape=(n,)) for p in paths] for s,paths in targets.items()}
    from multiprocessing import get_context
    with get_context('spawn').Pool(a.jobs,initializer=init,initargs=(root,a.stage)) as pool:
        for item in pool.imap(work,range(400)):
            table=item[0]
            if a.stage=='math':
                for idx,v in item[1]:values[idx]=v
                total=item[2] if total is None else total+item[2]
            else:
                for idx,parts in item[1]:
                    for suffix,v in zip(['','_q2'],parts):
                        for j in range(3):values[suffix][j][idx]=v[:,j]
            if table%25==0:print(a.stage,table,round(time.time()-t),'seconds',flush=True)
    if a.stage=='math':
        values.flush();prior=total[:,:4].sum(0)/max(total[:,4].sum(),1.)
        expectation=((total[:,:4]+20*prior)/(np.maximum(total[:,4:5],1.)+20)).astype(np.float32)
        np.save(cache/'expect_table.npy',expectation)
    else:
        for group in values.values():
            for v in group:v.flush()
    report=dict(stage=a.stage,actions=n,tables=400,seconds=time.time()-t,numpy=np.__version__)
    (root/f'{a.stage}_cache_report.json').write_text(json.dumps(report,indent=2));print(report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['core','math','policy'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--data',type=Path);p.add_argument('--jobs',type=int,default=4)
    main(p.parse_args())
