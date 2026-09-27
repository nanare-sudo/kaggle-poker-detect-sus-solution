"""Rebuild classic gameplay/sequence features for labeled-pair instance training."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['SEQ_FEAT']='1'
for n in ['POLICY_FIX','POLICY_V2','SURP_STRAT','ENGINE_WK','ENGINE_SZ','ENGINE_CF','AGG_QREPLACE','AGG_EPI','Q2_POOL','Q2_WIN_HEAD']:
    os.environ[n]='0'
os.environ['AGG_QFRACS']=''
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

STATE=None


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def init(root):
    global STATE
    root=Path(root);sys.path.insert(0,str(root/'src'))
    os.environ['POKER_CACHE']=str(root/'cache')
    os.environ['Q2_STATS_FILE']=str(root/'cache/q2_pstats.npz')
    os.environ['POKER_CHEN_ORDER_FILE']=str(root/'pipeline/chen_order_cpu.json')
    import pandas as pd
    lab=pd.read_parquet(root/'cache/labels.parquet');ev=pd.read_parquet(root/'cache/evidence.parquet')
    keys=lab.plo.to_numpy().astype(np.int64)*12000+lab.phi.to_numpy();order=np.argsort(keys)
    pair_ids=dict(zip(lab.pair_id,keys))
    STATE=dict(keys=keys[order],label=lab.label.to_numpy()[order],
        evidence=set((pair_ids[p],int(h)) for p,h in zip(ev.pair_id,ev.hidx)))


def work(table):
    import engine as E
    import evidence as EV
    import seqwire
    t=time.time();E.POLICY_Q2=False
    ht,hp,st=[E.A(n) for n in ['hand_table','hand_phase','hand_started_at']]
    hands=np.flatnonzero((ht==table)&(hp==0));hands=hands[np.argsort(st[hands],kind='stable')]
    k,h,to,X,names=E.table_pair_hands(hands)
    take=np.isin(k,STATE['keys']);k,h,to,X=k[take],h[take],to[take],X[take]
    order=np.lexsort((to,k));k,h,to,X=k[order],h[order],to[order],X[order]
    keep=EV.evid_keep(names)
    # live() also returns the full symmetric schema for an empty labeled set.
    seq,sn=seqwire.live(hands,k,h)
    matrix=np.concatenate([X[:,keep],seq],1).astype(np.float32)
    columns=[n for n,ok in zip(names,keep) if ok]+sn
    rows=np.searchsorted(STATE['keys'],k);np.testing.assert_array_equal(STATE['keys'][rows],k)
    ypair=STATE['label'][rows];y=np.array([(int(q),int(hh)) in STATE['evidence'] for q,hh in zip(k,h)],bool)
    assert not ((ypair==0)&y).any()
    print('INSTANCE TRAINING FEATURES',table,len(k),round(time.time()-t),flush=True)
    return table,dict(key=k,h=h,torder=to,X=matrix,y=y,ypair=ypair,
        table=np.full(len(k),table,np.int16)),columns


def main(a):
    root=a.root.resolve();init(root);a.out.mkdir(parents=True,exist_ok=True)
    sources={str(p.relative_to(root)):sha(p) for p in (root/'src').glob('*.py')}
    sources['driver']=sha(Path(__file__));sources['chen_order']=sha(root/'pipeline/chen_order_cpu.json')
    sources['sequence_schema']=sha(root/'features/sq_ph_names.json')
    fingerprint=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
    spec=a.out/'specification.json'
    if spec.exists():assert json.loads(spec.read_text())['source_sha256']==fingerprint
    else:spec.write_text(json.dumps(dict(source_sha256=fingerprint,sources=sources,public_upload=False),indent=2))
    todo=[]
    for table in range(a.tables):
        p=a.out/f'{table:03d}.npz'
        if p.exists():assert str(np.load(p)['source_sha256'])==fingerprint
        else:todo.append(table)
    from multiprocessing import get_context
    with get_context('spawn').Pool(a.jobs,initializer=init,initargs=(str(root),)) as pool:
        for table,values,names in pool.imap(work,todo):
            p=a.out/'feature_names.json'
            if p.exists():assert json.loads(p.read_text())==names
            else:p.write_text(json.dumps(names))
            np.savez_compressed(a.out/f'{table:03d}.npz',**values,source_sha256=fingerprint)
    print('REQUESTED INSTANCE TRAINING INPUTS COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--tables',type=int,default=400)
    p.add_argument('--jobs',type=int,default=4);main(p.parse_args())
