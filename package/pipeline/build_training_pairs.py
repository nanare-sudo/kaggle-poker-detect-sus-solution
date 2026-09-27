"""Rebuild full-development anchor and 2000-hand behavior training aggregates.

Inputs are the reconstructed gameplay/action cache and pure mathematical lookup.
No retained tree, historical feature matrix, prediction or routing file is read.
This is a training-input component, not a complete downstream training driver.
"""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
for n in ['POLICY_FIX','POLICY_V2','SURP_STRAT','ENGINE_WK','ENGINE_SZ','ENGINE_CF',
          'AGG_QREPLACE','AGG_EPI','Q2_POOL','Q2_WIN_HEAD']:
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
    STATE=root


def work(table):
    import engine as E
    import aggregate as A
    t=time.time();ht,hp,st=[E.A(n) for n in ['hand_table','hand_phase','hand_started_at']]
    hands=np.flatnonzero((ht==table)&(hp==0));hands=hands[np.argsort(st[hands],kind='stable')]
    assert len(hands)==3000
    result={};schemas={}
    for tag,selected,neural in [('dev_q2nnnoply',hands,True),('dev_w2000',hands[-2000:],False)]:
        E.POLICY_Q2=neural;E.POLICY_Q2_SFX='_q2nnnoply'
        k,h,to,X,names=E.table_pair_hands(selected)
        u,F,fn=A.aggregate_table(k,to,X,names)
        result[tag+'_key']=u;result[tag+'_X']=F.astype(np.float32)
        schemas[tag]=fn
    print('TRAINING PAIR FEATURES',table,round(time.time()-t),flush=True)
    return table,result,schemas


def main(a):
    root=a.root.resolve();init(root)
    required=['act_equity.npy','expect_table.npy']+[n+s+'.npy'
        for n in ['act_surprise','act_paggr','act_pfold'] for s in ['','_q2nnnoply']]
    assert all((root/'cache'/n).exists() for n in required)
    sources={str(p.relative_to(root)):sha(p) for p in (root/'src').glob('*.py')}
    sources['pipeline/chen_order_cpu.json']=sha(root/'pipeline/chen_order_cpu.json')
    sources['driver']=sha(Path(__file__))
    fingerprint=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
    a.out.mkdir(parents=True,exist_ok=True)
    manifest=a.out/'specification.json'
    if manifest.exists():assert json.loads(manifest.read_text())['source_sha256']==fingerprint
    else:manifest.write_text(json.dumps(dict(source_sha256=fingerprint,sources=sources,
        scope='Raw-gameplay pair-training aggregates only, no fitted head weights',public_upload=False),indent=2))
    todo=[]
    for table in range(a.tables):
        target=a.out/f'{table:03d}.npz'
        if target.exists():assert str(np.load(target)['source_sha256'])==fingerprint
        else:todo.append(table)
    from multiprocessing import get_context
    with get_context('spawn').Pool(a.jobs,initializer=init,initargs=(str(root),)) as pool:
        for table,values,schemas in pool.imap(work,todo):
            schema_file=a.out/'feature_names.json'
            if schema_file.exists():assert json.loads(schema_file.read_text())==schemas
            else:schema_file.write_text(json.dumps(schemas))
            np.savez_compressed(a.out/f'{table:03d}.npz',**values,source_sha256=fingerprint)
    print('REQUESTED TRAINING PAIR AGGREGATES COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--tables',type=int,default=400)
    p.add_argument('--jobs',type=int,default=4);main(p.parse_args())
