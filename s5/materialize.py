"""Offline S5 component exports and CSV assembly; never contacts Kaggle."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logit
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent/'package'
sys.path.insert(0,str(ROOT/'pipeline'))
from current_finish import collect,align,sha

def baseline(out,components=None,tabicl=None):
    if components is None:
        _,d=collect(ROOT);tabicl=ROOT/'tabicl96_predictions'
    else:d=dict(np.load(components))
    scores=[]
    for j in range(2):
        z=np.load(tabicl/f'context{j}_pred.npz');scores.append(z['prediction'][align(z['key'],d['key'])])
    T=np.mean(scores,axis=0);assert T.dtype==np.float32
    A=d['bothshort_seeds'].mean(1);p=np.clip(d['pool'],1e-9,1-1e-9)
    P=np.log(p/(1-p)).mean(1);l=lambda x:logit(np.clip(x,1e-7,1-1e-7))
    b=.75*(.5*l(A)+.5*l(T))+.25*P
    assert not out.exists();np.savez_compressed(out,key=d['key'],b_original=b,route=d['behavior'].argmax(1))
    print('Baseline export',len(b),sha(out),flush=True)

def reserve():
    start=time.time();dest=ROOT/'reserve_predictions';dest.mkdir(exist_ok=True)
    total=0
    for t in range(400):
        z=np.load(HERE/f'typed_tables/{t:03d}.npz')
        key,first,count=np.unique(z['k'],return_index=True,return_counts=True)
        chosen=np.full((len(key),5),-1,np.int64)
        for j,(a,n) in enumerate(zip(first,count)):
            rows=np.arange(a,a+n)[z['allowed'][a:a+n]]
            pick=rows[np.lexsort((z['to'][rows],-z['new'][rows]))[:5]]
            assert len(pick)==5;chosen[j]=z['h'][pick]
        path=dest/f'{t:03d}.npz'
        if path.exists():
            ref=np.load(path);np.testing.assert_array_equal(ref['key'],key);np.testing.assert_array_equal(ref['h'],chosen)
        else:np.savez(path,key=key,h=chosen)
        total+=len(key)
    assert total==10000
    done=ROOT/'receipts';done.mkdir(exist_ok=True)
    receipt=done/'reserve.json';assert not receipt.exists()
    receipt.write_text(json.dumps(dict(stage='reserve',package_sha256=sha(ROOT/'MANIFEST.json'),
        seconds=time.time()-start,command='s5/materialize.py reserve',pairs=total,
        source='Reserve scores computed live in s5/typed_inference.py using the identical teachers/rankers.',
        public_upload=False),indent=2)+'\n')

def assemble(reserve_csv,oc_scores,out,require_submitted_hash=True):
    original=pd.read_csv(reserve_csv,dtype=str,keep_default_na=False);frame=original.copy()
    pairs=pd.read_parquet(ROOT/'cache/eval_pairs.parquet')
    pair_key=pairs.plo.to_numpy().astype(np.int64)*12000+pairs.phi.to_numpy()
    ix=align(pair_key,(pairs.set_index('pair_id').loc[frame.pair_id].plo.to_numpy().astype(np.int64)*12000+
                       pairs.set_index('pair_id').loc[frame.pair_id].phi.to_numpy()))
    key_to_row=pd.Series(np.arange(len(frame)),index=pair_key[ix]);assert key_to_row.index.is_unique
    hids=np.load(ROOT/'cache/hand_ids.npy',allow_pickle=True);columns=[f'evidence_hand_{j}' for j in range(1,6)]
    seen=[]
    for t in range(400):
        z=np.load(HERE/f'typed_tables/{t:03d}.npz');dest=key_to_row.loc[z['key']].to_numpy()
        frame.loc[dest,columns]=hids[z['picks']];seen.extend(z['key'].tolist())
    assert len(seen)==len(set(seen))==10000
    oc=pd.read_parquet(oc_scores);lookup=pd.Series(np.arange(len(frame)),index=frame.pair_id)
    for pid,g in oc.groupby('pair_id'):
        g=g.sort_values('pick_rank').head(5);assert len(g)==5 and g.hand_id.is_unique
        frame.loc[lookup[pid],columns]=g.hand_id.to_numpy();frame.loc[lookup[pid],'predicted_behavior']='other_coordination'
    np.testing.assert_array_equal(frame.risk_score,original.risk_score)
    assert not out.exists();out.parent.mkdir(exist_ok=True);frame.to_csv(out,index=False)
    report=dict(rows=len(frame),typed_pairs=len(seen),oc_pairs=oc.pair_id.nunique(),
        source_reserve_sha256=sha(reserve_csv),output_sha256=sha(out),
        unchanged_risk=True,expected_sha256='cb25c90604e5c5bc8be323c9b3a83fb12f153f8b69775191467446ff847ffd8b',
        changed_route=int((frame.predicted_behavior!=original.predicted_behavior).sum()),
        changed_evidence_lists=int((frame[columns]!=original[columns]).any(axis=1).sum()))
    report['bytes_exact']=report['output_sha256']==report['expected_sha256']
    (out.parent/'S5_ASSEMBLY.json').write_text(json.dumps(report,indent=2)+'\n')
    if require_submitted_hash:assert report['bytes_exact'],report
    print(report,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['baseline','reserve','assemble'])
    p.add_argument('--out',type=Path);p.add_argument('--components',type=Path);p.add_argument('--tabicl',type=Path)
    p.add_argument('--newly-trained',action='store_true',help='Allow a different CSV hash after actual new fits; never changes retained replay defaults')
    p.add_argument('--reserve',type=Path,default=ROOT/'result/Reserve_J50_risk.csv')
    p.add_argument('--oc',type=Path,default=HERE/'oc/oc_block_norho_R1.parquet');a=p.parse_args()
    if a.stage=='baseline':baseline(a.out or HERE/'pair_baseline.npz',a.components,a.tabicl)
    elif a.stage=='reserve':reserve()
    else:assemble(a.reserve,a.oc,a.out or HERE/'result/S5_J50_risk.csv',not a.newly_trained)
