"""Fit the two historical action references from rebuilt gameplay arrays.

This training adapter does not read competition labels. Both models retain
the historical final-board-length input and are retrospective references.
Outputs are isolated; an optional retained model is used only after fitting.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure(root,arm):
    os.environ['POKER_CACHE']=str(root/'cache')
    os.environ['Q2_STATS_FILE']=str(root/'cache/q2_pstats.npz')
    os.environ['POLICY_FIX']='0' if arm=='classic' else '1'
    os.environ['POLICY_V2']='0' if arm=='classic' else '1'
    for n in ['POLICY_Q2','SURP_STRAT','ENGINE_WK','ENGINE_SZ','ENGINE_CF']:
        os.environ[n]='0'
    sys.path.insert(0,str(root/'src'))


def collect(task):
    root,arm,table=task;configure(Path(root),arm)
    import engine as E
    if arm=='q2':import action_context as Q
    ht,hp,st=[E.A(n) for n in ['hand_table','hand_phase','hand_started_at']]
    pieces=[];targets=[]
    for phase in [0,1]:
        hands=np.flatnonzero((ht==table)&(hp==phase));hands=hands[np.argsort(st[hands],kind='stable')]
        if arm=='classic':X,y=E.policy_features(hands)
        else:
            X,y,idx=Q.features(hands)
            # Preserve the fitted reference's retrospective final-board input.
            X[:,10]=np.repeat(E.A('hand_n_board')[hands],np.diff(E.A('act_offsets'))[hands])
        pieces.append(X);targets.append(y)
    return np.concatenate(pieces),np.concatenate(targets)


def main(a):
    import lightgbm as lgb
    t=time.time();root=a.root.resolve();configure(root,a.arm)
    assert lgb.__version__=='4.6.0'
    import engine as E
    if a.arm=='q2':
        import action_context as Q
        assert Q.STATS.exists(),'Build gameplay policy statistics before training'
        names=Q.col_names()
    else:names=E.POLICY_FEATURES
    a.out.mkdir(parents=True,exist_ok=True);path=a.out/f'{a.arm}.txt'
    assert not path.exists(),'Do not overwrite a completed training output'
    rng=np.random.default_rng(0)
    tables=rng.choice(400,140,replace=False) if a.arm=='classic' else np.sort(rng.permutation(400)[:100])
    if a.smoke:tables=tables[:2]
    from multiprocessing import get_context
    with get_context('spawn').Pool(2) as pool:
        parts=list(pool.imap(collect,[(str(root),a.arm,int(t)) for t in tables]))
    X=np.concatenate([x for x,y in parts]);y=np.concatenate([y for x,y in parts]).astype(int);del parts
    rr=rng if a.arm=='classic' else np.random.default_rng(1)
    take=rr.permutation(len(X))[:(5000 if a.smoke else 2000000 if a.arm=='classic' else 1500000)]
    va=take[:len(take)//10];tr=take[len(take)//10:]
    # Q2 historically materialized the same permuted matrix before splitting;
    # indexing here preserves every training/validation row and its order.
    params=dict(objective='multiclass',num_class=4,metric='multi_logloss',learning_rate=.15,
        num_leaves=63,min_child_samples=300,feature_fraction=.9,bagging_fraction=.7,
        bagging_freq=1,lambda_l2=5.,verbose=-1,n_jobs=5 if a.arm=='classic' else 3,seed=42,max_bin=63)
    train=lgb.Dataset(X[tr],y[tr],feature_name=names);valid=lgb.Dataset(X[va],y[va],reference=train)
    print('REFERENCE TRAIN INPUTS',a.arm,len(tr),len(va),X.shape[1],round(time.time()-t),flush=True)
    m=lgb.train(params,train,num_boost_round=5 if a.smoke else 250,valid_sets=[valid],
        callbacks=[lgb.early_stopping(40,verbose=False),lgb.log_evaluation(100)])
    m.save_model(str(path),num_iteration=m.best_iteration)
    p=m.predict(X[va],num_iteration=m.best_iteration)
    report=dict(arm=a.arm,smoke=a.smoke,train_rows=len(tr),valid_rows=len(va),features=names,
        train_tables=tables.tolist(),training_row_indices_sha256=hashlib.sha256(tr.tobytes()).hexdigest(),
        validation_row_indices_sha256=hashlib.sha256(va.tobytes()).hexdigest(),best_iteration=m.best_iteration,
        params=params,source_sha256=sha(__file__),model_sha256=sha(path),lightgbm=lgb.__version__,numpy=np.__version__,
        logloss=float(-np.log(np.maximum(p[np.arange(len(p)),y[va]],1e-9)).mean()),public_upload=False,
        scope='Action-reference training from rebuilt raw arrays; no evidence, pair, or behavior labels read')
    if a.reference:
        old=lgb.Booster(model_file=str(a.reference));before=old.predict(X[va])
        report['retained_model_comparison']=dict(rows=len(va),exact=bool(np.array_equal(p,before)),
            max_abs=float(np.abs(p-before).max()),reference_sha256=sha(a.reference),
            note='Post-training diagnostic only; no replacement or performance-based parameter choice')
    report['seconds']=time.time()-t
    (a.out/f'{a.arm}.json').write_text(json.dumps(report,indent=2))
    print('REFERENCE TRAIN COMPLETE',{k:v for k,v in report.items() if k not in ['features','train_tables']},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--arm',choices=['classic','q2'],required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reference',type=Path);p.add_argument('--smoke',action='store_true');main(p.parse_args())
