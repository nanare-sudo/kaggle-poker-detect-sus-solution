"""Train the deployed noply architecture without loading retained weights.

The original training functions are extracted from the frozen source, avoiding
its Kaggle-specific directory setup and cleanup. Neural retraining is stochastic;
the separate inference entry point reproduces the retained submission weights.
"""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['POLICY_FIX']='1';os.environ['POLICY_V2']='1'
os.environ['Q2_POOL']='0';os.environ['Q2_WIN_HEAD']='0'
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
import traceback
import numpy as np

T0=time.time();H=16;N_PLAYERS=12000;NFOLD=2;VAL_FRAC=.05
NAMES=['idx','X','cls','ply','hc','ha','c7','hid','tab','ph']
SMOKE=False;ON_KAGGLE=True;EPOCHS=24;BATCH=16384;LR=.003;VARIANTS=['noply']
WORK=None;OUTD=None


def log(value):print(round(time.time()-T0),value,flush=True)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(a):
    global SMOKE,EPOCHS,BATCH,WORK,OUTD
    root=a.root.resolve();cache=root/'cache';source=root/'assets/neural/policy_nn.py'
    sys.path.insert(0,str(root/'src'));os.environ['POKER_CACHE']=str(cache)
    os.environ['Q2_STATS_FILE']=str(cache/'q2_pstats.npz')
    import action_context as Q
    assert Q.STATS.exists(),'Prepare raw gameplay statistics before training'
    SMOKE=a.smoke;EPOCHS=1 if SMOKE else 24;BATCH=4096 if SMOKE else 16384
    WORK=str(a.out.resolve()/'features');OUTD=str(a.out.resolve()/'models')
    assert not a.out.exists(),'Use a fresh training directory to avoid stale partial features'
    Path(WORK).mkdir(parents=True);Path(OUTD).mkdir()
    wanted={'_table_feats','build_features','card_block','selftest_cards','make_model','train_device'}
    tree=ast.parse(source.read_text())
    definitions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted]
    assert {n.name for n in definitions}==wanted
    exec(compile(ast.Module(body=definitions,type_ignores=[]),str(source),'exec'),globals())
    config=dict(epochs=EPOCHS,batch=BATCH,learning_rate=LR,variants=VARIANTS,fold_seed=7,val_seed=11,
        heldout_fraction=VAL_FRAC,history=H,smoke=SMOKE,source_sha256=sha(source),adapter_sha256=sha(__file__),
        reads_retained_model_weights=False,public_upload=False)
    (a.out/'specification.json').write_text(json.dumps(config,indent=2))
    build_features(list(range(2)) if SMOKE else list(range(400)))
    import torch
    torch.set_num_threads(2)
    selftest_cards()
    hid=np.load(f'{WORK}/feat_hid.npy');idx=np.load(f'{WORK}/feat_idx.npy')
    n_hands=len(np.load(cache/'hand_table.npy',mmap_mode='r'))
    hand_fold=np.empty(n_hands,np.int8)
    hand_fold[np.random.default_rng(7).permutation(n_hands)]=(np.arange(n_hands)%NFOLD).astype(np.int8)
    hand_val=np.random.default_rng(11).random(n_hands)<VAL_FRAC
    row_fold,row_isval=hand_fold[hid],hand_val[hid]
    assert len(np.unique(idx))==len(idx)
    if SMOKE:workers=[('cpu',[0,1])]
    else:
        assert torch.cuda.device_count()>=2,'The recorded training recipe uses two CUDA devices'
        workers=[(f'cuda:{f}',[f]) for f in range(2)]
    results={};jobs=[threading.Thread(target=train_device,args=(dev,folds,row_fold,row_isval,results))
        for dev,folds in workers]
    for job in jobs:job.start()
    for job in jobs:job.join()
    errors=[v for k,v in results.items() if k[0]=='FEHLER'];assert not errors,'\n'.join(errors)
    models={}
    for f in range(2):
        path=Path(OUTD)/f'model_noply_f{f}.pt';state=torch.load(path,map_location='cpu',weights_only=True)
        assert not any('ply' in n for n in state)
        p=np.load(f'{WORK}/P_noply_f{f}.npy',mmap_mode='r')
        rows=np.load(f'{WORK}/te_f{f}.npy');assert (row_fold[rows]==f).all()
        assert p.shape==(len(rows),4) and np.isfinite(p).all()
        models[str(f)]=dict(path=str(path),sha256=sha(path),prediction_rows=len(rows),training=results[('noply',f)])
    report=dict(specification=config,models=models,actions=len(idx),torch=torch.__version__,numpy=np.__version__,
        devices=[d for d,fs in workers],seconds=time.time()-T0,
        scope='Neural action model training from reconstructed public gameplay; full downstream retraining remains separate',
        reproducibility='Original optimizer, schedule, sampling and architecture; no byte-identical stochastic retraining claim',
        public_upload=False)
    (a.out/'training.json').write_text(json.dumps(report,indent=2));log('NEURAL REFERENCE TRAINING COMPLETE')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');main(p.parse_args())
