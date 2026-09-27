"""Portable original and RCLEAN action inference; no stored predictions.

The original uses final board count retrospectively; RCLEAN replaces only
that coordinate by the board count observable on the current street.
Both retain the inherited hand-fold/transductive protocol.
"""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='1'
# Q2.features requires the 22-column base before adding its own 22 fields.
# The saved neural builder then deliberately restores final-board count at X[:,10].
os.environ['POLICY_FIX']='1';os.environ['POLICY_V2']='1'
os.environ['Q2_POOL']='0';os.environ['Q2_WIN_HEAD']='0'
import argparse
import ast
import json
from pathlib import Path
import sys
import threading
import time
import traceback
import numpy as np

T0=time.time();H=16;N_PLAYERS=12000;NFOLD=2;SMOKE=False;ON_KAGGLE=True
NAMES=['idx','X','cls','ply','hc','ha','c7','hid','tab','ph']
WORK=None


def log(value):print(round(time.time()-T0),value,flush=True)


def main(a):
    global WORK
    root=a.root.resolve();cache=root/'cache';assets=root/'assets/neural'
    sys.path.insert(0,str(root/'src'));os.environ['POKER_CACHE']=str(cache)
    os.environ['Q2_STATS_FILE']=str(cache/'q2_pstats.npz')
    import action_context as Q
    import engine as E
    # Pin the legacy GPU Chen tie ordering used to train both model variants.
    E.CHEN_PCT=np.load(assets/'legacy_chen_pct.npy')
    assert E.CHEN_PCT.shape==(52,52)
    if not Q.STATS.exists():Q.build_stats()
    tree=ast.parse((assets/'policy_nn.py').read_text())
    wanted={'_table_feats','build_features','card_block','make_model'}
    definitions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted]
    assert {n.name for n in definitions}==wanted
    exec(compile(ast.Module(body=definitions,type_ignores=[]),str(assets/'policy_nn.py'),'exec'),globals())
    WORK=str(root/'neural_features');Path(WORK).mkdir(exist_ok=True)
    build_features(list(range(400)))
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    assert torch.cuda.device_count()>=2,'Exact verified inference uses two Tesla T4 GPUs'
    torch.backends.cudnn.benchmark=True
    hid=np.load(f'{WORK}/feat_hid.npy');idx=np.load(f'{WORK}/feat_idx.npy')
    n_hands=len(np.load(cache/'hand_table.npy',mmap_mode='r'))
    hand_fold=np.empty(n_hands,np.int8)
    hand_fold[np.random.default_rng(7).permutation(n_hands)]=(np.arange(n_hands)%2).astype(np.int8)
    fold=hand_fold[hid]
    arrays={n:np.load(f'{WORK}/feat_{n}.npy',mmap_mode='r') for n in ['X','ph','ply','hc','ha','c7','cls']}
    pred={arm:np.full((len(idx),4),np.nan,np.float32) for arm in ['original','rclean']};errors=[]
    def infer(f):
        try:
            device=torch.device(f'cuda:{f}');torch.cuda.set_device(device)
            for arm in ['original','rclean']:
                name=f'model_noply_f{f}.pt' if arm=='original' else f'model_rclean_f{f}.pt'
                state=torch.load(assets/name,map_location=device,weights_only=True)
                assert not any('ply' in n for n in state)
                model=make_model(torch,nn,F,'noply',state['mu'],state['sd']).to(device)
                model.load_state_dict(state,strict=True);model.eval();rows=np.flatnonzero(fold==f)
                with torch.no_grad():
                    for start in range(0,len(rows),65536):
                        rr=rows[start:start+65536]
                        batch={n:torch.from_numpy(np.ascontiguousarray(v[rr])).to(device) for n,v in arrays.items()}
                        if arm=='rclean':
                            visible=torch.tensor([0,3,4,5],device=device,dtype=batch['X'].dtype)
                            batch['X'][:,10]=visible[batch['X'][:,0].long()]
                        with torch.autocast(device_type='cuda',dtype=torch.float16):logits=model(batch).float()
                        pred[arm][rr]=torch.softmax(logits,-1).cpu().numpy()
                del model,state;torch.cuda.empty_cache()
                log(f'Neural {arm} action fold {f} complete')
        except BaseException:errors.append(traceback.format_exc())
    jobs=[threading.Thread(target=infer,args=(f,)) for f in [0,1]]
    for job in jobs:job.start()
    for job in jobs:job.join()
    assert not errors,'\n'.join(errors)
    assert len(np.unique(idx))==len(idx)
    for arm,suffix in [('original','q2nnnoply'),('rclean','rclean')]:
        probability=pred[arm];assert np.isfinite(probability).all()
        surprise=-np.log(np.maximum(probability[np.arange(len(idx)),arrays['cls'].astype(int)],1e-7)).astype(np.float32)
        ordered=np.empty_like(probability,dtype=np.float16);ordered[idx]=probability.astype(np.float16)
        sur=np.empty_like(surprise);sur[idx]=surprise
        paths=[cache/f'act_{name}_{suffix}.npy' for name in ['surprise','paggr','pfold']]
        assert not any(p.exists() for p in paths)
        for p,value in zip(paths,[sur,ordered[:,0].astype(np.float32),ordered[:,1].astype(np.float32)]):np.save(p,value)
    report=dict(actions=len(idx),torch=torch.__version__,numpy=np.__version__,
        gpu=[torch.cuda.get_device_name(i) for i in [0,1]],seconds=time.time()-T0)
    (root/'neural_cache_report.json').write_text(json.dumps(report,indent=2));log(report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args())
