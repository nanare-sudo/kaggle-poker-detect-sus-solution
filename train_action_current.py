"""Fit original and board-visible neural action references on matching hand folds."""
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['POLICY_FIX']='1'
os.environ['POLICY_V2']='1'
os.environ['Q2_POOL']='0'
os.environ['Q2_WIN_HEAD']='0'
import argparse, ast, hashlib, json, shutil, subprocess, sys, time, traceback, zipfile
from pathlib import Path
import numpy as np

T0=time.time();H=16;N_PLAYERS=12000;NFOLD=2;VAL_FRAC=.05
NAMES=['idx','X','cls','ply','hc','ha','c7','hid','tab','ph']
SMOKE=False;ON_KAGGLE=True;EPOCHS=24;BATCH=16384;LR=.003;VARIANTS=['noply']
WORK=None;OUTD=None

def log(x): print(round(time.time()-T0), x, flush=True)

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def setup(root):
    sys.path.insert(0,str(root/'src'))
    os.environ['POKER_CACHE']=str(root/'cache')
    os.environ['Q2_STATS_FILE']=str(root/'cache/q2_pstats.npz')
    source=root/'assets/neural/policy_nn.py'
    wanted={'_table_feats','build_features','card_block','selftest_cards','make_model','train_device'}
    nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in wanted]
    assert {n.name for n in nodes}==wanted
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),globals())

def splits(root,work):
    hid=np.load(work/'feat_hid.npy',mmap_mode='r')
    n=len(np.load(root/'cache/hand_table.npy',mmap_mode='r'))
    f=np.empty(n,np.int8);f[np.random.default_rng(7).permutation(n)]=(np.arange(n)%2).astype(np.int8)
    v=np.random.default_rng(11).random(n)<.05
    return f[hid],v[hid]

def worker(a):
    global WORK,OUTD
    setup(a.root)
    WORK=str(a.root/a.arm);OUTD=str(a.out/a.arm)
    Path(OUTD).mkdir(exist_ok=True)
    import torch
    torch.set_num_threads(2)
    assert torch.cuda.device_count()==1
    selftest_cards()
    f,v=splits(a.root,Path(WORK));res={}
    train_device('cuda:0',[a.fold],f,v,res)
    errors=[x for k,x in res.items() if k[0]=='FEHLER'];assert not errors,errors
    p=Path(OUTD)/f'model_noply_f{a.fold}.pt'
    state=torch.load(p,map_location='cpu',weights_only=True)
    assert not any('ply' in n for n in state)
    report=dict(arm=a.arm,fold=a.fold,model_sha256=sha(p),source_sha256=sha(__file__),
        training=res[('noply',a.fold)],torch=torch.__version__,numpy=np.__version__)
    (Path(OUTD)/f'fold{a.fold}.json').write_text(json.dumps(report,indent=2))

def repair_inputs(root,out):
    src=root/'RFRESH';dst=root/'RCLEAN';dst.mkdir()
    for n in NAMES:
        if n!='X':os.link(src/f'feat_{n}.npy',dst/f'feat_{n}.npy')
    x=np.load(src/'feat_X.npy',mmap_mode='r')
    y=np.lib.format.open_memmap(dst/'feat_X.npy',mode='w+',dtype=x.dtype,shape=x.shape)
    changed=0;counts={};c7=np.load(src/'feat_c7.npy',mmap_mode='r')
    for b in range(0,len(x),1<<19):
        q=np.asarray(x[b:b+(1<<19)]).copy();street=np.rint(q[:,0]).astype(np.int32)
        assert np.isin(street,[0,1,2,3]).all()
        visible=np.asarray([0,3,4,5])[street]
        assert np.array_equal(visible,(c7[b:b+len(q),2:]>=0).sum(1))
        diff=q[:,10]!=visible;changed+=int(diff.sum())
        for s in range(4):counts[str(s)]=counts.get(str(s),0)+int((diff&(street==s)).sum())
        old=q.copy();q[:,10]=visible;y[b:b+len(q)]=q
        assert np.array_equal(np.delete(old,10,axis=1),np.delete(q,10,axis=1))
        # Future-board poison changes old input, cannot affect corrected feature.
        poison=old.copy();poison[:,10]=5-old[:,10]
        poison[:,10]=np.asarray([0,3,4,5])[np.rint(poison[:,0]).astype(np.int32)]
        assert np.array_equal(poison,q)
    y.flush()
    report=dict(actions=len(x),changed_column_10=changed,changed_by_street=counts,
        unchanged_other_columns=True,visible_card_count_exact=True,future_board_poison_invariant=True,
        old_X_sha256=sha(src/'feat_X.npy'),new_X_sha256=sha(dst/'feat_X.npy'))
    (out/'input_regression.json').write_text(json.dumps(report,indent=2));log(report)

def merge(root,out,arm):
    work=root/arm;dest=out/arm
    idx=np.load(work/'feat_idx.npy');cls=np.load(work/'feat_cls.npy').astype(np.int64)
    phase=np.load(work/'feat_ph.npy');tab=np.load(work/'feat_tab.npy');x=np.load(work/'feat_X.npy',mmap_mode='r')
    p=np.zeros((len(idx),4),np.float32);cover=np.zeros(len(idx),np.uint8)
    rowf,_=splits(root,work)
    for f in range(2):
        rows=np.load(work/f'te_f{f}.npy');assert (rowf[rows]==f).all()
        p[rows]=np.load(work/f'P_noply_f{f}.npy');cover[rows]+=1
    assert (cover==1).all() and np.isfinite(p).all()
    assert np.max(np.abs(p.sum(1)-1))<1e-5
    assert np.array_equal(np.sort(idx),np.arange(len(idx)))
    s=-np.log(np.maximum(p[np.arange(len(p)),cls],1e-7)).astype(np.float32)
    ag=np.zeros(len(idx),np.float32);fo=ag.copy();sur=ag.copy()
    ag[idx]=p[:,0].astype(np.float16).astype(np.float32)
    fo[idx]=p[:,1].astype(np.float16).astype(np.float32);sur[idx]=s
    for name,v in [('paggr',ag),('pfold',fo),('surprise',sur)]:
        np.save(dest/f'act_{name}_q2nnnoply.npy',v)
    ho=np.isin(tab,np.sort(np.random.default_rng(0).permutation(400)[100:140]))
    report=dict(arm=arm,n=len(s),ll_all=float(s.mean()),ll_holdout_tables_transductive=float(s[ho].mean()),
        ll_phase={str(i):float(s[phase==i].mean()) for i in [0,1]},
        ll_street={str(i):float(s[x[:,0]==i].mean()) for i in range(4)},
        note='Out-of-hand predictions; policy saw other hands from heldout tables, not an inductive table split.',
        caches={q.name:sha(q) for q in dest.glob('*.npy')},
        folds=[json.loads((dest/f'fold{f}.json').read_text()) for f in range(2)])
    (dest/'metrics.json').write_text(json.dumps(report,indent=2));log(report)


def coordinate(a):
    global WORK
    setup(a.root);WORK=str(a.root/'RFRESH');Path(WORK).mkdir()
    build_features(list(range(400)))
    repair_inputs(a.root,a.out)
    for arm in ['RFRESH','RCLEAN']:
        jobs=[]
        for f in range(2):
            fh=(a.out/f'{arm}_fold{f}.log').open('w')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(f))
            p=subprocess.Popen([sys.executable,__file__,'--worker','--root',str(a.root),'--out',str(a.out),
                '--arm',arm,'--fold',str(f)],env=env,stdout=fh,stderr=subprocess.STDOUT)
            jobs.append((p,fh))
        for p,fh in jobs:
            code=p.wait();fh.close();assert code==0,(arm,code)
        merge(a.root,a.out,arm)
    report=dict(RFRESH=json.loads((a.out/'RFRESH/metrics.json').read_text()),
        RCLEAN=json.loads((a.out/'RCLEAN/metrics.json').read_text()))
    (a.out/'COMPARISON.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--coordinate',action='store_true')
    p.add_argument('--root',type=Path);p.add_argument('--out',type=Path);p.add_argument('--arm',choices=['RFRESH','RCLEAN'])
    p.add_argument('--fold',type=int,choices=[0,1]);a=p.parse_args()
    if a.worker:worker(a)
    elif a.coordinate:coordinate(a)
    else:p.error('choose --coordinate or --worker, and provide --root and --out')
