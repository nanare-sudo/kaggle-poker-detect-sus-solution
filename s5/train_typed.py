"""Fit two evidence types per family using observed list-order breaks and censored hand targets."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='3'
import argparse,hashlib,json,sys,time
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE/'source'),str(HERE.parent/'package/src')]
import tdlib
import validation as V

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--check-types',type=Path)
    p.add_argument('--out',type=Path,default=HERE/'typed_fits');p.add_argument('--compare',action='store_true')
    a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True);started=time.time()
    z=np.load(a.inputs/'identity.npz');k,h=z['key'],z['h'];fam=z['family'].astype(np.int64)
    names=json.loads((a.inputs/'evidence_experiment.json').read_text())['features']
    # Enrichment acts separately per raw column and pair. The g1 common names
    # are an exact named subset of g0/g2, with the same chronological transforms.
    common_names=names['1'];Xcommon=np.empty((len(k),len(common_names)),np.float32)
    matrices={};seen=np.zeros(len(k),bool)
    for g in range(3):
        zz=np.load(a.inputs/f'family{g}.npz');X=zz['X'];row=zz['row']
        assert not seen[row].any();seen[row]=True
        idx=pd.Index(names[str(g)]).get_indexer(common_names);assert (idx>=0).all()
        Xcommon[row]=X[:,idx];matrices[g]=(X,row)
        assert np.all(fam[row]==g)
    assert seen.all()
    ts=np.asarray(np.load(a.cache/'hand_started_at.npy',mmap_mode='r')[h]).astype(np.int64)
    labels=pd.read_parquet(a.cache/'labels.parquet');evidence=pd.read_parquet(a.cache/'evidence.parquet')
    labels['key']=labels.plo.astype(np.int64)*12000+labels.phi
    evidence=evidence.merge(labels[['pair_id','key']],on='pair_id',validate='many_to_one')
    lookup=dict(zip(zip(evidence.key,evidence.hidx),evidence.evidence_rank))
    erank=np.array([lookup.get((q,hh),0) for q,hh in zip(k,h)])
    np.testing.assert_array_equal(erank>0,z['y']==1)
    keys,first,count=np.unique(k,return_index=True,return_counts=True)
    assert np.all(np.diff(k)>=0);blk=np.zeros(len(k),np.int8);two=np.zeros(len(keys),bool)
    for i,(start,n) in enumerate(zip(first,count)):
        rows=np.arange(start,start+n);li=rows[erank[rows]>0];li=li[np.argsort(erank[li])]
        desc=np.flatnonzero(np.diff(ts[li])<0)
        if len(desc):
            cut=desc[0]+1;two[i]=True;blk[li[:cut]]=1;blk[li[cut:]]=2
    Xt=np.concatenate([Xcommon,fam[:,None]],1);assert Xt.dtype==np.float64
    folds=V.table_folds(z['table'],seed=0);pt=np.full(len(k),np.nan);trmask=blk>0
    params=dict(tdlib.PB,n_jobs=3,seed=42)
    for f in range(5):
        train=trmask&(folds!=f);test=folds==f
        model=lgb.train(params,lgb.Dataset(Xt[train],(blk[train]==1).astype(int)),300)
        path=out/f'type_assignment_fold{f}.txt';assert not path.exists();model.save_model(path)
        pt[test]=model.predict(Xt[test],num_threads=3)
    typ=blk.copy()
    for i,(start,n) in enumerate(zip(first,count)):
        if two[i]:continue
        rows=np.arange(start,start+n);li=rows[erank[rows]>0];li=li[np.argsort(erank[li])]
        prob=np.clip(pt[li],1e-3,1-1e-3)
        best=max(range(len(li)+1),key=lambda j:np.log(prob[:j]).sum()+np.log(1-prob[j:]).sum())
        typ[li[:best]]=1;typ[li[best:]]=2
    np.savez_compressed(out/'type_assignments.npz',key=k,h=h,typ=typ,p_type1=pt,fold=folds,erank=erank)
    check={}
    if a.check_types:
        ref=np.load(a.check_types)
        for name,value in [('key',k),('h',h),('typ',typ),('p_type1',pt)]:
            np.testing.assert_array_equal(value,ref[name],err_msg=name)
        check['type_assignment_exact']=True
    y1,w1,y2,w2=tdlib.build_labels(first,count,erank,ts,typ)
    print('TYPED TARGETS',int(y1.sum()),int(y2.sum()),int((w1==0).sum()),int((w2==0).sum()),check,flush=True)
    del Xt,Xcommon
    records=[]
    for g,(X,row) in matrices.items():
        for tag,y,w in [('t1',y1,w1),('t2',y2,w2)]:
            selected=w[row]>0;dataset=lgb.Dataset(X[selected],y[row][selected],free_raw_data=False)
            for seed in [42,43,44]:
                model=lgb.train(dict(tdlib.PB,seed=seed,n_jobs=3),dataset,500)
                path=out/f'{tag}_g{g}_s{seed}.txt';assert not path.exists();model.save_model(path)
                record=dict(file=path.name,sha256=sha(path))
                if a.compare:
                    ref=HERE/'assets/typed'/path.name
                    record.update(reference_sha256=sha(ref),bytes_exact=path.read_bytes()==ref.read_bytes())
                    assert record['bytes_exact'],path.name
                records.append(record)
        print('TYPED FIT family',g,'six models fitted',round(time.time()-started),flush=True)
    report=dict(seconds=time.time()-started,pairs=len(keys),hands=len(k),two_block_lists=int(two.sum()),
        labels=dict(type1=int(y1.sum()),type2=int(y2.sum()),censored1=int((w1==0).sum()),censored2=int((w2==0).sum())),
        input_hashes={p.name:sha(p) for p in a.inputs.glob('*.npz')},models=records,**check)
    (out/'MODEL_CHECK.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
