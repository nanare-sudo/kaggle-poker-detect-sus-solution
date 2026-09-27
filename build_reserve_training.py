"""Current Reserve training inputs directly from gameplay and official labels.

No stored prediction, old feature matrix or old route enters construction.
Teacher targets distinguish listed-positive from confirmed-negative pair hands;
unlisted hands of positive pairs are excluded from teacher fitting. All positive
pair hands remain in the supervised evidence-ranking targets.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
os.environ['NPY_DISABLE_CPU_FEATURES']='X86_V4'
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
import pandas as pd
STATE=None

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def init(root):
    global STATE
    root=Path(root);sys.path.insert(0,str(root/'src'))
    os.environ['POKER_CACHE']=str(root/'cache');os.environ['Q2_STATS_FILE']=str(root/'cache/q2_pstats.npz')
    os.environ['POKER_CHEN_ORDER_FILE']=str(root/'pipeline/chen_order_cpu.json')
    import reserve_features as L
    lab=pd.read_parquet(root/'cache/labels.parquet');ev=pd.read_parquet(root/'cache/evidence.parquet')
    key=lab.plo.to_numpy().astype(np.int64)*12000+lab.phi.to_numpy();assert pd.Index(key).is_unique
    pt=np.full(12000,-1,np.int16);E=L.S.engine
    for seat in range(6):pt[E.A('seat_player')[:,seat]]=E.A('hand_table')
    assert (pt[key//12000]==pt[key%12000]).all()
    pair=dict(zip(lab.pair_id,key));fam=np.array([L.S.G.FAMS.index(f) if y==1 else -1 for f,y in zip(lab.behavior_family,lab.label)])
    STATE=dict(root=root,L=L,keys=key,table=pt[key//12000],positive=dict(zip(key,lab.label==1)),
        family=dict(zip(key,fam)),evidence=set((pair[p],int(h)) for p,h in zip(ev.pair_id,ev.hidx)))

def work(table):
    started=time.time();s=STATE;L=s['L'];want=s['keys'][s['table']==table]
    if not len(want):return table,dict(empty=True),{}
    data,teacher,names=L.live(table,0,want)
    k,h,to,base,nm,R,rn,C,cn,events,en=data
    positive=np.array([s['positive'][q] for q in k]);family=np.array([s['family'][q] for q in k],np.int8)
    y=np.array([(int(q),int(hh)) in s['evidence'] for q,hh in zip(k,h)],np.int8)
    assert not y[~positive].any()
    result=dict(empty=False,key=k,h=h,torder=to,y=y,positive=positive,family=family,**teacher)
    schemas=dict(teacher=names,rank={})
    for g in range(3):
        take=np.flatnonzero(positive&(family==g))
        if not len(take):continue
        for view in range(3):
            ix=take if view==0 else take[to[take]<2000] if view==1 else take[to[take]>=1000]
            if not len(ix):continue
            rr,cc=R[ix],C[ix]
            if view:
                rr,rnn=L.S.G.RF.compute(k[ix],h[ix]);cc,cnn=L.S.G.CI.compute(k[ix],h[ix]);assert rnn==rn and cnn==cn
            X,columns=L.S.G.matrix(k[ix],to[ix],base[ix],nm,rr,rn,cc,cn,events[ix],en,g,'CHRONO_ADD')
            schemas['rank'][str(g)]=columns
            for name,values in [('X',X),('key',k[ix]),('h',h[ix]),('to',to[ix]),('y',y[ix])]:
                result[f'g{g}_v{view}_{name}']=values
    print('CURRENT RESERVE TRAINING',table,len(want),len(k),round(time.time()-started),flush=True)
    return table,result,schemas

def collect(a):
    root=a.root.resolve();sys.path.insert(0,str(root/'src'));import validation as V
    tables=[np.load(a.out/f'tables/{t:03d}.npz') for t in range(400)]
    tables=[(t,z) for t,z in enumerate(tables) if not bool(z['empty'])]
    columns=json.loads((a.out/'features.json').read_text())
    k=np.concatenate([z['key'] for t,z in tables]);to=np.concatenate([z['torder'] for t,z in tables])
    order=np.lexsort((to,k));values={n:np.concatenate([z[n] for t,z in tables])[order] for n in ['key','h','y','positive','family','CLASSIC','NN_EVENTS']}
    table=np.concatenate([np.full(len(z['key']),t,np.int16) for t,z in tables])[order]
    outer=V.table_folds(table,seed=0);nested=a.out/'nested';nested.mkdir()
    np.savez_compressed(nested/'nested_training.npz',key=values['key'],h=values['h'],table=table,outer=outer,
        y=values['y'],eligible=(values['y']==1)|~values['positive'],negative=~values['positive'],
        CLASSIC=values['CLASSIC'],NN_EVENTS=values['NN_EVENTS'])
    params=json.loads((root/'assets/reserve/specification.json').read_text())['teacher_params']
    meta=dict(features=columns['teacher'],instance_params=params,table_outer=V.table_folds(np.arange(400),seed=0).tolist(),
        source='Rebuilt from raw gameplay and official labels',unknowns_used_as_negative=False)
    (nested/'nested_experiment.json').write_text(json.dumps(meta,indent=2))
    (nested/'input_hashes.json').write_text(json.dumps({'nested_training.npz':sha(nested/'nested_training.npz')},indent=2))
    rank=a.out/'rank';window=a.out/'window';rank.mkdir();window.mkdir()
    positive=values['positive'];pk=values['key'][positive];ph=values['h'][positive]
    assert len(pk)==45129 and len(np.unique(pk))==372 and values['y'].sum()==1817
    code=pk*2000000+ph;assert pd.Index(code).is_unique
    np.savez(rank/'identity.npz',key=pk,h=ph,y=values['y'][positive],table=table[positive],family=values['family'][positive])
    rm=dict(features=columns['rank'],files={});wm=dict(features=columns['rank'],files={})
    for g in range(3):
        blocks=[]
        for view in range(3):
            parts=[z for t,z in tables if f'g{g}_v{view}_X' in z]
            part={n:np.concatenate([z[f'g{g}_v{view}_{n}'] for z in parts]) for n in ['key','h','to','X','y']}
            seq=np.lexsort((part['to'],part['key']));part={n:v[seq] for n,v in part.items()}
            ix=pd.Index(code).get_indexer(part['key']*2000000+part['h']);assert (ix>=0).all()
            if view==0:
                path=rank/f'family{g}.npz';np.savez_compressed(path,X=part['X'],row=ix,y=part['y'],train_family=np.ones(len(ix),bool))
                rm['files'][path.name]=sha(path)
            blocks.append(dict(X=part['X'],query=part['key']*3+view,y=part['y'],
                weight=np.full(len(ix),[.5,.25,.25][view],np.float32)))
        query=np.concatenate([b['query'] for b in blocks]);order=np.argsort(query,kind='stable')
        path=window/f'family{g}.npz';np.savez_compressed(path,**{n:np.concatenate([b[n] for b in blocks])[order] for n in blocks[0]})
        wm['files'][path.name]=sha(path)
    rm['files']['identity.npz']=sha(rank/'identity.npz')
    (rank/'evidence_experiment.json').write_text(json.dumps(rm,indent=2));(window/'window_experiment.json').write_text(json.dumps(wm,indent=2))
    print('CURRENT RESERVE TRAINING MATRICES COMPLETE',len(k),len(pk),flush=True)

def main(a):
    if a.stage=='collect':return collect(a)
    tables=list(range(400)) if a.tables=='all' else [int(t) for t in a.tables.split(',')]
    dest=a.out/'tables';dest.mkdir(parents=True,exist_ok=True)
    fn=a.out/'features.json';names=json.loads(fn.read_text()) if fn.exists() else dict(teacher={},rank={})
    todo=[t for t in tables if not (dest/f'{t:03d}.npz').exists()]
    def save(table,values,schemas):
        for arm,features in schemas.items():
            for family,columns in features.items():
                if family in names[arm]:assert names[arm][family]==columns
                names[arm][family]=columns
        fn.write_text(json.dumps(names));np.savez_compressed(dest/f'{table:03d}.npz',**values)
    if a.jobs==1:
        init(a.root)
        for t in todo:save(*work(t))
    else:
        from multiprocessing import get_context
        with get_context('spawn').Pool(a.jobs,initializer=init,initargs=(a.root,)) as pool:
            for result in pool.imap_unordered(work,todo):save(*result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['build','collect']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--tables',default='0,22');p.add_argument('--jobs',type=int,default=1)
    a=p.parse_args();assert 1<=a.jobs<=3;main(a)
