"""Reconstruct the historical S5 OC selector from gameplay statistics.

The pair-baseline NPZ contains key, b_original and predicted route, with its
own provenance. The original Bothshort/T96/P18n ranking is used, never J50 or
postcompetition scores. Label zeros are not manufactured for OC candidates.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def aligned(keys,wanted):
    assert pd.Index(keys).is_unique
    ix=pd.Index(keys).get_indexer(wanted);assert (ix>=0).all();return ix

def base(cache,baseline,out,statistics):
    zc=np.load(statistics/'cardshare_acc.npz');zq=np.load(statistics/'cardshare_acc_q2.npz')
    zl=np.load(statistics/'llr_acc_c.npz');ptab,loc=zc['ptab'],zc['loc']
    def stats(q,p1,p2):
        a=ptab[p1]*900+loc[p1]*30+loc[p2];b=ptab[p2]*900+loc[p2]*30+loc[p1]
        v=[]
        for z in [zc,zq]:v.append((z['S'][q,a]+z['S'][q,b])/np.sqrt(np.maximum(z['V'][q,a]+z['V'][q,b],1e-9)))
        ll=(zl['S'][q,0,a]+zl['S'][q,0,b])/np.sqrt(np.maximum(zl['Sq'][q,a]+zl['Sq'][q,b],1e-9))
        return v[0],v[1],-(v[0]+v[1])/2+.75*ll,zc['N'][q,a]+zc['N'][q,b]
    allk=np.array([(t,i,j) for t in range(400) for i in range(30) for j in range(i+1,30)])
    inv=np.zeros((400,30),np.int64)
    for p in range(12000):inv[ptab[p],loc[p]]=p
    _,_,ud,nd=stats(0,inv[allk[:,0],allk[:,1]],inv[allk[:,0],allk[:,2]])
    mu,sd=ud[nd>=38].mean(),ud[nd>=38].std()
    ev=pd.read_parquet(cache/'eval_pairs.parquet').sort_values('pair_id').reset_index(drop=True)
    p1,p2=ev.p1.to_numpy(),ev.p2.to_numpy()
    v1,v2,u,n=stats(1,p1,p2)
    z=np.load(baseline);key=ev.plo.to_numpy().astype(np.int64)*12000+ev.phi.to_numpy();ix=aligned(z['key'],key)
    risk=np.clip(pd.Series(z['b_original'][ix]).rank(pct=True).to_numpy()*(1-2e-6)+1e-6,1e-6,1-1e-6)
    ev['z']=v1;ev['z_q2']=v2;ev['zc']=(v1+v2)/2;ev['u']=(u-mu)/sd
    ev['rank']=pd.Series(risk).rank(ascending=False).astype(int).to_numpy();ev['risk_score']=risk
    ev['predicted_behavior']=np.array(['directed_transfer','soft_play','coordinated_isolation'])[z['route'][ix]]
    cols=['pair_id','p1','p2','z','z_q2','zc','u','rank','risk_score','predicted_behavior']
    ev[cols].to_parquet(out/'evp_flags.parquet',index=False)
    (out/'base.json').write_text(json.dumps(dict(mu=mu,sd=sd,pairs=len(ev),baseline=str(baseline),sha256=sha(baseline)),indent=2)+'\n')

def script_run(original,out,variables,args=()):
    env=dict(os.environ,**{name:str(value) for name,value in variables.items()})
    with (out/(original.stem+('_'+str(args[0]) if args else '')+'.log')).open('w') as log:
        subprocess.run([sys.executable,'-B',str(original),*map(str,args)],cwd=out,env=env,
                       stdout=log,stderr=subprocess.STDOUT,check=True)

def zdep_top(cache,out,statistics):
    """First-action dependence conditional on own strength and facing a bet.

    The fitted recipe repeats observations for pairs in overlapping selection
    groups. The multiplicity below preserves that weighting.
    """
    load=lambda n:np.load(cache/(n+'.npy'),mmap_mode='r')
    sp=load('seat_player');ph=load('hand_phase');off=load('act_offsets');street=load('act_street')
    ac=load('act_action');seat=load('act_seat');amt=load('act_amount');tc=load('act_to_call');bb=load('hand_bb')
    h1=load('seat_hole1');h2=load('seat_hole2');pct=np.load(statistics/'llr_acc_c.npz')['pct']
    ev=np.flatnonzero(ph==1);hh=np.repeat(ev,6);pp=sp[ev].ravel();order=np.argsort(pp,kind='stable')
    hh=hh[order];pp=pp[order];pstart=np.searchsorted(pp,np.arange(12001))
    hands=lambda p:hh[pstart[p]:pstart[p+1]]
    fl=pd.read_parquet(out/'evp_flags.parquet');rows=[]
    rng=np.random.default_rng(0)
    flag=fl[(fl.zc<-2.5)&(fl['rank']<=1000)]
    soft=fl[(fl.zc<-1.5)&(fl['rank']<=1500)&~fl.pair_id.isin(flag.pair_id)]
    controls=fl[(fl['rank']<=1000)&(fl.zc>-.5)]
    controls=controls.iloc[rng.choice(len(controls),200,replace=False)]
    random=fl.iloc[rng.choice(len(fl),200,replace=False)]
    multiplicity=pd.concat([flag,soft,controls,random]).groupby('pair_id').size().to_dict()
    for r in fl[fl['rank']<=1500].itertuples(index=False):
        hs=np.intersect1d(hands(r.p1),hands(r.p2));seats=sp[hs]
        i1=(seats==r.p1).argmax(1);i2=(seats==r.p2).argmax(1)
        n=off[hs+1]-off[hs];pos=np.arange(n.sum())-np.repeat(np.cumsum(n)-n,n)
        ai=np.repeat(off[hs],n)+pos;hi=np.repeat(np.arange(len(hs)),n)
        aa=ac[ai];ss=seat[ai];mask=(street[ai]==0)&(aa<6)
        idx=np.flatnonzero(mask)
        _,first=np.unique(hi[idx]*6+ss[idx],return_index=True);ii=idx[first]
        take=(ss[ii]==i1[hi[ii]])|(ss[ii]==i2[hi[ii]]);ii=ii[take]
        # Preserve hand, p1, p2 ordering, including when p1 occupies a later seat.
        ii=ii[np.lexsort((ss[ii]!=i1[hi[ii]],hi[ii]))]
        me=ss[ii];pa=np.where(me==i1[hi[ii]],i2[hi[ii]],i1[hi[ii]])
        c1=h1[hs].astype(np.int64);c2=h2[hs].astype(np.int64)
        high=np.maximum(c1//4,c2//4);low=np.minimum(c1//4,c2//4)
        strength=pct[np.where(c1%4==c2%4,high*13+low,low*13+high)]
        own=strength[hi[ii],me];par=strength[hi[ii],pa]
        aggr=(aa[ii]==3)|(aa[ii]==4)|((aa[ii]==5)&(amt[ai[ii]]>tc[ai[ii]]))
        y=np.where(aa[ii]==0,-1.,np.where(aggr,1.,0.))
        cell=np.minimum((own*5).astype(int),4)*2+(tc[ai[ii]]>bb[hs[hi[ii]]])
        d=pd.DataFrame(dict(cell=cell,par=par,y=y))
        copies=multiplicity.get(r.pair_id,1)
        if copies>1:
            # Preserve both player directions in each repeated hand.
            hand_index=hi[ii]
            expanded=np.concatenate([np.tile(np.flatnonzero(hand_index==v),copies) for v in np.unique(hand_index)])
            d=d.iloc[expanded].reset_index(drop=True)
        groups=d.groupby('cell')
        pc=d.par-groups.par.transform('mean');yc=d.y-groups.y.transform('mean')
        cov=(pc*yc).sum();var=((pc**2)*(yc**2)).sum()
        rows.append(dict(pair_id=r.pair_id,zdep=cov/np.sqrt(var) if var>0 else 0.))
    pd.DataFrame(rows).to_parquet(out/'zdep_top1500.parquet',index=False)

def finish(out):
    fl=pd.read_parquet(out/'evp_flags.parquet')
    a=pd.read_parquet(out/'oc_pair_scores_model.parquet');b=pd.read_parquet(out/'oc_pair_scores_max.parquet')
    ta,tb=a.attrs['thr'],b.attrs['thr']
    strict=(a.s_oc>ta['3e-05'])&(b.s_oc>tb['0.0003'])&(a['rank']<=1500)
    votes=fl[['pair_id','p1','p2','rank','zc','u']].copy()
    np.testing.assert_array_equal(votes.pair_id,a.pair_id);votes['A']=strict
    q=pd.read_parquet(out/'pairstats_phase1.parquet');np.testing.assert_array_equal(q.pair_id,votes.pair_id)
    votes['B']=(q.zsum>3)&(q.z_pf>.5)
    votes=votes.merge(pd.read_parquet(out/'zdep_top1500.parquet'),on='pair_id',how='left',validate='one_to_one')
    votes['C']=(-votes.zc+votes.zdep)/np.sqrt(2)>3
    votes['nvote']=votes.A.astype(int)+votes.B.astype(int)+votes.C.astype(int)
    selected=((votes.nvote>=2)&(votes['rank']<=1500))|((votes.zc<-2.5)&(votes['rank']<=1000))|((votes.u>2.5)&(votes['rank']<=1500))
    votes.to_parquet(out/'votes.parquet',index=False)
    votes.loc[selected,['pair_id','p1','p2']].to_csv(out/'pairs_oc_block.csv',index=False)
    print('OC ROUTE',int(selected.sum()),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['base','detectors','zdep','finish'])
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--pair-baseline',type=Path)
    p.add_argument('--out',type=Path,default=HERE/'oc_routes');a=p.parse_args()
    cache=a.cache.resolve()
    out=a.out.resolve();out.mkdir(exist_ok=True);statistics=HERE/'oc_statistics'
    original=HERE/'source/route_original'
    if a.stage=='base':return base(cache,a.pair_baseline,out,statistics)
    if a.stage=='detectors':
        # The exact accumulator is immutable input to the two original formulas.
        link=out/'acc_bins7.npz'
        if not link.exists():link.symlink_to((statistics/'acc_bins7.npz').resolve())
        variables={'POKER_CACHE':str(cache),'OC_ROUTES':str(out)}
        for mode in ['model','max']:
            script_run(original/'final_detector_v2.py',out,variables,[mode])
        pctfile=out/'llr_acc_c.npz'
        if not pctfile.exists():pctfile.symlink_to((statistics/'llr_acc_c.npz').resolve())
        script_run(original/'detector_all.py',out,variables,[1])
        return
    if a.stage=='zdep':return zdep_top(cache,out,statistics)
    if a.stage=='finish':return finish(out)

if __name__=='__main__':main()
