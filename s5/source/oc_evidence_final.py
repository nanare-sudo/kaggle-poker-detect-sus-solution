"""Preflop OC hand scores and chronological list decoder.

Scores use aggression and folds relative to partner strength and action-reference
probabilities. Bin weights compare detector-selected and random pair mixtures;
they are heuristic event weights, not calibrated collusion probabilities.
The decoder assumes type-prioritized chronological lists capped at five events.
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd

C=''
O=''  # Assigned by oc_inference before Data() is constructed.
HERE=os.path.dirname(os.path.abspath(__file__))
L=lambda n: np.load(C+n+'.npy', mmap_mode='r')

def cls(c1,c2):
    r1,r2=c1//4,c2//4; s1,s2=c1%4,c2%4
    hi=np.maximum(r1,r2); lo=np.minimum(r1,r2)
    return np.where(s1==s2,hi*13+lo,lo*13+hi)

class Data:
    def __init__(self):
        t0=time.time()
        self.off=np.asarray(L('act_offsets')); self.st=L('act_street'); self.ac=L('act_action'); self.se=L('act_seat')
        self.amt=L('act_amount'); self.tcl=L('act_to_call'); self.pf=L('act_pfold_q2nnnoply'); self.pg=L('act_paggr_q2nnnoply')
        self.sp=np.asarray(L('seat_player')); h1=np.asarray(L('seat_hole1')).astype(np.int64); h2=np.asarray(L('seat_hole2')).astype(np.int64)
        self.ph=np.asarray(L('hand_phase')); self.tsa=np.asarray(L('hand_started_at'))
        self.hids=np.load(C+'hand_ids.npy',allow_pickle=True)
        pct=np.load(O+'llr_acc_c.npz')['pct']
        self.strength=pct[cls(h1,h2)].astype(np.float32)   # (n_hands,6)
        ev=np.flatnonzero(self.ph==1)
        hh=np.repeat(ev,6); pp=self.sp[ev].ravel()
        o=np.argsort(pp,kind='stable'); self.pp=pp[o]; self.hh=hh[o]
        self.pstart=np.searchsorted(self.pp,np.arange(self.sp.max()+2))
        self.t_load=time.time()-t0
    def hands_of(self,p): return self.hh[self.pstart[p]:self.pstart[p+1]]
    def shared(self,p1,p2): return np.intersect1d(self.hands_of(p1),self.hands_of(p2))

    def decisions(self,p1,p2):
        """Preflop-Entscheidungen beider Mitglieder (Partner nicht gefoldet) in gemeinsamen Auswertungshaenden."""
        hs=self.shared(p1,p2)
        if len(hs)==0: return hs, None
        off=self.off; seats=self.sp[hs]; i1=(seats==p1).argmax(1); i2=(seats==p2).argmax(1)
        n=off[hs+1]-off[hs]; a0=np.repeat(off[hs],n); k=np.arange(n.sum())-np.repeat(np.cumsum(n)-n,n); ai=a0+k
        hidx=np.repeat(np.arange(len(hs)),n)
        s_=np.asarray(self.st[ai]); a_=np.asarray(self.ac[ai]); e_=np.asarray(self.se[ai]).astype(np.int64)
        foldpos=np.full((len(hs),6),10**9,dtype=np.int64); fm=a_==0; foldpos[hidx[fm],e_[fm]]=k[fm]
        am=np.asarray(self.amt[ai]); tc=np.asarray(self.tcl[ai])
        aggr=(a_==3)|(a_==4)|((a_==5)&(am>tc))
        memb=((e_==i1[hidx])|(e_==i2[hidx]))&(a_<6)&(s_==0)
        idx=np.flatnonzero(memb)
        hh_=hidx[idx]; me=e_[idx]; pa=np.where(me==i1[hh_],i2[hh_],i1[hh_])
        firstaggr=np.full((len(hs),6),10**9,dtype=np.int64)
        for j in np.flatnonzero(aggr&(s_==0))[::-1]: firstaggr[hidx[j],e_[j]]=k[j]
        d=pd.DataFrame(dict(hi=hh_,me=me,pa=pa,k=k[idx],act=a_[idx],aggr=aggr[idx],fold=a_[idx]==0,
            s_me=self.strength[hs[hh_],me],s_pa=self.strength[hs[hh_],pa],
            pa_folded=foldpos[hh_,pa]<k[idx],pa_aggr_before=firstaggr[hh_,pa]<k[idx],
            p0f=np.asarray(self.pf[ai[idx]]).astype(np.float64),p0g=np.asarray(self.pg[ai[idx]]).astype(np.float64)))
        return hs, d[~d.pa_folded].reset_index(drop=True)

def load_calib(path):
    c=json.load(open(path))
    out={}
    for key,cat in (('pA','A'),('pB','B'),('pA0','A0')):
        b=np.array(c[key]['bins']); p=np.array(c[key]['prec'])
        p=np.minimum.accumulate(p)    # monoton fallend in p0
        out[cat]=(b,p)
    return out

def prec_lookup(cal,cat,p0):
    b,p=cal[cat]; i=np.clip(np.searchsorted(b,p0,side='right')-1,0,len(p)-1); return p[i]

def pb_less_than(pis,K=5):
    """P(Summe Bernoulli(pis) < K) fuer jedes Praefix (exklusiv) der Reihenfolge: Poisson-Binomial-DP."""
    n=len(pis); out=np.empty(n); dist=np.zeros(K+1); dist[0]=1.0   # dist[j]=P(N=j), j<K; dist[K]=P(N>=K)
    for i in range(n):
        out[i]=dist[:K].sum()
        p=pis[i]; new=np.empty(K+1)
        new[0]=dist[0]*(1-p); new[1:K]=dist[1:K]*(1-p)+dist[:K-1]*p; new[K]=dist[K]+dist[K-1]*p
        dist=new
    return out

def score_pair(D,cal,p1,p2,w,pi_none=0.02,K=5):
    hs,d=D.decisions(p1,p2)
    nh=len(hs)
    if nh==0: return None
    piA=np.zeros(nh); piB=np.zeros(nh); p0A=np.ones(nh); p0B=np.ones(nh); nA=np.zeros(nh,int); nB=np.zeros(nh,int); after=np.zeros(nh,bool)
    if d is not None and len(d):
        gap=d.s_pa.values-d.s_me.values
        catA=d.aggr.values&(gap>=0.2); catA0=d.aggr.values&(np.abs(gap)<0.2)&(d.s_pa.values>=0.6); catB=d.fold.values&(-gap>=0.2)
        pa=np.zeros(len(d)); pa[catA]=prec_lookup(cal,'A',d.p0g.values[catA]); pa[catA0]=prec_lookup(cal,'A0',d.p0g.values[catA0])
        pb=np.zeros(len(d)); pb[catB]=prec_lookup(cal,'B',d.p0f.values[catB])
        hi=d.hi.values
        np.maximum.at(piA,hi,pa); np.maximum.at(piB,hi,pb)
        p0a=np.where(catA|catA0,d.p0g.values,1.0); p0b=np.where(catB,d.p0f.values,1.0)
        np.minimum.at(p0A,hi,p0a); np.minimum.at(p0B,hi,p0b)
        np.add.at(nA,hi,(catA|catA0).astype(int)); np.add.at(nB,hi,catB.astype(int))
        # Ereignis nach Partner-Aggression (Info)
        ev=(catA|catA0|catB)&d.pa_aggr_before.values; np.logical_or.at(after,hi[ev],True)
    pi=np.maximum(np.maximum(piA,piB),pi_none)
    typ=np.where(piA>=piB,'A','B'); typ[(piA<=0)&(piB<=0)]='-'
    t=D.tsa[hs]; order=np.argsort(t,kind='stable'); trank=np.empty(nh,int); trank[order]=np.arange(nh)
    # H1: reine Zeitordnung
    pis_t=pi[order]; h1=np.empty(nh); h1[order]=pb_less_than(pis_t,K)
    # H2: Raise-Block (A und '-') zuerst, dann Fold-Block; H3 umgekehrt
    isB=(typ=='B'); isA=~isB
    def block(first_mask):
        o1=order[first_mask[order]]; o2=order[~first_mask[order]]
        seq=np.concatenate([o1,o2]); out=np.empty(nh); out[seq]=pb_less_than(pi[seq],K); return out
    h2=block(isA); h3=block(isB)
    memb=w[0]*h1+w[1]*h2+w[2]*h3
    oc=pi*memb
    df=pd.DataFrame(dict(hand_id=D.hids[hs],h=hs,t=t,t_rank=trank,t_q=(trank+0.5)/nh,oc_score=oc,pi=pi,typ=typ,
        p0_min=np.where(typ=='B',p0B,np.where(typ=='A',p0A,1.0)),n_evA=nA,n_evB=nB,after_partner_aggr=after,
        m_h1=h1,m_h2=h2,m_h3=h3))
    df=df.sort_values(['oc_score','t'],ascending=[False,True]).reset_index(drop=True)
    df['pick_rank']=np.arange(1,nh+1)
    return df

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pairs',default=None,help='csv/parquet mit pair_id,p1,p2 (Standard: 74 markierte zc<-2.5 & rank<=1000)')
    ap.add_argument('--out',default=os.path.join(HERE,'oc_hand_scores.parquet'))
    ap.add_argument('--calib',default=os.path.join(HERE,'calib.json'))
    ap.add_argument('--w',nargs=3,type=float,default=[0.5,0.25,0.25],help='Gewichte H1 (Zeit), H2 (Raise zuerst), H3 (Fold zuerst)')
    ap.add_argument('--pi_none',type=float,default=0.02)
    a=ap.parse_args()
    t0=time.time()
    if a.pairs is None:
        fl=pd.read_parquet(O+'evp_flags.parquet'); pairs=fl[(fl.zc<-2.5)&(fl['rank']<=1000)][['pair_id','p1','p2']]
    else:
        pairs=pd.read_parquet(a.pairs) if a.pairs.endswith('.parquet') else pd.read_csv(a.pairs)
        pairs=pairs[['pair_id','p1','p2']]
    cal=load_calib(a.calib); D=Data(); w=np.array(a.w)/sum(a.w)
    print(f'Daten geladen {D.t_load:.1f}s, Paare {len(pairs)}',flush=True)
    out=[]; t1=time.time()
    for i,r in enumerate(pairs.itertuples(index=False)):
        df=score_pair(D,cal,int(r.p1),int(r.p2),w,a.pi_none)
        if df is None: continue
        df.insert(0,'pair_id',r.pair_id); out.append(df)
    res=pd.concat(out,ignore_index=True)
    res.to_parquet(a.out)
    dt=time.time()-t1
    print(f'{len(out)} Paare, {len(res)} Haende, Scoring {dt:.1f}s ({dt/max(len(out),1)*1000:.0f} ms/Paar), gesamt {time.time()-t0:.1f}s -> {a.out}')
    return res
