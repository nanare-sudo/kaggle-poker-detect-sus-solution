"""Gebinnte Momentensummen je geordnetem Tischpaar (Akteur i, Partner k) und Phase.
Erste freiwillige Preflop-Entscheidung des Akteurs. Bins: pos (0 = Akteur handelt vor der ersten
freiwilligen Aktion des Partners, 1 = danach), eigene Staerke (3), Partnerstaerke (4).
Groessen: Sf=sum(fold-P0fold), Vf=sum p(1-p), Sa=sum(aggr-P0aggr), Va, N.
Label-frei; erlaubt beliebige lineare Paarstatistiken ohne neuen Durchlauf."""
import numpy as np, time
import os
C=os.environ['POKER_CACHE'].rstrip('/')+'/'
L=lambda n: np.load(C+n+'.npy', mmap_mode='r')
t0=time.time()
off=L('act_offsets'); st=L('act_street'); ac=L('act_action'); se=L('act_seat'); amt=L('act_amount'); tcl=L('act_to_call')
pf=L('act_pfold_q2nnnoply'); pg=L('act_paggr_q2nnnoply')
sp=L('seat_player'); h1=L('seat_hole1'); h2=L('seat_hole2'); tab=L('hand_table'); ph=L('hand_phase')
pct=np.load(os.path.join(os.environ['OC_STATISTICS'],'llr_acc_c.npz'))['pct']
def cls(c1,c2):
    r1,r2=c1//4,c2//4; s1,s2=c1%4,c2%4
    hi=np.maximum(r1,r2); lo=np.minimum(r1,r2)
    return np.where(s1==s2, hi*13+lo, lo*13+hi)
spa=np.asarray(sp).reshape(-1); u,first=np.unique(spa,return_index=True)
ptab=np.asarray(tab)[first//6].astype(np.int64); loc=np.zeros(12000,np.int64)
for tb in range(400):
    ps=np.flatnonzero(ptab==tb); loc[ps]=np.arange(len(ps))
NB=400*900
OWN_EDGES=np.array([0.4,0.7]); PAR_EDGES=np.array([0.3,0.6,0.85])
NO=3; NP=4
ACC=np.zeros((2,2,NO,NP,7,NB))
H=len(ph); CH=100000
pairs_i=[(i,j) for i in range(6) for j in range(6) if i!=j]
for h0 in range(0,H,CH):
    h1_=min(H,h0+CH); a0,a1=int(off[h0]),int(off[h1_])
    s_=np.asarray(st[a0:a1]); a_=np.asarray(ac[a0:a1]); e_=np.asarray(se[a0:a1]).astype(np.int64)
    p_f=np.asarray(pf[a0:a1]).astype(np.float64); p_g=np.asarray(pg[a0:a1]).astype(np.float64)
    am=np.asarray(amt[a0:a1]); tc=np.asarray(tcl[a0:a1])
    hx=np.repeat(np.arange(h0,h1_),np.diff(np.asarray(off[h0:h1_+1])))
    idx=np.flatnonzero((s_==0)&(a_<6))
    key=(hx[idx]-h0)*6+e_[idx]
    uk,fi=np.unique(key,return_index=True); fidx=idx[fi]
    nh=h1_-h0
    pos=np.full((nh,6),np.iinfo(np.int64).max); rf=np.zeros((nh,6)); vf=np.zeros((nh,6)); ra=np.zeros((nh,6)); va=np.zeros((nh,6)); yf=np.zeros((nh,6)); ya=np.zeros((nh,6))
    hh=uk//6; ss=uk%6; pos[hh,ss]=fidx
    act=a_[fidx]; fold=(act==0).astype(np.float64)
    aggr=((act==3)|(act==4)|((act==5)&(am[fidx]>tc[fidx]))).astype(np.float64)
    ppf=np.clip(p_f[fidx],1e-4,1-1e-4); ppg=np.clip(p_g[fidx],1e-4,1-1e-4)
    rf[hh,ss]=fold-ppf; vf[hh,ss]=ppf*(1-ppf); ra[hh,ss]=aggr-ppg; va[hh,ss]=ppg*(1-ppg); yf[hh,ss]=fold; ya[hh,ss]=aggr
    has=pos<np.iinfo(np.int64).max
    c1=np.asarray(h1[h0:h1_]).astype(np.int64); c2=np.asarray(h2[h0:h1_]).astype(np.int64)
    strength=pct[cls(c1,c2)]
    ob=np.searchsorted(OWN_EDGES,strength,side='right'); pb=np.searchsorted(PAR_EDGES,strength,side='right')
    pl=np.asarray(sp[h0:h1_]).astype(np.int64); tb=ptab[pl]; lc=loc[pl]
    phs=np.asarray(ph[h0:h1_]).astype(np.int64)
    for i,j in pairs_i:
        ok=has[:,i]
        after=(has[:,j]&(pos[:,j]<pos[:,i]))
        k=tb[:,i]*900+lc[:,i]*30+lc[:,j]
        for q in (0,1):
            for ps_ in (0,1):
                m=ok&(phs==q)&(after==bool(ps_))
                km=k[m]; o=ob[m,i]; p=pb[m,j]
                flat=((o*NP)+p)*NB+km
                sz=NO*NP*NB
                for qi,arr in enumerate((rf,vf,ra,va,None,yf,ya)):
                    w=None if arr is None else arr[m,i]
                    ACC[q,ps_,:,:,qi,:]+=np.bincount(flat,weights=w,minlength=sz).reshape(NO,NP,NB)
    if h0%500000==0: print(h0,time.time()-t0,flush=True)
np.savez('acc_bins7.npz',ACC=ACC.astype(np.float32),ptab=ptab,loc=loc,pct=pct,OWN_EDGES=OWN_EDGES,PAR_EDGES=PAR_EDGES)
print('done',time.time()-t0)
