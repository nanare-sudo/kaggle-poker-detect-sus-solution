import numpy as np, pandas as pd, time
import os
C=os.environ['POKER_CACHE'].rstrip('/')+'/'
L=lambda n: np.load(C+n+'.npy', mmap_mode='r')
t0=time.time()
off=L('act_offsets'); st=L('act_street'); ac=L('act_action'); se=L('act_seat'); pf=L('act_pfold_q2nnnoply')
sp=L('seat_player'); h1=L('seat_hole1'); h2=L('seat_hole2'); tab=L('hand_table'); ph=L('hand_phase')
EQ=np.load(C+'preflop_equity.npy').astype(np.float64)
# class weights (combos): pairs 6, suited 4, offsuit 12
w=np.zeros(169)
for hi in range(13):
    for lo in range(13):
        if hi==lo: w[hi*13+lo]=6
        elif hi>lo: w[hi*13+lo]=4   # suited above diag: cls=hi*13+lo
        else: w[hi*13+lo]=12        # offsuit: lo*13+hi with lo<hi -> index lo*13+hi where first<second
ev_rand=(EQ*w[None,:]).sum(1)/w.sum()
order=np.argsort(ev_rand); cum=np.cumsum(w[order])/w.sum()
pct=np.empty(169); pct[order]=cum-w[order]/w.sum()/2
def cls(c1,c2):
    r1,r2=c1//4,c2//4; s1,s2=c1%4,c2%4
    hi=np.maximum(r1,r2); lo=np.minimum(r1,r2)
    return np.where(s1==s2, hi*13+lo, lo*13+hi)
# player -> table, local idx
spa=np.asarray(sp).reshape(-1)
u,first=np.unique(spa,return_index=True)
ptab=np.asarray(tab)[first//6].astype(np.int64)
loc=np.zeros(12000,np.int64)
for tb in range(400):
    ps=np.flatnonzero(ptab==tb); loc[ps]=np.arange(len(ps))
assert np.bincount(ptab,minlength=400).max()==30
NB=400*900
S=np.zeros((2,NB)); V=np.zeros((2,NB)); N=np.zeros((2,NB))
H=len(ph); CH=100000
pairs_i=[(i,j) for i in range(6) for j in range(6) if i!=j]
for h0 in range(0,H,CH):
    h1_=min(H,h0+CH); a0,a1=int(off[h0]),int(off[h1_])
    s_=np.asarray(st[a0:a1]); a_=np.asarray(ac[a0:a1]); e_=np.asarray(se[a0:a1]).astype(np.int64); p_=np.asarray(pf[a0:a1]).astype(np.float64)
    hx=np.repeat(np.arange(h0,h1_),np.diff(np.asarray(off[h0:h1_+1])))
    m=(s_==0)&(a_<6)
    idx=np.flatnonzero(m)
    key=(hx[idx]-h0)*6+e_[idx]
    uk,fi=np.unique(key,return_index=True)
    fidx=idx[fi]
    nh=h1_-h0
    pos=np.full((nh,6),np.iinfo(np.int64).max); res=np.zeros((nh,6)); var=np.zeros((nh,6))
    hh=uk//6; ss=uk%6
    pos[hh,ss]=fidx
    fold=(a_[fidx]==0).astype(np.float64); pp=np.clip(p_[fidx],1e-4,1-1e-4)
    res[hh,ss]=fold-pp; var[hh,ss]=pp*(1-pp)
    has=pos<np.iinfo(np.int64).max
    c1=np.asarray(h1[h0:h1_]).astype(np.int64); c2=np.asarray(h2[h0:h1_]).astype(np.int64)
    strength=pct[cls(c1,c2)]-0.5
    pl=np.asarray(sp[h0:h1_]).astype(np.int64)
    tb=ptab[pl]; lc=loc[pl]
    phs=np.asarray(ph[h0:h1_]).astype(np.int64)
    for i,j in pairs_i:
        ok=has[:,i]&has[:,j]&(pos[:,i]<pos[:,j])
        k=tb[ok,i]*900+lc[ok,i]*30+lc[ok,j]
        pz=phs[ok]
        for q in (0,1):
            mm=pz==q
            S[q]+=np.bincount(k[mm],weights=res[ok,i][mm]*strength[ok,j][mm],minlength=NB)
            V[q]+=np.bincount(k[mm],weights=var[ok,i][mm]/12.0,minlength=NB)
            N[q]+=np.bincount(k[mm],minlength=NB)
    if h0%500000==0: print(h0,time.time()-t0,flush=True)
np.savez('cardshare_acc.npz',S=S,V=V,N=N,ptab=ptab,loc=loc,pct=pct)
print('done',time.time()-t0)
