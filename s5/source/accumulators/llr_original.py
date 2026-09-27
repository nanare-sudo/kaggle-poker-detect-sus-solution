"""OC-Detektor v2: Likelihood-Ratio 'Spieler handelt (teilweise) auf die Karten des Partners'.
Fuer jede Preflop-Entscheidung a des Akteurs i und jeden anderen Sitz k:
  P0(a) = NN-Policy (eigene Karten, spielerbedingt)
  P1(a) = P0 mit Staerke-Logodds des Partners statt der eigenen (Populationsantwort je Kontext)
  LLR = log((1-alpha) P0(a) + alpha P1(a)) - log P0(a)
Akkumuliert je geordnetem Paar (i,k) und Phase. Label-frei."""
import numpy as np, time, sys
import os
C=os.environ['POKER_CACHE'].rstrip('/')+'/'
L=lambda n: np.load(C+n+'.npy', mmap_mode='r')
t0=time.time()
off=L('act_offsets'); st=L('act_street'); ac=L('act_action'); se=L('act_seat'); amt=L('act_amount'); tcl=L('act_to_call')
pf=L('act_pfold_q2nnnoply'); pg=L('act_paggr_q2nnnoply')
sp=L('seat_player'); h1=L('seat_hole1'); h2=L('seat_hole2'); tab=L('hand_table'); ph=L('hand_phase')
EQ=np.load(C+'preflop_equity.npy').astype(np.float64)
w=np.zeros(169)
for hi in range(13):
    for lo in range(13):
        w[hi*13+lo]=6 if hi==lo else (4 if hi>lo else 12)
evr=(EQ*w[None,:]).sum(1)/w.sum(); order=np.argsort(evr); cum=np.cumsum(w[order])/w.sum()
pct=np.empty(169); pct[order]=cum-w[order]/w.sum()/2
NBIN=20
def cls(c1,c2):
    r1,r2=c1//4,c2//4; s1,s2=c1%4,c2%4
    hi=np.maximum(r1,r2); lo=np.minimum(r1,r2)
    return np.where(s1==s2,hi*13+lo,lo*13+hi)
spa=np.asarray(sp).reshape(-1); u,first=np.unique(spa,return_index=True)
ptab=np.asarray(tab)[first//6].astype(np.int64); loc=np.zeros(12000,np.int64)
for tb in range(400):
    ps=np.flatnonzero(ptab==tb); loc[ps]=np.arange(len(ps))
H=len(ph); CH=100000
# ---------- pass 1: population table counts[ctx, bin, class]
cnt=np.zeros((2,NBIN,3))
def chunk(h0,h1_):
    a0,a1=int(off[h0]),int(off[h1_])
    s_=np.asarray(st[a0:a1]); a_=np.asarray(ac[a0:a1])
    m=(s_==0)&(a_<6); idx=np.flatnonzero(m)
    hx=np.repeat(np.arange(h0,h1_),np.diff(np.asarray(off[h0:h1_+1])))[idx]
    e=np.asarray(se[a0:a1])[idx].astype(np.int64)
    act=a_[idx]; am=np.asarray(amt[a0:a1])[idx]; tc=np.asarray(tcl[a0:a1])[idx]
    klass=np.where(act==0,0,np.where((act==3)|(act==4)|((act==5)&(am>tc)),2,1))
    ctx=(tc>0).astype(np.int64)
    c1=np.asarray(h1[h0:h1_]).astype(np.int64); c2=np.asarray(h2[h0:h1_]).astype(np.int64)
    sbin=np.minimum((pct[cls(c1,c2)]*NBIN).astype(np.int64),NBIN-1)   # (nh,6)
    p0f=np.asarray(pf[a0:a1])[idx].astype(np.float64); p0g=np.asarray(pg[a0:a1])[idx].astype(np.float64)
    return hx,e,klass,ctx,sbin,p0f,p0g
for h0 in range(0,H,CH):
    hx,e,klass,ctx,sbin,_,_=chunk(h0,min(H,h0+CH))
    ob=sbin[hx-h0,e]
    np.add.at(cnt,(ctx,ob,klass),1)
P=(cnt+1)/(cnt+1).sum(2,keepdims=True); LP=np.log(P)
print('pass1',time.time()-t0,flush=True); np.save('pop_logp.npy',LP)
ALPHAS=np.array([0.1,0.25,0.5])
NB=400*900
Sllr=np.zeros((2,len(ALPHAS),NB)); Nd=np.zeros((2,NB)); Sq=np.zeros((2,NB))
for h0 in range(0,H,CH):
    h1_=min(H,h0+CH)
    hx,e,klass,ctx,sbin,p0f,p0g=chunk(h0,h1_)
    p0=np.stack([p0f,np.clip(1-p0f-p0g,0,1),p0g],1); p0=np.clip(p0,1e-4,1); p0/=p0.sum(1,keepdims=True)
    lp0=np.log(p0)
    ob=sbin[hx-h0,e]
    pl=np.asarray(sp[h0:h1_]).astype(np.int64)
    phs=np.asarray(ph[h0:h1_]).astype(np.int64)[hx-h0]
    actor=pl[hx-h0,e]
    pa0=p0[np.arange(len(klass)),klass]
    llr_all=np.zeros((len(ALPHAS),len(klass),6)); keys=np.zeros((len(klass),6),np.int64)
    for k in range(6):
        kb=sbin[hx-h0,k]
        lp1=lp0+LP[ctx,kb,:]-LP[ctx,ob,:]
        p1=np.exp(lp1); p1/=p1.sum(1,keepdims=True)
        pa1=p1[np.arange(len(klass)),klass]
        for ai,al in enumerate(ALPHAS):
            llr_all[ai,:,k]=np.log((1-al)*pa0+al*pa1)-np.log(pa0)
        keys[:,k]=ptab[actor]*900+loc[actor]*30+loc[pl[hx-h0,k]]
    oth=np.ones((len(klass),6),bool); oth[np.arange(len(klass)),e]=False
    for ai in range(len(ALPHAS)):
        m_=np.where(oth,llr_all[ai],0).sum(1)/5.0
        cen=llr_all[ai]-m_[:,None]
        for q in (0,1):
            mm=(phs==q)[:,None]&oth
            Sllr[q,ai]+=np.bincount(keys[mm],weights=cen[mm],minlength=NB)
            if ai==0:
                Sq[q]+=np.bincount(keys[mm],weights=(cen[mm])**2,minlength=NB)
                Nd[q]+=np.bincount(keys[mm],minlength=NB)
    if h0%500000==0: print(h0,time.time()-t0,flush=True)
np.savez('llr_acc_c.npz',S=Sllr,N=Nd,Sq=Sq,alphas=ALPHAS,ptab=ptab,loc=loc,pct=pct)
print('done',time.time()-t0)
