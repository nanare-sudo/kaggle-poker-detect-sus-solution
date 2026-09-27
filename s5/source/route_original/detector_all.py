"""Paarstatistik fuer ALLE Auswertungspaare (und Dev-Labelpaare): Zahl der Preflop-Ereignisse
wr (Aggression mit s_me<0.8 bei Partner s_pa>=0.8, Partner nicht gefoldet) und pf (Fold mit s_me>=0.8 bei
Partner s_pa<0.6) gegen Policy-Erwartung, plus LLR-Summe der Zellgewichte."""
import numpy as np, pandas as pd, time, sys
import os
C=os.environ['POKER_CACHE'].rstrip('/')+'/'
O=os.environ['OC_ROUTES'].rstrip('/')+'/'
L=lambda n: np.load(C+n+'.npy', mmap_mode='r')
t0=time.time()
off=np.asarray(L('act_offsets')); st=np.asarray(L('act_street')); ac=np.asarray(L('act_action')); se=np.asarray(L('act_seat')).astype(np.int64)
amt=np.asarray(L('act_amount')); tcl=np.asarray(L('act_to_call'))
pf=np.asarray(L('act_pfold_q2nnnoply')); pg=np.asarray(L('act_paggr_q2nnnoply'))
sp=np.asarray(L('seat_player')); h1=np.asarray(L('seat_hole1')).astype(np.int64); h2=np.asarray(L('seat_hole2')).astype(np.int64)
ph=np.asarray(L('hand_phase'))
pct=np.load(O+'llr_acc_c.npz')['pct']
def cls(c1,c2):
    r1,r2=c1//4,c2//4; s1,s2=c1%4,c2%4
    hi=np.maximum(r1,r2); lo=np.minimum(r1,r2)
    return np.where(s1==s2,hi*13+lo,lo*13+hi)
STR=pct[cls(h1,h2)]  # (2M,6)
# nur Preflop-Aktionen vorab: Maske
phase=int(sys.argv[1]) if len(sys.argv)>1 else 1
ev=np.flatnonzero(ph==phase)
hh=np.repeat(ev,6); pp=sp[ev].ravel(); o=np.argsort(pp,kind='stable'); pp=pp[o]; hh=hh[o]
pstart=np.searchsorted(pp,np.arange(12001))
def hands_of(p): return hh[pstart[p]:pstart[p+1]]
print('index',time.time()-t0,flush=True)
if phase==1:
    fl=pd.read_parquet(O+'evp_flags.parquet')[['pair_id','p1','p2','rank','zc']]
else:
    fl=pd.read_parquet(C+'labels.parquet')[['pair_id','p1','p2','label','behavior_family']]
res=np.zeros((len(fl),8))
for i,(p1,p2) in enumerate(zip(fl.p1.values,fl.p2.values)):
    hs=np.intersect1d(hands_of(int(p1)),hands_of(int(p2)))
    if len(hs)==0: continue
    seats=sp[hs]; i1=(seats==p1).argmax(1); i2=(seats==p2).argmax(1)
    n=off[hs+1]-off[hs]; a0=np.repeat(off[hs],n); k=np.arange(n.sum())-np.repeat(np.cumsum(n)-n,n); ai=a0+k
    hidx=np.repeat(np.arange(len(hs)),n)
    s_=st[ai]; a_=ac[ai]; e_=se[ai]
    pre=s_==0
    foldpos=np.full((len(hs),6),10**9,dtype=np.int64); fm=(a_==0)&pre; foldpos[hidx[fm],e_[fm]]=k[fm]
    memb=pre&(a_<6)&((e_==i1[hidx])|(e_==i2[hidx]))
    idx=np.flatnonzero(memb)
    if len(idx)==0: continue
    hh_=hidx[idx]; me=e_[idx]; pa=np.where(me==i1[hh_],i2[hh_],i1[hh_])
    alive=foldpos[hh_,pa]>k[idx]
    sm=STR[hs[hh_],me]; spa=STR[hs[hh_],pa]
    act=a_[idx]; am=amt[ai[idx]]; tc=tcl[ai[idx]]
    aggr=(act==3)|(act==4)|((act==5)&(am>tc)); fold=act==0
    p0g=pg[ai[idx]].astype(np.float64); p0f=pf[ai[idx]].astype(np.float64)
    cwr=alive&(sm<0.8)&(spa>=0.8); cpf=alive&(sm>=0.8)&(spa<0.6)
    # Anti-Zellen (Deficit): Aggression mit starker eigener Hand bei schwachem Partner
    canti=alive&(sm>=0.8)&(spa<0.4)
    res[i]=[len(hs),(cwr&aggr).sum(),p0g[cwr].sum(),(cpf&fold).sum(),p0f[cpf].sum(),(canti&aggr).sum(),p0g[canti].sum(),len(idx)]
    if i%10000==0: print(i,time.time()-t0,flush=True)
out=fl.copy()
for j,c in enumerate(['nh','n_wr','E_wr','n_pf','E_pf','n_anti','E_anti','ndec']): out[c]=res[:,j]
out['z_wr']=(out.n_wr-out.E_wr)/np.sqrt(out.E_wr+0.5); out['z_pf']=(out.n_pf-out.E_pf)/np.sqrt(out.E_pf+0.5)
out['z_anti']=(out.n_anti-out.E_anti)/np.sqrt(out.E_anti+0.5)
out['zsum']=(out.n_wr+out.n_pf-out.E_wr-out.E_pf)/np.sqrt(out.E_wr+out.E_pf+0.5)
out.to_parquet(f'pairstats_phase{phase}.parquet'); print('done',time.time()-t0)
