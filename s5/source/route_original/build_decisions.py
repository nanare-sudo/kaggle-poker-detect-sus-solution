"""Entscheidungstabelle aller Aktionen von Paarmitgliedern in gemeinsamen Auswertungshaenden.
Gruppen: M = markiert (zc<-2.5 & rank<=1000), S = weich (zc<-1.5 & rank<=1500, ohne M),
C = Kontrolle Top-Risiko (rank<=1000 & zc>-0.5, Stichprobe), R = Zufallspaare (Stichprobe)."""
import numpy as np, pandas as pd, time, os
import os
C=os.environ['POKER_CACHE'].rstrip('/')+'/'
O=os.environ['OC_ROUTES'].rstrip('/')+'/'
OUT=os.environ['OC_CALIBRATION'].rstrip('/')+'/'
L=lambda n: np.load(C+n+'.npy', mmap_mode='r')
t0=time.time()
off=np.asarray(L('act_offsets')); st=L('act_street'); ac=L('act_action'); se=L('act_seat'); amt=L('act_amount'); tcl=L('act_to_call')
pot=L('act_pot_before'); stk=L('act_stack_before'); ato=L('act_amount_to'); npa=L('act_players_active')
pf=L('act_pfold_q2nnnoply'); pg=L('act_paggr_q2nnnoply')
sp=np.asarray(L('seat_player')); h1=np.asarray(L('seat_hole1')).astype(np.int64); h2=np.asarray(L('seat_hole2')).astype(np.int64)
ph=np.asarray(L('hand_phase')); tsa=np.asarray(L('hand_started_at')); btn=np.asarray(L('hand_button_seat')); bb=np.asarray(L('hand_bb'))
net=np.asarray(L('seat_net')); sfold=np.asarray(L('seat_folded')); sshow=np.asarray(L('seat_showdown'))
hids=np.load(C+'hand_ids.npy',allow_pickle=True)
pct=np.load(O+'llr_acc_c.npz')['pct']
def cls(c1,c2):
    r1,r2=c1//4,c2//4; s1,s2=c1%4,c2%4
    hi=np.maximum(r1,r2); lo=np.minimum(r1,r2)
    return np.where(s1==s2,hi*13+lo,lo*13+hi)
# Spieler->Haende Index (nur Auswertung)
ev=np.flatnonzero(ph==1)
hh=np.repeat(ev,6); pp=sp[ev].ravel()
o=np.argsort(pp,kind='stable'); pp=pp[o]; hh=hh[o]
pstart=np.searchsorted(pp,np.arange(12001))
def hands_of(p): return hh[pstart[p]:pstart[p+1]]
def shared(p1,p2): return np.intersect1d(hands_of(p1),hands_of(p2))
print('index',time.time()-t0,flush=True)
fl=pd.read_parquet(O+'evp_flags.parquet')
M=fl[(fl.zc<-2.5)&(fl['rank']<=1000)].copy(); M['grp']='M'
S=fl[(fl.zc<-1.5)&(fl['rank']<=1500)&~fl.pair_id.isin(M.pair_id)].copy(); S['grp']='S'
rng=np.random.default_rng(0)
Cc=fl[(fl['rank']<=1000)&(fl.zc>-0.5)]; Cc=Cc.sample(200,random_state=0).copy(); Cc['grp']='C'
Rr=fl[~fl.pair_id.isin(pd.concat([M,S,Cc]).pair_id)].sample(200,random_state=1).copy(); Rr['grp']='R'
sel=pd.concat([M,S,Cc,Rr],ignore_index=True)
print(sel.grp.value_counts().to_dict(),flush=True)
rows=[]
for _,r in sel.iterrows():
    p1,p2=int(r.p1),int(r.p2); hs=shared(p1,p2)
    if len(hs)==0: continue
    seats=sp[hs]; i1=(seats==p1).argmax(1); i2=(seats==p2).argmax(1)
    # Aktionsindizes
    n=off[hs+1]-off[hs]; a0=np.repeat(off[hs],n); k=np.arange(n.sum())-np.repeat(np.cumsum(n)-n,n); ai=a0+k
    hidx=np.repeat(np.arange(len(hs)),n)
    s_=np.asarray(st[ai]); a_=np.asarray(ac[ai]); e_=np.asarray(se[ai]).astype(np.int64)
    # Foldposition je (hand,seat)
    foldpos=np.full((len(hs),6),10**9,dtype=np.int64)
    fm=a_==0; foldpos[hidx[fm],e_[fm]]=k[fm]
    # erste Aggression je (hand,seat,street): pos
    aggr=(a_==3)|(a_==4)|((a_==5)&(np.asarray(amt[ai])>np.asarray(tcl[ai])))
    firstaggr=np.full((len(hs),6,4),10**9,dtype=np.int64)
    am_=aggr
    # rueckwaerts setzen -> minimum
    for j in np.flatnonzero(am_)[::-1]:
        firstaggr[hidx[j],e_[j],s_[j]]=k[j]
    # Anzahl Aggressionen des Partners vor Aktion (ganze Hand)
    memb=(e_==i1[hidx])|(e_==i2[hidx]); memb&=a_<6
    idx=np.flatnonzero(memb)
    if len(idx)==0: continue
    hh_=hidx[idx]; me=e_[idx]; pa=np.where(me==i1[hh_],i2[hh_],i1[hh_])
    strength=pct[cls(h1[hs],h2[hs])]  # (nh,6)
    act=a_[idx]; am=np.asarray(amt[ai[idx]]); tc=np.asarray(tcl[ai[idx]])
    kk=np.where(act==0,0,np.where((act==3)|(act==4)|((act==5)&(am>tc)),2,1))
    p0f=np.asarray(pf[ai[idx]]).astype(np.float64); p0g=np.asarray(pg[ai[idx]]).astype(np.float64)
    df=pd.DataFrame(dict(pair_id=r.pair_id,grp=r.grp,h=hs[hh_],t=tsa[hs[hh_]],k_in_hand=k[idx],
        street=s_[idx],act=act,kcls=kk,ctx=(tc>0).astype(np.int8),to_call=tc,amount=am,amount_to=np.asarray(ato[ai[idx]]),
        pot=np.asarray(pot[ai[idx]]),stack=np.asarray(stk[ai[idx]]),bb=bb[hs[hh_]],nact=np.asarray(npa[ai[idx]]),
        me_seat=me,pa_seat=pa,me_is_p1=(me==i1[hh_]),pos=(me-btn[hs[hh_]])%6,
        s_me=strength[hh_,me],s_pa=strength[hh_,pa],
        pa_folded=foldpos[hh_,pa]<k[idx],
        pa_aggr_street=firstaggr[hh_,pa,s_[idx]]<k[idx],
        pa_aggr_any=(firstaggr[hh_,pa,:].min(1)<k[idx]),
        me_aggr_before=(firstaggr[hh_,me,:].min(1)<k[idx]),
        p0f=p0f,p0g=p0g,net_me=net[hs[hh_],me],net_pa=net[hs[hh_],pa],
        me_show=sshow[hs[hh_],me],pa_show=sshow[hs[hh_],pa]))
    rows.append(df)
D=pd.concat(rows,ignore_index=True)
D['p0']=np.where(D.kcls==0,D.p0f,np.where(D.kcls==2,D.p0g,np.clip(1-D.p0f-D.p0g,1e-4,1)))
D.to_parquet(OUT+'decisions.parquet')
print(D.shape,D.grp.value_counts().to_dict(),time.time()-t0)
