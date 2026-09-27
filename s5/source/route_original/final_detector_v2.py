"""OC-Paardetektor v3b: wie final_detector.py, aber Varianz des Kontrasts empirisch je Paar und Zeile
(m(1-m) mit m = beobachtete Fold-/Raise-Rate des Paars in der Zeile) statt Policy-Varianz p(1-p).
Grund: Paare mit systematisch abweichender Aktionsrate (CI-Isolierer) blaehen z bei Policy-Varianz auf."""
import numpy as np, pandas as pd, sys, os
MODE=sys.argv[1] if len(sys.argv)>1 else 'max'   # 'emp' | 'max' | 'model'
z=np.load('acc_bins7.npz'); ACC=z['ACC'].astype(np.float64); ptab=z['ptab']; loc=z['loc']
ev=pd.read_parquet(os.path.join(os.environ['OC_ROUTES'],'evp_flags.parquet'))
lab=pd.read_parquet(os.path.join(os.environ['POKER_CACHE'],'labels.parquet'))
def keys(p1,p2): return ptab[p1]*900+loc[p1]*30+loc[p2], ptab[p1]*900+loc[p2]*30+loc[p1]
def cstat(w,qty,q):
    Sr=ACC[q][...,qty,:]; Vr=ACC[q][...,qty+1,:]; N=ACC[q][...,4,:]; Y=ACC[q][...,5 if qty==0 else 6,:]
    Nrow=N.sum(2,keepdims=True); wbar=(w[...,None]*N).sum(2,keepdims=True)/np.maximum(Nrow,1); wc=w[...,None]-wbar
    m=Y.sum(2,keepdims=True)/np.maximum(Nrow,1); vemp=m*(1-m)
    Vmodel=((wc**2)*Vr).sum((0,1,2)); Vemp=((wc**2)*N*vemp).sum((0,1,2))
    V={'emp':Vemp,'model':Vmodel,'max':np.maximum(Vmodel,Vemp)}[MODE]
    return (wc*Sr).sum((0,1,2)), V
parc=np.array([0.15,0.45,0.725,0.925])-0.5; ownmod=np.array([2.0,1.0,1.0])
wf=np.zeros((2,3,4)); wa=np.zeros((2,3,4))
for o in range(3): wf[0,o]=-parc*ownmod[o]; wa[0,o]=parc*ownmod[o]
zdir={}
for q in (0,1):
    Sf,Vf=cstat(wf,0,q); Sa,Va=cstat(wa,2,q)
    zdir[q]=Sf/np.sqrt(np.maximum(Vf,1e-9))+Sa/np.sqrt(np.maximum(Va,1e-9))
allk=np.array([(tb,i,j) for tb in range(400) for i in range(30) for j in range(i+1,30)])
k12=allk[:,0]*900+allk[:,1]*30+allk[:,2]; k21=allk[:,0]*900+allk[:,2]*30+allk[:,1]
N0=ACC[0,:,:,:,4,:].sum((0,1,2)); N1=ACC[1,:,:,:,4,:].sum((0,1,2))
pos_lab=lab[lab.label==1]; kl1,kl2=keys(pos_lab.p1.values,pos_lab.p2.values); labk=np.minimum(kl1,kl2)
nd=N0[k12]+N0[k21]; kmin=np.minimum(k12,k21); isl=np.isin(kmin,labk); okd=(nd>=71)&~isl
d12,d21=zdir[0][k12],zdir[0][k21]; mu,sd=d12[okd].mean(),d12[okd].std()
dsum=((d12-mu)+(d21-mu))/sd/np.sqrt(2)
ke1,ke2=keys(ev.p1.values,ev.p2.values)
e12=(zdir[1][ke1]-mu)/sd; e21=(zdir[1][ke2]-mu)/sd; esum=(e12+e21)/np.sqrt(2)
thr={f:np.quantile(dsum[okd],1-f) for f in (1e-3,3e-4,1e-4,3e-5)}
order=np.argsort(kmin); pos_idx=order[np.searchsorted(kmin[order],labk)]; s_pos=dsum[pos_idx]; fam=pos_lab.behavior_family.values
print(f'MODE {MODE}: dev mu {mu:.3f} sd {sd:.3f}; Schwellen',{f'{f:.0e}':round(t,3) for f,t in thr.items()})
for f in np.unique(fam): print('  ',f,'n',(fam==f).sum(),'mean %.2f sd %.2f max %.2f'%(s_pos[fam==f].mean(),s_pos[fam==f].std(),s_pos[fam==f].max()),'ueber Schwellen',{f'{ff:.0e}':int((s_pos[fam==f]>t).sum()) for ff,t in thr.items()})
for lo,hi in ((71,120),(120,200),(200,300),(300,2000)):
    m=okd&(nd>=lo)&(nd<hi); print(f'   N {lo}-{hi}: q(1-1e-4) {np.quantile(dsum[m],1-1e-4):.2f} q(1-3e-5) {np.quantile(dsum[m],1-3e-5):.2f}')
n15=ev['rank'].values<=1500
for f,t in thr.items():
    m=esum>t
    print(f'   fpr {f:.0e}: eval alle {m.sum()}, Null {f*len(ev):.1f}, Ueberschuss {m.sum()-f*len(ev):.1f}; rank<=1500 {(m&n15).sum()}, 1501-3000 {(m&(ev["rank"].values>1500)&(ev["rank"].values<=3000)).sum()}, >3000 {(m&(ev["rank"].values>3000)).sum()}')
out=ev[['pair_id','p1','p2','rank','zc','u','predicted_behavior']].copy()
out['s12']=e12; out['s21']=e21; out['s_oc']=esum; out['n_eval']=N1[ke1]+N1[ke2]
out['tier']=np.select([esum>thr[3e-5],esum>thr[1e-4],esum>thr[3e-4],esum>thr[1e-3]],[1,2,3,4],0)
out['flag_strict']=(out.tier==1)&(out['rank']<=1500)
out['flag_medium']=(out.tier<=2)&(out.tier>0)&(out['rank']<=1500)
out['flag_outside']=(out.tier==1)&(out['rank']>1500)
out.attrs={'mode':MODE,'thr':{str(k):float(v) for k,v in thr.items()}}
out.to_parquet(f'oc_pair_scores_{MODE}.parquet')
sel=out[out.flag_strict].sort_values('s_oc',ascending=False)
old=ev[(ev.zc<-2.5)&(ev['rank']<=1000)].pair_id
print('   strict',len(sel),'in zc-74',sel.pair_id.isin(old).sum(),'| medium',out.flag_medium.sum(),'| outside',out.flag_outside.sum(),'| beide Richtungen>2',round(((sel.s12>2)&(sel.s21>2)).mean(),3),'| predicted',sel.predicted_behavior.value_counts().to_dict())
