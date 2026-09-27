"""Kalibrierung: Praezision 'gepflanzt' je Handkategorie aus M (74 markierte) gegen R (200 Zufallspaare), Basis decisions.parquet."""
import pandas as pd, numpy as np, json
import os
Q=os.environ['OC_CALIBRATION'].rstrip('/')+'/'
D=pd.read_parquet(Q+'decisions.parquet'); P=D[(D.street==0)&(~D.pa_folded)&(D.act<6)].copy()
aggr=P.kcls==2; fold=P.kcls==0
P['gap_a']=P.s_pa-P.s_me; P['gap_f']=P.s_me-P.s_pa
P['e']=np.where(aggr,-np.log(np.clip(P.p0g,1e-6,1)),np.where(fold,-np.log(np.clip(P.p0f,1e-6,1)),0.0))
P['p0a']=np.where(aggr,P.p0g,np.where(fold,P.p0f,1.0))
# Kategorien je Entscheidung
P['cat']='none'
P.loc[aggr&(P.gap_a>=0.2),'cat']='A'          # Raise auf Partnerkarten (Richtung)
P.loc[fold&(P.gap_f>=0.2),'cat']='B'          # Premium-Fold bei schwachem Partner
P.loc[aggr&(P.gap_a<0.2)&(P.gap_a>=-0.2)&(P.s_pa>=0.6),'cat']='A0'   # Raise, Partner stark, kein Gefaelle
P.loc[aggr&(P.gap_a<-0.2),'cat']='Aneg'        # Raise gegen Richtung (eigene staerker)
P.loc[fold&(P.gap_f<0.2),'cat']='Bneg'
# Handebene: staerkste Entscheidung je Hand (max e) innerhalb Kategorie A/B/A0
def hand_agg(g):
    best={}
    for c in ['A','B','A0','Aneg']:
        x=g[g.cat==c]
        best[c]=x.p0a.min() if len(x) else 1.0
    # Raise-War: beide Mitglieder aggressiv
    both=(g[g.kcls==2].me_seat.nunique()==2)
    return pd.Series(dict(pA=best['A'],pB=best['B'],pA0=best['A0'],pAneg=best['Aneg'],both=both,minp0_aggr=g[g.kcls==2].p0g.min() if (g.kcls==2).any() else 1.0))
H=P.groupby(['grp','pair_id','h']).apply(hand_agg).reset_index()
# Handzahl je Gruppe (alle gemeinsamen Haende, nicht nur mit Preflop-Entscheidung): aus D
nh=D.groupby('grp').h.nunique(); npair=D.groupby('grp').pair_id.nunique()
print('Haende/Paar',(nh/npair).round(1).to_dict())
bins=[0,0.02,0.05,0.1,0.2,0.35,0.6,1.01]
lab=[f'[{bins[i]},{bins[i+1]})' for i in range(len(bins)-1)]
calib={}
for c in ['pA','pB','pA0','pAneg']:
    print('== Kategorie',c)
    tab={}
    for g in ['M','C','R','S']:
        x=H[(H.grp==g)&(H[c]<1.0)]
        cnt=np.histogram(x[c],bins)[0]/nh[g]   # je Hand
        tab[g]=cnt
    prec=np.clip(1-tab['R']/np.maximum(tab['M'],1e-9),0,1)
    print(pd.DataFrame({'M/100h':tab['M']*100,'C/100h':tab['C']*100,'R/100h':tab['R']*100,'S/100h':tab['S']*100,'prec(M vs R)':prec},index=lab).round(3).to_string())
    calib[c]={'bins':bins,'rateM':tab['M'].tolist(),'rateR':tab['R'].tolist(),'prec':prec.tolist()}
# Raise-War ohne A/B-Treffer
for g in ['M','C','R']:
    x=H[(H.grp==g)&(H.pA>=1)&(H.pB>=1)&H.both]
    print(g,'RaiseWar ohne A/B: je 100 Haende',round(len(x)/nh[g]*100,2),' davon minp0<0.2',round((x.minp0_aggr<0.2).sum()/nh[g]*100,2),' <0.05',round((x.minp0_aggr<0.05).sum()/nh[g]*100,2))
# Kombination A und B in einer Hand
for g in ['M','R']:
    x=H[H.grp==g]; print(g,'A&B beide <0.2 je 100 Haende',round(((x.pA<0.2)&(x.pB<0.2)).sum()/nh[g]*100,2),' A<0.2',round((x.pA<0.2).sum()/nh[g]*100,2),' B<0.2',round((x.pB<0.2).sum()/nh[g]*100,2),' A<0.2 oder B<0.2',round(((x.pA<0.2)|(x.pB<0.2)).sum()/nh[g]*100,2))
# Gap-Abhaengigkeit der Praezision fuer A (p0<0.2)
P['gbin']=pd.cut(P.gap_a,[0.2,0.4,0.6,1.01],right=False)
for g in ['M','R']:
    x=P[(P.grp==g)&(P.cat=='A')&(P.p0a<0.2)]; print(g,'A p0<0.2 nach gap je 100 Haende',(x.groupby('gbin',observed=False).size()/nh[g]*100).round(2).to_dict())
    x=P[(P.grp==g)&(P.cat=='B')&(P.p0a<0.2)]; P['gbinf']=pd.cut(P.gap_f,[0.2,0.4,0.6,1.01],right=False); x=P[(P.grp==g)&(P.cat=='B')&(P.p0a<0.2)]; print(g,'B p0<0.2 nach gap je 100 Haende',(x.groupby('gbinf',observed=False).size()/nh[g]*100).round(2).to_dict())
# Kontext: Aktion vor/nach Partneraktion
for g in ['M','R']:
    x=P[(P.grp==g)&(P.cat=='A')&(P.p0a<0.2)]; print(g,'A p0<0.2: Anteil nach Partner-Aggression',round(x.pa_aggr_any.mean(),2),' je 100 Haende vor/nach',round((~x.pa_aggr_any).sum()/nh[g]*100,2),round(x.pa_aggr_any.sum()/nh[g]*100,2))
    x=P[(P.grp==g)&(P.cat=='B')&(P.p0a<0.2)]; print(g,'B p0<0.2: Anteil nach Partner-Aggression',round(x.pa_aggr_any.mean(),2),' je 100 Haende vor/nach',round((~x.pa_aggr_any).sum()/nh[g]*100,2),round(x.pa_aggr_any.sum()/nh[g]*100,2))
json.dump(calib,open('calib_raw.json','w'),indent=1)
H.to_parquet('hands_calib.parquet')
