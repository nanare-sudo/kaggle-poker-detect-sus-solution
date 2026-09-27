"""Shared decision, hand and role features for evidence models."""
import os
os.environ.setdefault('SEQ_FEAT','1')
os.environ.setdefault('POLICY_Q2','1')
os.environ.setdefault('POLICY_Q2_SFX','_q2')
from pathlib import Path
import numpy as np
import engine
import evidence as EV
import seqwire
import role_features as RF
import ci_features as CI
import decision_features as E
import evidence_matrix as G
import evidence_filter as AF

def live_data(table,phase,want):
    ht,hp,st=engine.A('hand_table'),engine.A('hand_phase'),engine.A('hand_started_at')
    sel=np.flatnonzero((ht==table)&(hp==phase))
    sel=sel[np.argsort(st[sel],kind='stable')]
    k,h,to,X,names=engine.table_pair_hands(sel)
    take=np.isin(k,want)
    k,h,to,X=k[take],h[take],to[take],X[take]
    o=np.lexsort((to,k)); k,h,to,X=k[o],h[o],to[o],X[o]
    keep=EV.evid_keep(names)
    base=X[:,keep]; nm=[n for n,b in zip(names,keep) if b]
    S,sn=seqwire.live(sel,k,h)
    base,nm=np.concatenate([base,S],1),nm+sn
    R,rn=RF.compute(k,h); C,cn=CI.compute(k,h)
    events,en=E.compute(k,h)
    return k,h,to,base,nm,R,rn,C,cn,events,en
