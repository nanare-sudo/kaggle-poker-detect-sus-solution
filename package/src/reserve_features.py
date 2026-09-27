"""Feature matrices shared by Reserve training and inference."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['SEQ_FEAT'] = '1'
for n in ['POLICY_FIX','POLICY_V2','SURP_STRAT','ENGINE_WK','ENGINE_SZ','ENGINE_CF',
          'AGG_QREPLACE','AGG_EPI','Q2_POOL','Q2_WIN_HEAD']:
    os.environ[n] = '0'
os.environ['AGG_QFRACS'] = ''
import argparse
import json
from pathlib import Path
import time
import numpy as np
import hand_features as S
import seqwire

ROOT = Path(__file__).resolve().parents[1]


def live(table, phase, want):
    E = S.engine
    E.POLICY_Q2 = True; E.POLICY_Q2_SFX = '_q2'
    data = S.live_data(table, phase, want)
    k,h,to,base,names,R,rn,C,cn,events,en = data
    ht,hp,st = [E.A(n) for n in ['hand_table','hand_phase','hand_started_at']]
    sel = np.flatnonzero((ht == table) & (hp == phase))
    sel = sel[np.argsort(st[sel],kind='stable')]
    seq, sn = seqwire.live(sel,k,h)
    teachers = {}; feature_names = {}
    for arm in ['CLASSIC','NN_EVENTS']:
        E.POLICY_Q2 = arm == 'NN_EVENTS'; E.POLICY_Q2_SFX = '_q2nnnoply'
        kk,hh,tt,X,nm = E.table_pair_hands(sel)
        take = np.isin(kk,want)
        kk,hh,tt,X = kk[take],hh[take],tt[take],X[take]
        order = np.lexsort((tt,kk)); kk,hh,tt,X = kk[order],hh[order],tt[order],X[order]
        for a,b in [(k,kk),(h,hh),(to,tt)]: np.testing.assert_array_equal(a,b)
        keep = S.EV.evid_keep(nm)
        parts = [X[:,keep],seq]
        fn = [n for n,b in zip(nm,keep) if b]+sn
        if arm == 'NN_EVENTS': parts.append(events); fn += en
        teachers[arm] = np.concatenate(parts,1); feature_names[arm] = fn
    E.POLICY_Q2 = True; E.POLICY_Q2_SFX = '_q2'
    return data, teachers, feature_names
