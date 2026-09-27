"""Family-specific construction of the evidence feature matrix."""
import os
os.environ.setdefault('SEQ_FEAT','1')
import numpy as np
import phfeat
import role_features as RF
import ci_features as CI
FAMS = ["directed_transfer", "soft_play", "coordinated_isolation"]

def matrix(k,to,base,nm,R,rn,C,cn,events,en,gi,arm):
    """Shared training/serving construction; no labels or predictions are inputs."""
    raw,names = base,nm
    if gi == 0:
        raw,names = np.concatenate([base,R],1),nm+rn
    elif gi == 2:
        raw,names = np.concatenate([base,C],1),nm+cn
    clean = arm.startswith("CHRONO")
    X,nn = phfeat.enrich(k,raw,names,torder=to,
                        rank_ties="chronological" if clean else "ordinal",
                        gap_fill="pair" if clean else "global")
    if arm in ("ADD","CHRONO_ADD"):
        X,nn = np.concatenate([X,events],1),nn+en
    elif arm not in ("REF","CHRONO"):
        raise ValueError(arm)
    return X,nn
