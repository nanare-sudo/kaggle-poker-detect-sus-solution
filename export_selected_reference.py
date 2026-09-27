"""Reproduce the second selected file (old S5) from S5/J50 and generated b0."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
EXPECTED='9ecbbc514ddb6a95f107d805796116f2161d61b86cec9536f4fa45116f090efe'

def main(a):
    z=np.load(a.baseline);assert pd.Index(z['key']).is_unique
    ep=pd.read_parquet(HERE/'package/cache/eval_pairs.parquet').set_index('pair_id')
    frame=pd.read_csv(a.s5,dtype=str,keep_default_na=False);pairs=ep.loc[frame.pair_id]
    key=pairs.plo.to_numpy().astype(np.int64)*12000+pairs.phi.to_numpy()
    score=pd.Series(z['b_original'],index=z['key']).loc[key].to_numpy()
    risk=np.clip(pd.Series(score).rank(pct=True).to_numpy()*(1-2e-6)+1e-6,1e-6,1-1e-6)
    frame.risk_score=risk;assert not a.out.exists();frame.to_csv(a.out,index=False)
    got=hashlib.sha256(a.out.read_bytes()).hexdigest();report=dict(sha256=got,expected=EXPECTED,byte_exact=got==EXPECTED,
        rows=len(frame),only_risk_changed=True,baseline_sha256=hashlib.sha256(a.baseline.read_bytes()).hexdigest(),
        submitted=False,source='Generated Bothshort/T96/P18n b0 and S5/J50 evidence; no old CSV input')
    (a.out.parent/'SELECTED_REFERENCE_CHECK.json').write_text(json.dumps(report,indent=2)+'\n');assert got==EXPECTED;print(report)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,default=HERE/'s5/pair_baseline.npz')
    p.add_argument('--s5',type=Path,default=HERE/'s5/result/S5_J50_risk.csv')
    p.add_argument('--out',type=Path,default=HERE/'s5/result/sub_oc_O97_norisk_R1.csv');main(p.parse_args())
