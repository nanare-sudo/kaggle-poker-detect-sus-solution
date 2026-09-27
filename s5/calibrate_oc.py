"""Reproduce the OC M-versus-R bin tables from observed actions.

M denotes detector-selected pairs, not labeled collusion. This reproduces the
historical heuristic precision table and its fixed 0.45 rescaling. It does not
validate those numbers as calibrated collusion or official-list probabilities.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
from oc_routes import script_run
HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--routes',type=Path,default=HERE/'oc_routes')
    p.add_argument('--newly-trained',action='store_true',help='Fit current bin weights without requiring equality to retained weights')
    a=p.parse_args()
    cache=a.cache.resolve();routes=a.routes.resolve();out=HERE/'oc_calibration';out.mkdir(exist_ok=True)
    sources=HERE/'source/route_original';started=time.time()
    variables={'POKER_CACHE':str(cache),'OC_ROUTES':str(routes),'OC_CALIBRATION':str(out)}
    script_run(sources/'build_decisions.py',out,variables)
    script_run(sources/'calib.py',out,variables)
    raw=json.loads((out/'calib_raw.json').read_text());old=json.loads((HERE/'assets/calib_norho.json').read_text())
    result={}
    for cat in ['pA','pA0','pB']:
        values=np.round(raw[cat]['prec'],3)
        if not a.newly_trained:np.testing.assert_array_equal(values,old[cat]['prec_MvsR'],err_msg=cat)
        scaled=.45*values/values[0]
        if not a.newly_trained:np.testing.assert_array_equal(scaled,old[cat]['prec'],err_msg=cat+' scaled')
        result[cat]=dict(bins=raw[cat]['bins'],prec_MvsR=values.tolist(),pi_max=.45,prec=scaled.tolist())
    result['_doc']='Actual norho parameters; unused rho_listed metadata omitted.'
    (out/'calib_norho.json').write_text(json.dumps(result,indent=2)+'\n')
    d=pd.read_parquet(out/'decisions.parquet')
    report=dict(seconds=time.time()-started,rows=len(d),groups=d.groupby('grp').pair_id.nunique().to_dict(),
                denominator_unique_hand_ids=d.groupby('grp').h.nunique().to_dict(),
                retained_parameters_compared=not a.newly_trained,
                used_parameters_exact=all(np.array_equal(result[c]['prec'],old[c]['prec']) for c in ['pA','pA0','pB']),labels_used=False,
                warning='Detector-selected M and random R are mixtures; the precision interpretation is a heuristic.')
    (HERE/'OC_CALIBRATION_CHECK.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)

if __name__=='__main__':main()
