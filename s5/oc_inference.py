"""Rank shared evaluation hands for OC-routed pairs using fixed action bins and chronological decoding."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'source'))
import oc_evidence_final as OC

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--pairs',type=Path,required=True);ap.add_argument('--check',type=Path)
    a=ap.parse_args();start=time.time();out=HERE/'oc';out.mkdir(exist_ok=True)
    EQ=np.load(a.cache/'preflop_equity.npy').astype(np.float64)
    w=np.array([6 if hi==lo else 4 if hi>lo else 12 for hi in range(13) for lo in range(13)])
    evr=(EQ*w[None,:]).sum(1)/w.sum();order=np.argsort(evr);cum=np.cumsum(w[order])/w.sum()
    pct=np.empty(169);pct[order]=cum-w[order]/w.sum()/2
    np.savez(out/'llr_acc_c.npz',pct=pct)
    OC.C=str(a.cache.resolve())+'/';OC.O=str(out.resolve())+'/'
    D=OC.Data();cal=OC.load_calib(HERE/'assets/calib_norho.json')
    pairs=pd.read_parquet(a.pairs) if a.pairs.suffix=='.parquet' else pd.read_csv(a.pairs)
    assert pairs.pair_id.is_unique
    frames=[]
    for r in pairs.itertuples(index=False):
        df=OC.score_pair(D,cal,int(r.p1),int(r.p2),np.array([0.,1.,0.]),.02,5)
        assert df is not None;df.insert(0,'pair_id',r.pair_id);frames.append(df)
    result=pd.concat(frames,ignore_index=True)
    path=out/'oc_block_norho_R1.parquet';assert not path.exists();result.to_parquet(path,index=False)
    report=dict(pairs=len(pairs),hands=len(result),seconds=time.time()-start,
        pair_input=str(a.pairs),pair_input_sha256=hashlib.sha256(a.pairs.read_bytes()).hexdigest(),
        weights=[0,1,0],calibration='calib_norho.json',pi_none=.02,
        scope='Live action-to-hand-score inference; upstream pair detector is a separate stage.')
    if a.check:
        ref=pd.read_parquet(a.check);cols=ref.columns.tolist()
        x=result.sort_values(['pair_id','hand_id']).reset_index(drop=True)[cols]
        y=ref.sort_values(['pair_id','hand_id']).reset_index(drop=True)
        pd.testing.assert_frame_equal(x,y,check_exact=True)
        report.update(all_columns_exact=True,picks_exact=int((result.pick_rank<=5).sum()))
    (HERE/'OC_HAND_REPLAY.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
