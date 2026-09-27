"""Validate a final CSV independently against the official raw files."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def validate(csv, data):
    out=pd.read_csv(csv,dtype=str,keep_default_na=False)
    sample=pd.read_csv(data/'sample_submission.csv',dtype=str)
    assert list(out.columns)==list(sample.columns)
    assert out.pair_id.is_unique and out.pair_id.tolist()==sample.pair_id.tolist()
    assert (out.to_numpy()!='').all()
    risk=pd.to_numeric(out.risk_score).to_numpy()
    assert np.isfinite(risk).all() and ((risk>=0)&(risk<=1)).all()
    assert out.predicted_behavior.isin(['none','directed_transfer','soft_play','coordinated_isolation','other_coordination']).all()
    cols=[f'evidence_hand_{i}' for i in range(1,6)]
    evidence=out[cols].to_numpy();valid=evidence!='NO_EVIDENCE'
    for row in evidence:
        hs=row[row!='NO_EVIDENCE'];assert len(hs)==len(set(hs))
    handids=pd.Index(np.unique(evidence[valid]))
    hands=pd.read_parquet(data/'hands.parquet',columns=['hand_id','phase'])
    pos=pd.Index(hands.hand_id).get_indexer(handids)
    assert (pos>=0).all() and (hands.phase.to_numpy()[pos]=='evaluation').all()
    players=pd.Index(pd.read_parquet(data/'players.parquet',columns=['player_id']).player_id)
    pairs=pd.read_csv(data/'evaluation_pairs.csv').set_index('pair_id').loc[out.pair_id]
    p1=players.get_indexer(pairs.player_1);p2=players.get_indexer(pairs.player_2)
    assert (p1>=0).all() and (p2>=0).all()
    seats=np.full((len(handids),6),-1,np.int32)
    for batch in pq.ParquetFile(data/'seats.parquet').iter_batches(batch_size=200000,columns=['hand_id','seat_no','player_id']):
        frame=batch.to_pandas();h=handids.get_indexer(frame.hand_id);take=h>=0
        ix=h[take];seat=frame.seat_no.to_numpy()[take]
        assert (seats[ix,seat]<0).all()
        seats[ix,seat]=players.get_indexer(frame.player_id[take])
    assert (seats>=0).all()
    for j in range(5):
        take=valid[:,j];h=handids.get_indexer(evidence[take,j]);assert (h>=0).all()
        assert ((seats[h]==p1[take,None]).any(1)&(seats[h]==p2[take,None]).any(1)).all()
    return dict(rows=len(out),sha256=hashlib.sha256(csv.read_bytes()).hexdigest(),
        evidence_references=int(valid.sum()),unique_hands=len(handids),
        no_evidence=int((~valid).sum()),validation='Raw official schema, IDs, phase, shared players, unique hands, finite risk and behavior checks passed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('csv',type=Path);p.add_argument('--data',type=Path,default=Path('data'));p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();report=validate(a.csv,a.data);a.out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
