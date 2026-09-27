"""Rebuild core arrays from raw files with explicit joins and bounded memory.

The output must be new. Neither competition data nor existing caches are changed.
Actions are indexed by (hand_id, action_no), seats by (hand_id, seat_no).
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from prep import encode_cards, STREETS, ACTIONS


def build(data, out):
    t = time.time()
    assert not out.exists(), f'Refusing existing output: {out}'
    out.mkdir(parents=True)
    def save(n, x): np.save(out/f'{n}.npy', x)
    def alloc(n, dtype, shape):
        return np.lib.format.open_memmap(out/f'{n}.npy',mode='w+',dtype=dtype,shape=shape)
    players = pd.read_parquet(data/'players.parquet').sort_values('player_id').reset_index(drop=True)
    pid = pd.Index(players.player_id); assert pid.is_unique
    save('player_ids',pid.to_numpy())
    save('player_account_age',players.account_age_days.to_numpy(np.int32))
    cats={}
    for n in ['experience_hands_bucket','preferred_stake','region_bucket','client_family']:
        c=players[n].astype('category');cats[n]=list(c.cat.categories)
        save('player_'+n,c.cat.codes.to_numpy(np.int8))
    hands=pd.read_parquet(data/'hands.parquet');hid=pd.Index(hands.hand_id);H=len(hid)
    assert hid.is_unique
    save('hand_ids',hid.to_numpy())
    c=hands.table_id.astype('category');cats['table_id']=list(c.cat.categories)
    save('hand_table',c.cat.codes.to_numpy(np.int16))
    save('hand_started_at',hands.started_at.astype('int64').to_numpy())
    save('hand_phase',(hands.phase.to_numpy()=='evaluation').astype(np.int8))
    for n,c,dt in [('button_seat','button_seat',np.int8),('sb','small_blind',np.int16),
                  ('bb','big_blind',np.int16),('final_pot','final_pot',np.int32),
                  ('n_showdown','players_at_showdown',np.int8)]:
        save('hand_'+n,hands[c].to_numpy(dt))
    board=hands.board_cards.fillna('').astype(str).str.split()
    save('hand_n_board',board.str.len().to_numpy(np.int8))
    bcards=np.full((H,5),-1,np.int8)
    for j in range(5):bcards[:,j]=encode_cards(board.str[j])
    save('hand_board',bcards);del board,bcards,hands
    seat_cols={'player':np.int16,'stack':np.int32,'contrib':np.int32,'net':np.int32,
        'folded':np.bool_,'showdown':np.bool_,'won_share':np.float32,'hole1':np.int8,'hole2':np.int8}
    seats={n:alloc('seat_'+n,dt,(H,6)) for n,dt in seat_cols.items()}
    seen=np.zeros(H*6,bool);seat_aligned=True;cursor=0
    for batch in pq.ParquetFile(data/'seats.parquet').iter_batches(batch_size=200000):
        s=batch.to_pandas();h=hid.get_indexer(s.hand_id);seat=s.seat_no.to_numpy(np.int64)
        assert (h>=0).all() and ((seat>=0)&(seat<6)).all()
        ix=h*6+seat;assert not seen[ix].any() and len(np.unique(ix))==len(ix)
        seen[ix]=True;seat_aligned &= np.array_equal(ix,np.arange(cursor,cursor+len(ix)));cursor+=len(ix)
        pl=pid.get_indexer(s.player_id);assert (pl>=0).all();seats['player'][h,seat]=pl
        for n,c in [('stack','starting_stack'),('contrib','total_contribution'),('net','net_chips'),
                    ('folded','folded'),('showdown','went_to_showdown'),('won_share','won_share')]:
            seats[n][h,seat]=s[c].to_numpy(seat_cols[n])
        for j in [1,2]:seats['hole'+str(j)][h,seat]=encode_cards(s['hole_card_'+str(j)])
    assert seen.all();del seen
    for x in seats.values():x.flush()
    seat_player=seats['player'];seats={}
    print('SEATS COMPLETE',H*6,round(time.time()-t),'seconds',flush=True)
    # Count by logical hand first; never infer hand membership from a row block.
    counts=np.zeros(H,np.int64);action_file=pq.ParquetFile(data/'actions.parquet')
    for batch in action_file.iter_batches(batch_size=300000,columns=['hand_id']):
        h=hid.get_indexer(batch.column(0).to_pandas());assert (h>=0).all()
        counts+=np.bincount(h,minlength=H)
    assert (counts>0).all()
    off=np.r_[0,np.cumsum(counts)];N=int(off[-1]);save('act_offsets',off)
    ah=alloc('act_hidx',np.int32,(N,))
    for a in range(0,H,100000):
        b=min(a+100000,H);ah[off[a]:off[b]]=np.repeat(np.arange(a,b,dtype=np.int32),counts[a:b])
    ah.flush();del ah
    acols={'seat':np.int8,'player':np.int16,'street':np.int8,'action':np.int8,
        'amount':np.int32,'amount_to':np.int32,'pot_before':np.int32,'stack_before':np.int32,
        'to_call':np.int32,'players_active':np.int8}
    acts={n:alloc('act_'+n,dt,(N,)) for n,dt in acols.items()}
    seen=np.zeros(N,bool);action_aligned=True;cursor=0
    for batch in action_file.iter_batches(batch_size=200000):
        s=batch.to_pandas();h=hid.get_indexer(s.hand_id);ano=s.action_no.to_numpy(np.int64)
        assert (h>=0).all() and ((ano>=0)&(ano<counts[h])).all()
        ix=off[h]+ano;assert not seen[ix].any() and len(np.unique(ix))==len(ix)
        seen[ix]=True;action_aligned &= np.array_equal(ix,np.arange(cursor,cursor+len(ix)));cursor+=len(ix)
        pl=pid.get_indexer(s.player_id);assert (pl>=0).all()
        eq=seat_player[h]==pl[:,None];assert (eq.sum(1)==1).all()
        acts['seat'][ix]=eq.argmax(1);acts['player'][ix]=pl
        for n,vocab in [('street',STREETS),('action',ACTIONS)]:
            vals=pd.Categorical(s[n],categories=vocab).codes;assert (vals>=0).all()
            acts[n][ix]=vals
        for n in ['amount','amount_to','pot_before','stack_before','to_call','players_active']:
            acts[n][ix]=s[n].to_numpy(acols[n])
    assert seen.all()
    for x in acts.values():x.flush()
    print('ACTIONS COMPLETE',N,round(time.time()-t),'seconds',flush=True)
    pcode={p:i for i,p in enumerate(pid)}
    lab=pd.read_csv(data/'development_labels.csv');evp=pd.read_csv(data/'evaluation_pairs.csv')
    evi=pd.read_csv(data/'development_evidence.csv')
    for df in (lab,evp):
        df['p1']=df.player_1.map(pcode);df['p2']=df.player_2.map(pcode)
        assert df[['p1','p2']].notna().all().all()
        df['plo']=np.minimum(df.p1,df.p2);df['phi']=np.maximum(df.p1,df.p2)
    evi['hidx']=hid.get_indexer(evi.hand_id);assert (evi.hidx>=0).all()
    lab.to_parquet(out/'labels.parquet');evp.to_parquet(out/'eval_pairs.parquet');evi.to_parquet(out/'evidence.parquet')
    (out/'cat_maps.json').write_text(json.dumps(cats,indent=1))
    (out/'meta.json').write_text(json.dumps(dict(n_hands=H,n_players=len(pid),n_actions=N,
        seats_aligned=bool(seat_aligned),actions_aligned=bool(action_aligned),streets=STREETS,actions_vocab=ACTIONS),indent=1))
    print('COMPLETE',round(time.time()-t),'seconds',flush=True)
