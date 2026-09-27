"""Fuse pair-risk components and assemble Reserve evidence predictions."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logit


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def align(keys,wanted):
    assert pd.Index(keys).is_unique
    ix=pd.Index(keys).get_indexer(wanted);assert (ix>=0).all(), 'Missing pair key'
    np.testing.assert_array_equal(keys[ix],wanted)
    return ix


def collect(root):
    parts=[np.load(root/f'current_pairs/{t:03d}.npz') for t in range(400)]
    columns=['key','anchor','behavior','pool','tabicl','tabicl192','evidence']
    data={n:np.concatenate([p[n] for p in parts]) for n in columns}
    for name in ['bothshort_seeds','rclean_seeds']:
        data[name]=np.concatenate([p[name].T for p in parts])
    assert len(data['key'])==112540 and pd.Index(data['key']).is_unique
    ep=pd.read_parquet(root/'cache/eval_pairs.parquet')
    keys=ep.plo.to_numpy().astype(np.int64)*12000+ep.phi.to_numpy()
    ix=align(data['key'],keys)
    return ep,{n:v[ix] for n,v in data.items()}


def prepare(root):
    ep,d=collect(root)
    for width in [96,192]:
        out=root/f'tabicl{width}_inputs';out.mkdir(exist_ok=True)
        assert not (out/'input_manifest.json').exists()
        layout=np.asfortranarray if width==96 else np.ascontiguousarray
        values=d['tabicl'] if width==96 else d['tabicl192']
        np.testing.assert_array_equal(d['tabicl192'][:,:96],d['tabicl'])
        np.savez_compressed(out/'evaluation.npz',X_eval=layout(values),key_eval=d['key'])
        for j in [0,1]:
            context=np.load(root/f'current_contexts/context{j}.npz')
            assert context['X_train'].shape==(9860,192)
            np.savez_compressed(out/f'context{j}.npz',X_train=layout(context['X_train'][:,:width]),
                y_train=context['y_train'],key_train=context['key_train'])
        manifest=dict(width=width,rows=len(ep),files={p.name:sha(p) for p in out.glob('*.npz')})
        (out/'input_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Prepared T96 and T192 with original memory layouts',flush=True)


def fusion(root,d):
    tab={}
    for width in [96,192]:
        inputs=root/f'tabicl{width}_inputs';folder=root/f'tabicl{width}_predictions'
        manifest=json.loads((inputs/'input_manifest.json').read_text())
        for name,digest in manifest['files'].items():assert sha(inputs/name)==digest
        values=[]
        for j in [0,1]:
            path=folder/f'context{j}_pred.npz';z=np.load(path)
            report=json.loads((folder/f'context{j}_report.json').read_text())
            assert report['normalization_order']==['none','power']
            assert report['eval_rows']==112540 and report['train_rows']==9860
            assert report['features']==width and not report['smoke']
            assert report['input_files']==manifest['files'] and report['prediction_sha256']==sha(path)
            # T192 deployment promotes each context output before averaging;
            # T96 keeps its original float32 mean. Preserve both contracts.
            prediction=z['prediction'][align(z['key'],d['key'])]
            values.append(prediction.astype(np.float64) if width==192 else prediction)
        tab[width]=np.mean(values,axis=0)
    return combine(d,tab)


def combine(d,tab):
    """Pure arithmetic, separately checked on all stored deployment components."""
    def numpy_logit(p,eps):
        p=np.clip(p,eps,1-eps)
        return np.log(p/(1-p))
    P=numpy_logit(d['pool'],1e-9).mean(1)
    old=.75*(.5*numpy_logit(d['anchor'].mean(1),1e-7)+.5*numpy_logit(tab[96],1e-7))+.25*P
    A=d['bothshort_seeds'].mean(1);C=d['rclean_seeds'].mean(1)
    l=lambda v:logit(np.clip(v,1e-7,1-1e-7))
    b0=.75*(.5*l(A)+.5*l(tab[96]))+.25*P
    b1=.75*(.5*l(C)+.5*l(tab[192]))+.25*P
    b=.5*b0+.5*b1
    assert np.isfinite(old).all() and np.isfinite(b).all()
    def rank(s):return np.clip(pd.Series(s).rank(pct=True).to_numpy()*(1-2e-6)+1e-6,1e-6,1-1e-6)
    risk=rank(b);oldrank=rank(old)
    gate=oldrank>=np.partition(oldrank,len(oldrank)-10000)[len(oldrank)-10000]
    assert int(gate.sum())==10000, 'Historical tie treatment must select 10000 rows'
    return risk,gate


def select(root):
    _,d=collect(root);risk,gate=fusion(root,d)
    out=root/'selection.npz';assert not out.exists()
    np.savez(out,key=d['key'][gate],route=d['behavior'][gate].argmax(1))
    np.savez(root/'current_risk.npz',key=d['key'],risk=risk)
    print('Generated historical Reserve gate',int(gate.sum()),flush=True)


def assemble(root):
    ep,d=collect(root);risk,gate=fusion(root,d)
    selection=np.load(root/'selection.npz')
    np.testing.assert_array_equal(selection['key'],d['key'][gate])
    np.testing.assert_array_equal(selection['route'],d['behavior'][gate].argmax(1))
    # Variant 0 is the old decision ranker on the full field. The same 10000
    # gated rows were overwritten by MAP and then Reserve; Reserve supersedes MAP.
    chosen=d['evidence'][:,0].copy();seen=[]
    for table in range(400):
        z=np.load(root/f'reserve_predictions/{table:03d}.npz')
        ix=align(d['key'],z['key']);assert gate[ix].all()
        chosen[ix]=z['h'];seen.extend(z['key'].tolist())
    np.testing.assert_array_equal(np.sort(seen),np.sort(selection['key']))
    assert (chosen>=0).all()
    handids=np.load(root/'cache/hand_ids.npy',allow_pickle=True)
    families=np.array(['directed_transfer','soft_play','coordinated_isolation'])
    frame=pd.DataFrame(dict(pair_id=ep.pair_id,risk_score=risk,
        predicted_behavior=families[d['behavior'].argmax(1)]))
    for j in range(5):frame[f'evidence_hand_{j+1}']=handids[chosen[:,j]]
    # Official sample order; IDs are only join keys, never model inputs.
    sample=pd.read_csv(root/'raw_sample_submission.csv',usecols=['pair_id'])
    assert len(sample)==len(frame) and sample.pair_id.is_unique and frame.pair_id.is_unique
    frame=frame.set_index('pair_id').loc[sample.pair_id].reset_index()
    out=root/'result';out.mkdir(exist_ok=True);path=out/'Reserve_J50_risk.csv'
    assert not path.exists();frame.to_csv(path,index=False)
    report=dict(rows=len(frame),sha256=sha(path),reserve_pairs=len(seen),
        scope='Fresh current Reserve/J50 inference, no count decoder or OC overlay',public_upload=False)
    (out/'assembly.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','select','assemble'])
    p.add_argument('--root',type=Path,required=True);a=p.parse_args();globals()[a.stage](a.root.resolve())
