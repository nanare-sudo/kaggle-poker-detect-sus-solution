"""Portable TabICL input preparation and CSV assembly from newly generated outputs."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def align(keys, wanted):
    assert len(np.unique(keys)) == len(keys)
    order = np.argsort(keys); ix = np.searchsorted(keys[order], wanted)
    assert (ix < len(keys)).all()
    result = order[ix]
    np.testing.assert_array_equal(keys[result], wanted)
    return result


def collect(root):
    parts = [np.load(root / f'out/{t:03d}.npz') for t in range(400)]
    manifests = {str(p['manifest_sha256']) for p in parts}
    assert len(manifests) == 1
    data = {n: np.concatenate([p[n] for p in parts])
        for n in ['key', 'anchor', 'behavior', 'pool', 'tabicl', 'evidence']}
    assert len(data['key']) == len(np.unique(data['key'])) == 112540
    ep = pd.read_parquet(root / 'cache/eval_pairs.parquet')
    keys = ep.plo.to_numpy().astype(np.int64)*12000 + ep.phi.to_numpy()
    order = align(data['key'], keys)
    return ep, {n: v[order] for n, v in data.items()}, manifests.pop()


def prepare(root):
    ep, data, manifest = collect(root)
    out = root / 'tabicl_inputs'; out.mkdir(exist_ok=True)
    assert not (out / 'input_manifest.json').exists(), 'Completed inputs must not be overwritten'
    np.savez_compressed(out / 'evaluation.npz', X_eval=np.asfortranarray(data['tabicl']), key_eval=data['key'])
    for j in [0, 1]:
        z = np.load(root / f'context_replay/context{j}.npz')
        assert z['X_train'].shape == (9860, 96)
        np.savez_compressed(out / f'context{j}.npz', X_train=np.asfortranarray(z['X_train']),
            y_train=z['y_train'], key_train=z['key_train'])
    report = dict(rows=len(ep), feature_manifest=manifest,
        files={p.name: sha(p) for p in out.glob('*.npz')})
    (out / 'input_manifest.json').write_text(json.dumps(report, indent=2))
    print('Prepared fresh TabICL inputs', len(ep), flush=True)


def assemble(root):
    ep, d, manifest = collect(root)
    input_manifest = json.loads((root / 'tabicl_inputs/input_manifest.json').read_text())
    assert input_manifest['feature_manifest'] == manifest
    for n, expected in input_manifest['files'].items():
        assert sha(root / 'tabicl_inputs' / n) == expected
    def logit(p, eps):
        p = np.clip(p, eps, 1-eps)
        return np.log(p/(1-p))
    tt = []
    for j in [0, 1]:
        folder = root / 'tabicl_predictions'
        z = np.load(folder / f'context{j}_pred.npz')
        report = json.loads((folder / f'context{j}_report.json').read_text())
        assert report['normalization_order'] == ['none', 'power']
        assert report['eval_rows'] == 112540 and report['train_rows'] == 9860 and not report['smoke']
        np.testing.assert_array_equal(z['key'], d['key'])
        assert report['input_files'] == input_manifest['files']
        assert report['prediction_sha256'] == sha(folder / f'context{j}_pred.npz')
        tt.append(z['prediction'])
    A = d['anchor'].mean(1); P = logit(d['pool'], 1e-9).mean(1); T = np.mean(tt, axis=0)
    score = .75*(.5*logit(A, 1e-7)+.5*logit(T, 1e-7))+.25*P
    assert np.isfinite(score).all()
    risk = np.clip(pd.Series(score).rank(pct=True).to_numpy()*(1-2e-6)+1e-6, 1e-6, 1-1e-6)
    gate = risk >= np.partition(risk, len(risk)-10000)[len(risk)-10000]
    hh = d['evidence'][:, 0].copy(); hh[gate] = d['evidence'][gate, 1]
    handids = np.load(root / 'cache/hand_ids.npy', allow_pickle=True)
    families = np.array(['directed_transfer', 'soft_play', 'coordinated_isolation'])
    result = pd.DataFrame(dict(pair_id=ep.pair_id, risk_score=risk,
        predicted_behavior=families[d['behavior'].argmax(1)]))
    for j in range(5):
        values = np.full(len(hh), 'NO_EVIDENCE', dtype=object); valid = hh[:, j] >= 0
        values[valid] = handids[hh[valid, j]]
        result[f'evidence_hand_{j+1}'] = values
    out = root / 'result'; out.mkdir(exist_ok=True); path = out / 'submission.csv'
    assert not path.exists(), 'Completed submissions must not be overwritten'
    result.sort_values('pair_id', kind='stable').to_csv(path, index=False)
    report = dict(rows=len(result), selected_pairs=int(gate.sum()), sha256=sha(path),
        feature_manifest=manifest, public_upload=False)
    (out / 'assembly.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['prepare', 'assemble'])
    p.add_argument('--root', type=Path, required=True); a = p.parse_args()
    globals()[a.stage](a.root.resolve())
