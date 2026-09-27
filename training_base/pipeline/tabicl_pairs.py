"""Run the frozen TabICL checkpoint with deterministic normalization ordering."""
import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker(j, root):
    import torch
    import importlib.metadata
    from tabicl import TabICLClassifier
    from tabicl._sklearn.preprocessing import EnsembleGenerator
    from collections import OrderedDict
    assert importlib.metadata.version('tabicl') == '2.2.0'
    original = EnsembleGenerator._generate_ensemble
    def stable(ensemble):
        values = original(ensemble)
        assert all(set(v) == {'none', 'power'} for v in values)
        return tuple(OrderedDict((n, v[n]) for n in ['none', 'power']) for v in values)
    EnsembleGenerator._generate_ensemble = stable
    torch.set_num_threads(2); t = time.time()
    data = root / 'tabicl_inputs'; out = root / 'tabicl_predictions'
    manifest = json.loads((data / 'input_manifest.json').read_text())
    for n, expected in manifest['files'].items():
        assert sha(data / n) == expected
    z = np.load(data / f'context{j}.npz'); v = np.load(data / 'evaluation.npz')
    model = TabICLClassifier(n_estimators=8, batch_size=2, kv_cache=True,
        device=f'cuda:{j}', use_amp=True, use_fa3=False, n_jobs=2, random_state=42+j,
        allow_auto_download=False, model_path=root / 'assets/tabicl/tabicl-classifier-v2-20260212.ckpt')
    model.fit(z['X_train'], z['y_train'])
    pp = []
    for a in range(0, len(v['X_eval']), 1024):
        pp.append(model.predict_proba(v['X_eval'][a:a+1024])[:, 1])
    pred = np.concatenate(pp); assert np.isfinite(pred).all() and ((pred >= 0) & (pred <= 1)).all()
    path = out / f'context{j}_pred.npz'; assert not path.exists()
    np.savez(path, prediction=pred, key=v['key_eval'])
    report = dict(context=j, eval_rows=len(pred), train_rows=len(z['y_train']), smoke=False,
        model_seed=42+j, seconds=time.time()-t, numpy=np.__version__, torch=torch.__version__,
        gpu=torch.cuda.get_device_name(j), normalization_order=list(model.ensemble_generator_.ensemble_configs_),
        input_files=manifest['files'], prediction_sha256=sha(path))
    (out / f'context{j}_report.json').write_text(json.dumps(report, indent=2))
    print('TabICL complete', report, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); a = p.parse_args()
    root = a.root.resolve(); (root / 'tabicl_predictions').mkdir(exist_ok=True)
    import torch
    assert torch.cuda.device_count() >= 2, 'Verified recipe uses two Tesla T4 GPUs'
    from multiprocessing import get_context
    jobs = [get_context('spawn').Process(target=worker, args=(j, root)) for j in [0, 1]]
    for job in jobs: job.start()
    for job in jobs:
        job.join(); assert job.exitcode == 0
