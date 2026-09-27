"""Compute OC card-dependence, action-likelihood and binned residual statistics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
JOBS = {
    'cardshare': 'cardshare_acc.npz',
    'cardshare_q2': 'cardshare_acc_q2.npz',
    'llr': 'llr_acc_c.npz',
    'bins': 'acc_bins7.npz',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--stage', choices=['all', *JOBS], default='all')
    args = parser.parse_args()
    cache = args.cache.resolve()
    out = HERE / 'oc_statistics'
    out.mkdir(exist_ok=True)
    env = dict(os.environ, POKER_CACHE=str(cache), OC_STATISTICS=str(out),
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    for name, filename in JOBS.items():
        if args.stage not in ('all', name):
            continue
        source = HERE / 'source/accumulators' / (name + '_original.py')
        target, receipt = out / filename, out / (name + '.json')
        if receipt.exists():
            record = json.loads(receipt.read_text())
            assert record['source_sha256'] == sha(source)
            assert record['output_sha256'] == sha(target)
            assert record['cache'] == str(cache)
            continue
        assert not target.exists(), f'Output without a completion record: {target}'
        started = time.time()
        with (out / (name + '.log')).open('w') as log:
            subprocess.run([sys.executable, '-B', str(source)], cwd=out, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        record = dict(stage=name, seconds=time.time()-started, source_sha256=sha(source),
                      output_sha256=sha(target), cache=str(cache))
        receipt.write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
