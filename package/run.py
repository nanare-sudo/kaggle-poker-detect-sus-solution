"""Single entry point for inference with retained weights; never uploads results."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time

STAGES = ['core', 'math', 'policy', 'neural', 'features', 'contexts',
          'tabicl-inputs', 'tabicl', 'select', 'reserve', 'assemble', 'validate']
NEEDS = dict(core=[], math=['core'], policy=['core', 'math'], neural=['core', 'policy'],
    features=['math', 'policy', 'neural'], contexts=['math', 'policy', 'neural'],
    **{'tabicl-inputs': ['features', 'contexts'], 'tabicl': ['tabicl-inputs'],
       'select': ['features','tabicl'], 'reserve': ['select'],
       'assemble': ['features', 'tabicl','reserve'], 'validate': ['assemble']})


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def main(a):
    root = Path(__file__).resolve().parent
    assert not any(p.is_symlink() for p in (root/'cache').glob('*')), \
        'This is a read-only cache-linked integration workspace. Never run cache builders here; export a fresh source/weights-only runtime.'
    manifest = json.loads((root / 'MANIFEST.json').read_text())
    for n, expected in manifest['files'].items():
        assert sha(root / n) == expected, f'Package changed: {n}'
    fingerprint = sha(root / 'MANIFEST.json')
    done = root / 'receipts'; done.mkdir(exist_ok=True)
    src = root / 'src'; entry = root / 'pipeline'
    # Resolving a venv's python symlink would silently bypass that environment.
    cpu, gpu = str(a.cpu_python.expanduser().absolute()), str(a.gpu_python.expanduser().absolute())
    commands = {s: [cpu, str(entry / 'prepare_cache.py'), s, '--root', str(root), '--jobs', str(a.jobs)]
                for s in ['core', 'math', 'policy']}
    commands['core'] += ['--data', str(a.data.resolve())]
    commands['neural'] = [gpu, str(entry / 'neural_actions.py'), '--root', str(root)]
    for s, module in [('contexts', 'tabicl_contexts')]:
        commands[s] = [cpu, str(src / (module + '.py')), '--tables', '400', '--jobs', str(a.jobs)]
    commands['features']=[cpu,str(entry/'current_pairs.py'),'--tables','all','--jobs',str(a.jobs)]
    commands['tabicl-inputs'] = [cpu, str(entry / 'current_finish.py'), 'prepare', '--root', str(root)]
    commands['tabicl'] = [gpu, str(entry / 'tabicl_pairs.py'), '--root', str(root)]
    commands['select'] = [cpu,str(entry/'current_finish.py'),'select','--root',str(root)]
    commands['reserve'] = [cpu,str(entry/'current_reserve.py'),'--tables','all','--jobs',str(a.jobs)]
    commands['assemble'] = [cpu, str(entry / 'current_finish.py'), 'assemble', '--root', str(root)]
    commands['validate'] = [cpu, str(src / 'validate_submission.py'), str(root / 'result/Reserve_J50_risk.csv'),
                           '--data', str(a.data.resolve()), '--out', str(root / 'result/independent_validation.json')]
    for stage in STAGES if a.stage == 'all' else [a.stage]:
        path = done / (stage + '.json')
        if path.exists():
            assert json.loads(path.read_text())['package_sha256'] == fingerprint
            print('Already completed', stage, flush=True); continue
        for prior in NEEDS[stage]:
            assert json.loads((done / (prior + '.json')).read_text())['package_sha256'] == fingerprint
        # Each child starts clean with its own Python environment and cache root.
        env = dict(os.environ); env.pop('PYTHONPATH', None)
        for name in ['POLICY_FIX', 'POLICY_V2', 'SURP_STRAT', 'ENGINE_WK', 'ENGINE_SZ',
                     'ENGINE_CF', 'AGG_QREPLACE', 'AGG_EPI', 'Q2_POOL', 'Q2_WIN_HEAD']:
            env[name] = '0'
        env['AGG_QFRACS'] = ''; env['SEQ_FEAT'] = '1'
        env.pop('POKER_CHEN_ORDER_FILE', None)
        env.pop('NPY_DISABLE_CPU_FEATURES', None)
        if stage not in ['neural', 'tabicl']:
            env['POKER_CHEN_ORDER_FILE'] = str(entry / 'chen_order_cpu.json')
            # The original CPU used baseline log1p. NumPy 2.4.2's X86_V4
            # path changes tiny residual features enough to alter TabICL ranks.
            env['NPY_DISABLE_CPU_FEATURES'] = 'X86_V4'
        env.update(POKER_DATA=str(a.data.resolve()), POKER_CACHE=str(root / 'cache'),
                   Q2_STATS_FILE=str(root / 'cache/q2_pstats.npz'), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        print('Starting', stage, flush=True); t = time.time()
        if stage=='core':
            sample=a.data.resolve()/'sample_submission.csv'
            assert sample.is_file()
            shutil.copy2(sample,root/'raw_sample_submission.csv')
        subprocess.run(commands[stage], cwd=root, env=env, check=True)
        report = dict(stage=stage, package_sha256=fingerprint, seconds=time.time()-t,
                      command=commands[stage], public_upload=False)
        path.write_text(json.dumps(report, indent=2))
    print('Requested inference stages completed; no result was submitted.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['all']+STAGES, default='all')
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--cpu-python', type=Path, default=Path(sys.executable))
    p.add_argument('--gpu-python', type=Path, default=Path(sys.executable))
    p.add_argument('--jobs', type=int, default=4)
    main(p.parse_args())
