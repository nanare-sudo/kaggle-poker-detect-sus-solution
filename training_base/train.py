"""Source-based training and inference entry; no competition upload actions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

STAGES = ['core', 'math', 'stats', 'train-classic', 'train-q2', 'install-policies',
    'policy', 'train-neural', 'install-neural', 'neural', 'pair-inputs', 'pair-heads',
    'instance-inputs', 'instance-final', 'instance-teachers', 'pool-inputs', 'pool-heads',
    'evidence-inputs', 'evidence-heads', 'assemble-assets', 'features', 'contexts',
    'tabicl-inputs', 'tabicl', 'assemble', 'validate', 'export']
GPU_STAGES = {'train-neural', 'neural', 'tabicl'}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def commands(root, data, cpu, gpu, jobs):
    pipe, src, train = root / 'pipeline', root / 'src', root / 'training'
    def script(name, *args, device='cpu'):
        return [gpu if device == 'gpu' else cpu, str(pipe / name), *map(str, args)]
    cmd = {s: script('prepare_cache.py', s, '--root', root, '--jobs', jobs) for s in ['core', 'math', 'policy']}
    cmd['core'] += ['--data', str(data)]
    for s in ['stats', 'install-policies', 'install-neural', 'assemble-assets', 'export']:
        cmd[s] = script('training_support.py', s, '--root', root)
    for name in ['classic', 'q2']:
        cmd['train-' + name] = script('train_action_references.py', '--root', root, '--arm', name,
                                     '--out', train / 'action_references')
    cmd['train-neural'] = script('train_neural_reference.py', '--root', root, '--out', train / 'neural', device='gpu')
    cmd['neural'] = script('neural_actions.py', '--root', root, device='gpu')
    for name in ['pair', 'instance', 'pool', 'evidence']:
        module = 'build_training_pairs.py' if name == 'pair' else f'build_training_{name}.py'
        args = ['--root', root, '--out', train / (name + '_inputs'), '--tables', 400, '--jobs', jobs]
        if name == 'pool':
            args += ['--teachers', train / 'instance_teachers']
        cmd[name + '-inputs'] = script(module, *args)
    cmd['pair-heads'] = script('train_pair_heads.py', '--root', root, '--inputs', train / 'pair_inputs', '--out', train / 'pair_heads')
    cmd['instance-final'] = script('train_instance_reference.py', '--inputs', train / 'instance_inputs', '--out', train / 'instance_final')
    cmd['instance-teachers'] = script('train_instance_teachers.py', '--inputs', train / 'instance_final/training_inputs.npz',
        '--names', train / 'instance_inputs/feature_names.json', '--out', train / 'instance_teachers')
    for name in ['pool', 'evidence']:
        cmd[name + '-heads'] = script(f'train_{name}_heads.py', '--root', root, '--inputs', train / (name + '_inputs'), '--out', train / (name + '_heads'))
    for name, module in [('features', 'pair_inference.py'), ('contexts', 'tabicl_contexts.py')]:
        cmd[name] = [cpu, str(src / module), '--tables', '400', '--jobs', str(jobs)]
    cmd['tabicl-inputs'] = script('finish.py', 'prepare', '--root', root)
    cmd['tabicl'] = script('tabicl_pairs.py', '--root', root, device='gpu')
    cmd['assemble'] = script('finish.py', 'assemble', '--root', root)
    cmd['validate'] = [cpu, str(src / 'validate_submission.py'), str(root / 'result/submission.csv'),
                       '--data', str(data), '--out', str(root / 'result/independent_validation.json')]
    assert set(cmd) == set(STAGES)
    return cmd


def output_patterns(stage):
    special = {'core': ['cache/*'],
        'math': ['cache/act_equity.npy', 'cache/preflop_equity.npy', 'cache/expect_table.npy', 'math_cache_report.json'],
        'stats': ['cache/q2_pstats.npz'],
        'train-classic': ['training/action_references/classic.*'],
        'train-q2': ['training/action_references/q2.*'],
        'install-policies': ['assets/policy_*.txt'],
        'policy': [f'cache/act_{n}{s}.npy' for n in ['surprise', 'paggr', 'pfold'] for s in ['', '_q2']],
        'train-neural': ['training/neural/training.json', 'training/neural/specification.json', 'training/neural/models/*.pt'],
        'install-neural': ['assets/neural/*.pt'],
        'neural': ['cache/*_q2nnnoply.npy', 'neural_cache_report.json'],
        'pair-heads': ['training/pair_heads/**/*', 'features/pair_*dev_q2nnnoply.*', 'features/pair_*dev_w2000.*'],
        'assemble-assets': ['MODEL_MANIFEST.json', 'assets/**/*'],
        'features': ['out/*.npz', 'out/source_manifest.json'],
        'contexts': ['context_replay/*.npz'],
        'tabicl-inputs': ['tabicl_inputs/*'],
        'tabicl': ['tabicl_predictions/*'],
        'assemble': ['result/submission.csv', 'result/assembly.json'],
        'validate': ['result/independent_validation.json'],
        'export': ['inference_export/**/*']}
    return special[stage] if stage in special else ['training/' + stage.replace('-', '_') + '/**/*']


def environment(root, data, stage):
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    for n in ['POLICY_FIX', 'POLICY_V2', 'SURP_STRAT', 'ENGINE_WK', 'ENGINE_SZ', 'ENGINE_CF',
              'AGG_QREPLACE', 'AGG_EPI', 'Q2_POOL', 'Q2_WIN_HEAD']:
        env[n] = '0'
    env.update(AGG_QFRACS='', SEQ_FEAT='1', POKER_DATA=str(data), POKER_CACHE=str(root / 'cache'),
               Q2_STATS_FILE=str(root / 'cache/q2_pstats.npz'), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    env.update(POLICY_Q2='0', POLICY_Q2_SFX='_q2')
    env.pop('POKER_CHEN_ORDER_FILE', None)
    env.pop('NPY_DISABLE_CPU_FEATURES', None)
    if stage not in GPU_STAGES:
        env['POKER_CHEN_ORDER_FILE'] = str(root / 'pipeline/chen_order_cpu.json')
        env['NPY_DISABLE_CPU_FEATURES'] = 'X86_V4'
    return env


def main(a):
    package = Path(__file__).resolve().parent
    manifest = json.loads((package / 'TRAINING_MANIFEST.json').read_text())
    for name, expected in manifest['files'].items():
        assert sha(package / name) == expected, ('Package changed', name)
    fingerprint = sha(package / 'TRAINING_MANIFEST.json')
    root, data = a.out.resolve(), a.data.resolve()
    assert root != package and not root.is_relative_to(package)
    cpu = str(a.cpu_python.expanduser().absolute())
    gpu = str(a.gpu_python.expanduser().absolute())
    cmd = commands(root, data, cpu, gpu, a.jobs)
    if a.plan:
        print(json.dumps(dict(source_manifest_sha256=fingerprint, output_root=str(root),
            commands=cmd, stages=STAGES, public_upload=False), indent=2))
        return
    assert data.is_dir()
    if not root.exists():
        shutil.copytree(package, root)
        (root / 'TRAINING_ROOT.json').write_text(json.dumps(dict(source_manifest_sha256=fingerprint, data=str(data)), indent=2))
    else:
        marker = json.loads((root / 'TRAINING_ROOT.json').read_text())
        assert marker == dict(source_manifest_sha256=fingerprint, data=str(data))
        for name, expected in manifest['files'].items():
            assert sha(root / name) == expected, ('Runtime source changed', name)
    receipts, logs = root / 'training_receipts', root / 'training_logs'
    receipts.mkdir(exist_ok=True)
    logs.mkdir(exist_ok=True)
    (root / 'training').mkdir(exist_ok=True)
    verified = set()
    checked_environments = set()
    def check_environment(stage):
        role = 'gpu' if stage in GPU_STAGES else 'cpu'
        if role in checked_environments:
            return
        requirements = [line.strip().split('==') for line in (root / f'requirements-{role}.txt').read_text().splitlines() if line.strip()]
        code = ('import importlib.metadata as m,json,sys\n'
                f'required={dict(requirements)!r}\n'
                'actual={name:m.version(name) for name in required}\n'
                'assert actual==required,(actual,required)\n'
                'print(json.dumps(dict(python=sys.version,packages=actual)))\n')
        output = subprocess.check_output([gpu if role == 'gpu' else cpu, '-c', code],
            env=environment(root, data, stage), text=True)
        (root / f'training_environment_{role}.json').write_text(json.dumps(json.loads(output), indent=2))
        checked_environments.add(role)
    def verify(stage):
        path = receipts / (stage + '.json')
        report = json.loads(path.read_text())
        assert report['source_manifest_sha256'] == fingerprint and report['stage'] == stage
        if stage not in verified:
            for name, expected in report['artifacts'].items():
                assert sha(root / name) == expected, ('Completed artifact changed', stage, name)
            verified.add(stage)
        return sha(path)
    for stage in STAGES if a.stage == 'all' else [a.stage]:
        check_environment(stage)
        prior = STAGES[:STAGES.index(stage)]
        dependencies = {p: verify(p) for p in prior}
        path = receipts / (stage + '.json')
        if path.exists():
            verify(stage)
            print('Already complete', stage, flush=True)
            continue
        started = time.time()
        print('TRAINING PIPELINE START', stage, flush=True)
        with (logs / (stage + '.log')).open('w') as log:
            subprocess.run(cmd[stage], cwd=root, env=environment(root, data, stage),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        files = {p for pattern in output_patterns(stage) for p in root.glob(pattern) if p.is_file()}
        assert files, ('No stage output', stage)
        report = dict(stage=stage, source_manifest_sha256=fingerprint, prior_receipt_sha256=dependencies,
            command=cmd[stage], seconds=time.time() - started,
            artifacts={str(p.relative_to(root)): sha(p) for p in sorted(files)}, public_upload=False)
        path.write_text(json.dumps(report, indent=2))
        verified.add(stage)
        print('TRAINING PIPELINE COMPLETE', stage, round(report['seconds']), flush=True)
    if all((receipts / (s + '.json')).exists() for s in STAGES):
        validation = json.loads((root / 'result/independent_validation.json').read_text())
        final = dict(all_training_and_inference_stages_complete=True, source_manifest_sha256=fingerprint,
            csv_sha256=sha(root / 'result/submission.csv'), validation=validation, public_upload=False,
            scope='Fitted models and inference results')
        (root / 'TRAINING_COMPLETE.json').write_text(json.dumps(final, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cpu-python', type=Path, default=Path(sys.executable))
    parser.add_argument('--gpu-python', type=Path, default=Path(sys.executable))
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--stage', choices=['all'] + STAGES, default='all')
    parser.add_argument('--plan', action='store_true')
    main(parser.parse_args())
