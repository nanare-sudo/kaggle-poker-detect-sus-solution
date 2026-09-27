"""Install newly trained assets and create a data-free inference export."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def copy_new(source, target, expected=None):
    assert source.is_file() and not target.exists(), (source, target)
    if expected is not None:
        assert sha(source) == expected
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    assert sha(source) == sha(target)


def main(a):
    root = a.root.resolve()
    training, assets = root / 'training', root / 'assets'
    if a.stage == 'stats':
        sys.path.insert(0, str(root / 'src'))
        import action_context as Q
        assert not Q.STATS.exists()
        Q.build_stats()
    elif a.stage == 'install-policies':
        for name in ['classic', 'q2']:
            folder = training / 'action_references'
            report = json.loads((folder / (name + '.json')).read_text())
            assert not report['smoke']
            copy_new(folder / (name + '.txt'), assets / ('policy_' + name + '.txt'), report['model_sha256'])
    elif a.stage == 'install-neural':
        folder = training / 'neural'
        report = json.loads((folder / 'training.json').read_text())
        assert not report['specification']['smoke']
        assert report['specification']['epochs'] == 24
        for f in range(2):
            name = f'model_noply_f{f}.pt'
            copy_new(folder / 'models' / name, assets / 'neural' / name, report['models'][str(f)]['sha256'])
    elif a.stage == 'assemble-assets':
        pair = json.loads((training / 'pair_heads/training.json').read_text())
        assert pair['complete']
        for name in ['anchor', 'behavior']:
            assert not (assets / name).exists()
            shutil.copytree(training / 'pair_heads' / name, assets / name)
        instance = json.loads((training / 'instance_final/training.json').read_text())
        assert instance['status'] == 'PASS'
        copy_new(training / 'instance_final/instance.txt', assets / 'instance.txt', instance['model_sha256'])
        pool = json.loads((training / 'pool_heads/training.json').read_text())
        assert pool['status'] == 'PASS'
        for seed in [42, 43]:
            name = f'pool_seed{seed}.txt'
            copy_new(training / 'pool_heads' / name, assets / 'pool' / name, pool['models'][str(seed)]['sha256'])
        evidence = json.loads((training / 'evidence_heads/training.json').read_text())
        assert evidence['status'] == 'PASS'
        for g in range(3):
            for name, item in evidence['families'][str(g)]['models'].items():
                copy_new(training / 'evidence_heads' / name, assets / name, item['sha256'])
        teachers = json.loads((training / 'instance_teachers/training.json').read_text())
        assert teachers['status'] == 'PASS'
        config = json.loads((root / 'training_config.json').read_text())
        recipe = dict(tabicl_features=config['tabicl_features'],
            evidence_features=json.loads((training / 'evidence_inputs/feature_names.json').read_text()),
            pool_thresholds=teachers['q_oof_mean'])
        assert not (assets / 'recipe.json').exists()
        (assets / 'recipe.json').write_text(json.dumps(recipe, indent=2))
        manifest = {str(p.relative_to(assets)): sha(p) for p in assets.rglob('*') if p.is_file()}
        (root / 'MODEL_MANIFEST.json').write_text(json.dumps(dict(files=manifest,
            source='New training stages plus declared public TabICL checkpoint', public_upload=False), indent=2))
    elif a.stage == 'export':
        validation = json.loads((root / 'result/independent_validation.json').read_text())
        assert validation['sha256'] == sha(root / 'result/submission.csv')
        target = root / 'inference_export'
        assert not target.exists()
        target.mkdir()
        for name in ['src', 'pipeline', 'assets', 'licenses']:
            shutil.copytree(root / name, target / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        (target / 'features').mkdir()
        copy_new(root / 'features/sq_ph_names.json', target / 'features/sq_ph_names.json')
        copy_new(root / 'run_inference.py', target / 'run.py')
        for name in ['requirements-cpu.txt', 'requirements-gpu.txt']:
            copy_new(root / name, target / name)
        (target / 'README.md').write_text('''# Inference assets from a fresh source-based training run

Run `python run.py --data /path/to/official/data --cpu-python /path/to/cpu/python
--gpu-python /path/to/gpu/python --jobs 4` using the pinned requirements.
This export contains no competition raw data, training examples,
predictions or submission. The TabICL foundation checkpoint is a declared
public pretrained asset. Other fitted weights were generated by this run.
Neural training is stochastic. Stage receipts record the fitted model hashes.
''')
        files = {str(p.relative_to(target)): sha(p) for p in target.rglob('*') if p.is_file()}
        assert not any(n.endswith(('.csv', '.npz', '.parquet')) for n in files)
        (target / 'MANIFEST.json').write_text(json.dumps(dict(files=files,
            expected_csv_sha256=validation['sha256'], scope='Newly trained inference export', public_upload=False), indent=2))
    print('TRAINING SUPPORT COMPLETE', a.stage, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['stats', 'install-policies', 'install-neural', 'assemble-assets', 'export'])
    p.add_argument('--root', type=Path, required=True)
    main(p.parse_args())
