from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import zipfile

import pytest

from download_models import install
from verify_release import verify
import train_current

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('script,args,stages', [
    ('run_solution.py', ['--data', '/example/data'], 23),
    ('train_current.py', ['--data', '/example/data', '--out', '/example/train'], 37),
    ('train_s5.py', ['--data', '/example/data', '--base-run', '/example/train', '--out', '/example/s5'], 14),
])
def test_command_plans(script, args, stages):
    completed = subprocess.run([sys.executable, '-B', str(ROOT / script), *args, '--plan'],
                               check=True, capture_output=True, text=True)
    plan = json.loads(completed.stdout)
    assert len(plan['stages']) == stages


def test_source_release_hashes():
    assert verify(ROOT, source_only=True)['files_verified'] > 100


def test_hand_feature_import_preserves_reference_defaults():
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'package/src'))
    for name in ['SEQ_FEAT', 'POLICY_Q2', 'POLICY_Q2_SFX']:
        env.pop(name, None)
    code = ('import hand_features as h; '
            'assert h.engine.POLICY_Q2; '
            'assert h.engine.POLICY_Q2_SFX == "_q2"; '
            'assert h.seqwire.enabled()')
    subprocess.run([sys.executable, '-B', '-c', code], env=env, check=True, capture_output=True)


def test_model_installer_checks_content_and_preserves_existing_files(tmp_path):
    data = b'synthetic test weights'
    digest = hashlib.sha256(data).hexdigest()
    (tmp_path / 'MODELS.json').write_text(json.dumps({'files': {'assets/test.bin': digest}}))
    archive = tmp_path / 'weights.zip'
    with zipfile.ZipFile(archive, 'w') as zipped:
        zipped.writestr('sha256/' + digest, data)
    assert install(archive, tmp_path) == 1
    assert install(archive, tmp_path) == 0
    (tmp_path / 'assets/test.bin').write_bytes(b'locally changed')
    with pytest.raises(ValueError, match='Refusing to replace'):
        install(archive, tmp_path)
    assert (tmp_path / 'assets/test.bin').read_bytes() == b'locally changed'


def test_model_installer_rejects_corrupt_archive(tmp_path):
    digest = hashlib.sha256(b'expected').hexdigest()
    (tmp_path / 'MODELS.json').write_text(json.dumps({'files': {'assets/test.bin': digest}}))
    archive = tmp_path / 'weights.zip'
    with zipfile.ZipFile(archive, 'w') as zipped:
        zipped.writestr('sha256/' + digest, b'incorrect')
    with pytest.raises(ValueError, match='checksum'):
        install(archive, tmp_path)
    assert not (tmp_path / 'assets/test.bin').exists()


def test_training_initialization_copies_only_manifest_files(tmp_path, monkeypatch):
    project = tmp_path / 'project'
    package = project / 'package'
    (package / 'assets/neural').mkdir(parents=True)
    recipe = {'tabicl_features': ['a'], 'tabicl_features192': ['a', 'b']}
    (package / 'assets/recipe.json').write_text(json.dumps(recipe))
    (package / 'model_code.py').write_text('MODEL_WIDTH = 4\n')
    (package / 'not_in_release.txt').write_text('local file')
    files = {name: hashlib.sha256((package / name).read_bytes()).hexdigest()
             for name in ['assets/recipe.json', 'model_code.py']}
    (package / 'MANIFEST.json').write_text(json.dumps({'files': files}))
    fitted = tmp_path / 'fitted'
    for directory in ['legacy/cache', 'legacy/assets/neural', 'action_current/RCLEAN']:
        (fitted / directory).mkdir(parents=True)
    (fitted / 'legacy/cache/array.npy').write_bytes(b'synthetic cache')
    (fitted / 'legacy/assets/recipe.json').write_text(json.dumps(recipe))
    (fitted / 'legacy/assets/neural/model.pt').write_bytes(b'synthetic fitted model')
    for fold in (0, 1):
        (fitted / f'action_current/RCLEAN/model_noply_f{fold}.pt').write_bytes(b'corrected model')
    for name in ['surprise', 'paggr', 'pfold']:
        (fitted / f'action_current/RCLEAN/act_{name}_q2nnnoply.npy').write_bytes(b'corrected features')
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'sample_submission.csv').write_text('pair_id\nexample\n')
    monkeypatch.setattr(train_current, 'HERE', project)
    train_current.initialize(fitted, data)
    current = fitted / 'current'
    assert not (current / 'not_in_release.txt').exists()
    assert (current / 'model_code.py').read_text() == 'MODEL_WIDTH = 4\n'
    assert (current / 'cache/array.npy').is_symlink()
    assert (current / 'assets/neural/model.pt').read_bytes() == b'synthetic fitted model'
    assert (current / 'assets/neural/model_rclean_f0.pt').read_bytes() == b'corrected model'
    with pytest.raises(AssertionError):
        train_current.initialize(fitted, data)
