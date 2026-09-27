"""Verify release files and the ordered TabICL schemas using the standard library."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def verify(root, source_only=False):
    spec = json.loads((root / 'SOLUTION_MANIFEST.json').read_text())
    models = json.loads((root / 'MODELS.json').read_text())['files']
    count = 0
    for name, expected in spec['files'].items():
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError(f'Invalid release path: {name}')
        if source_only and name in models:
            continue
        path = root / rel
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'Missing release file: {name}')
        if sha(path) != expected:
            raise ValueError(f'Checksum mismatch: {name}')
        if path.suffix == '.py':
            compile(path.read_text(), name, 'exec')
        count += 1
    recipe = json.loads((root / 'package/assets/recipe.json').read_text())
    assert len(recipe['tabicl_features']) == 96
    assert len(recipe['tabicl_features192']) == 192
    assert recipe['tabicl_features192'][:96] == recipe['tabicl_features']
    return dict(files_verified=count, models_required=not source_only,
                tabicl_schemas='96 and 192 columns, shared ordered prefix')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--source-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(args.root.resolve(), args.source_only), indent=2))
