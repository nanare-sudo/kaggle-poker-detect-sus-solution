"""Install the separately distributed model archive with SHA-256 verification."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.request import urlopen
import zipfile

ROOT = Path(__file__).resolve().parent


def install(archive, root=ROOT):
    files = json.loads((root / 'MODELS.json').read_text())['files']
    installed = 0
    with zipfile.ZipFile(archive) as zipped:
        for name, digest in files.items():
            rel = Path(name)
            if rel.is_absolute() or '..' in rel.parts:
                raise ValueError(f'Invalid model path: {name}')
            dest = root / rel
            if not dest.resolve().is_relative_to(root.resolve()):
                raise ValueError(f'Model path escapes the project: {name}')
            if dest.exists():
                if hashlib.sha256(dest.read_bytes()).hexdigest() == digest:
                    continue
                raise ValueError(f'Refusing to replace a different model: {name}')
            data = zipped.read('sha256/' + digest)
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f'Model checksum mismatch: {name}')
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open('xb') as stream:
                stream.write(data)
            installed += 1
    return installed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--archive', type=Path, help='Local model ZIP')
    group.add_argument('--url', help='URL of the model ZIP from this release')
    args = parser.parse_args()
    if args.archive:
        count = install(args.archive)
    else:
        with tempfile.TemporaryFile() as stream:
            with urlopen(args.url, timeout=60) as response:
                while block := response.read(8 << 20):
                    stream.write(block)
            stream.seek(0)
            count = install(stream)
    print(f'Installed {count} model files; all model checksums verified.')


if __name__ == '__main__':
    main()
