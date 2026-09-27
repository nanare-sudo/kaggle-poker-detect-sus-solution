from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'package/src'),
               str(ROOT / 'package/pipeline'), str(ROOT / 's5/source')]
