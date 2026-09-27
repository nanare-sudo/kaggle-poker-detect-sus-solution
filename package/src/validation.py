"""Fixed table folds and development pair labels (unknown pairs retain label -1)."""
from pathlib import Path
import json,re
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'cache'
FEAT=ROOT/'features'
NP_PLAYERS=12000
MIN_SHARED_DEV=57

def _is_window(suffix: str) -> bool:
    """Recognize exposure windows by a suffix such as _w2000."""
    return re.search(r"_w\d+", suffix) is not None

def table_folds(tab: np.ndarray, n_splits: int = 5, seed: int = 0):
    """Tische deterministisch, aber pseudo-zufaellig auf Folds verteilen.

    Die Tischnummer selbst ist ein ID-Artefakt; sie wird ausschliesslich zur
    Gruppierung benutzt, nie als Merkmal. Um keine Tischnummern-Systematik
    (z.B. Blindstufen-Bloecke) in die Folds zu tragen, wird permutiert.
    """
    rng = np.random.default_rng(seed)
    n_tables = 400
    perm = rng.permutation(n_tables)
    fold_of_table = {t: int(perm[t] % n_splits) for t in range(n_tables)}
    tab = np.asarray(tab)
    assert tab.min() >= 0 and tab.max() < n_tables, "table_folds: Tischnummer ausserhalb 0..399"
    return np.array([fold_of_table[int(t)] for t in tab], dtype=np.int8)

def load_dev_meta(suffix: str = ""):
    """Load labels, tables and shared-hand exposure without materializing all features."""
    tag = "dev" + suffix
    key = np.load(FEAT / f"pair_key_{tag}.npy")
    names0 = json.loads((FEAT / f"pair_names_{tag}.json").read_text())
    lab = pd.read_parquet(CACHE / "labels.parquet")
    lk = lab.plo.to_numpy().astype(np.int64) * NP_PLAYERS + lab.phi.to_numpy()
    pos_in = np.searchsorted(key, lk)
    hit = (pos_in < len(key)) & (key[np.minimum(pos_in, len(key) - 1)] == lk)
    if not hit.all():
        # Im verkuerzten Fenster koennen Paare ganz ohne gemeinsame Hand fehlen.
        assert _is_window(suffix), "gelabeltes Paar fehlt im Universum"
        print(f"[load_dev_meta{suffix}] {int((~hit).sum())} gelabelte Paare fehlen "
              f"im Fenster ganz")
    y = np.full(len(key), -1, dtype=np.int8)
    y[pos_in[hit]] = lab.label.to_numpy()[hit]
    fam = np.full(len(key), "", dtype=object)
    fam[pos_in[hit]] = lab.behavior_family.to_numpy()[hit]
    sp = np.load(CACHE / "seat_player.npy", mmap_mode="r")
    ht = np.load(CACHE / "hand_table.npy", mmap_mode="r")
    ptab = np.full(NP_PLAYERS, -1, dtype=np.int16)
    for c in range(6):
        ptab[np.asarray(sp[:, c])] = np.asarray(ht)
    tab = ptab[(key // NP_PLAYERS).astype(np.int64)]
    # n_shared direkt aus dem Paar-Universum zaehlen. Eine einzelne Spalte aus
    # der C-geordneten Merkmalsdatei zu lesen wuerde die kompletten 2,6 GB in den
    # Seitencache ziehen — das Zaehlen kostet 10 s und ein paar hundert MB.
    if _is_window(suffix):
        # Im Fenster stimmt die Zaehlung ueber die volle Phase nicht mehr. Die
        # Spalte aus der Merkmalsdatei ist hier die einzige korrekte Quelle.
        mm = np.load(FEAT / f"pair_X_{tag}.npy", mmap_mode="r")
        n_shared = np.asarray(mm[:, names0.index("n_shared")], dtype=np.float64)
        del mm
    else:
        import pairs as P
        uniq, cnt, *_ = P.build_universe(0)
        ns = pd.Series(cnt, index=uniq)
        n_shared = ns.reindex(key).to_numpy()
    assert not np.isnan(n_shared).any(), "Paar ohne gezaehlte gemeinsame Haende"
    universe = n_shared >= (38 if _is_window(suffix) else MIN_SHARED_DEV)
    return key, y, fam, tab, universe, names0
