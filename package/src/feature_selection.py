"""Gain selection and ordered materialization of pair features."""
from pathlib import Path
import numpy as np
import derive
import validation as V
FEAT=Path(__file__).resolve().parents[1]/'features'
DEV_TAG='dev'

def _select_columns(k=1200, sub_rows=45000, seed=0, tag=None, train_mask=None,
                    selector_jobs=None):
    """Waehlt auf einer Zeilen-Teilstichprobe die k wichtigsten Spalten aus.

    Rueckgabe: (pair_keys, y, fam, tab, universe, Rohnamen, behaltene Rohspalten,
                endgueltige Namensliste nach derive).
    Die Eingabespalten von derive.py werden IMMER behalten, sonst lassen sich die
    Ratenkontraste spaeter nicht mehr berechnen.
    Bei Kreuzvalidierung MUSS train_mask die Trainingstische des jeweiligen
    Folds markieren. Ohne Maske ist diese Funktion nur fuer das finale Training.
    """
    import model as M
    tag = tag or DEV_TAG
    key, y, fam, tab, universe, names0 = V.load_dev_meta(tag[3:])
    allowed = np.ones(len(y), dtype=bool) if train_mask is None else np.asarray(train_mask)
    if allowed.shape != y.shape or allowed.dtype != np.bool_:
        raise ValueError("train_mask muss eine boolesche Maske aller Paarzeilen sein")
    if not np.any(allowed & (y == 1)):
        raise ValueError("Merkmalsauswahl braucht positive TRAININGSpaare")
    mm = np.load(FEAT / f"pair_X_{tag}.npy", mmap_mode="r")
    rng = np.random.default_rng(seed)
    lab_rows = np.flatnonzero(allowed & (y >= 0))
    unl = np.flatnonzero(allowed & universe & (y < 0))
    take = np.union1d(lab_rows, rng.choice(unl, min(sub_rows, len(unl)), replace=False))
    Xs = np.asarray(mm[take])
    del mm
    Xs, nd = derive.add_ratio_features(Xs, names0)
    sel = M.select_features(Xs, y[take], nd, np.ones(len(take), bool), k=k,
                            sub=len(take), n_jobs=selector_jobs)
    del Xs
    # Spalten, die derive braucht, unbedingt behalten
    need = set()
    for num, den, _tag in derive.RATIOS:
        for side in ("pa", "pd"):
            need |= {f"{side}_{num}__mean", f"{side}_{den}__mean"}
    for num, den, _t in [("foldto_sum", "faced_sum", ""), ("callto_sum", "faced_sum", ""),
                         ("reraise_sum", "faced_sum", ""), ("q_foldbest_sum", "faced_sum", ""),
                         ("squeeze_fold", "squeeze_count", "")]:
        need |= {f"{num}__mean", f"{den}__mean"}
    idx0 = {n: i for i, n in enumerate(names0)}
    keep = {i for i in sel if i < len(names0)}
    keep |= {idx0[n] for n in need if n in idx0}
    base_keep = np.array(sorted(keep))
    final_names = derive.add_ratio_features(
        np.zeros((1, len(base_keep)), np.float32), [names0[i] for i in base_keep])[1]
    print(f"Merkmalsvorauswahl: {len(base_keep)} Rohspalten von {len(names0)} "
          f"-> {len(final_names)} nach derive", flush=True)
    return key, y, fam, tab, universe, names0, base_keep, final_names

def _materialize(tag, base_keep, names0, names, rows=None):
    mm = np.load(FEAT / f"pair_X_{tag}.npy", mmap_mode="r")
    X = np.asarray(mm[:, base_keep]) if rows is None else np.asarray(mm[np.ix_(rows, base_keep)])
    del mm
    nm = [names0[i] for i in base_keep]
    X, nm2 = derive.add_ratio_features(X, nm)
    assert nm2 == names, "Spaltenreihenfolge dev/eval weicht ab"
    return np.ascontiguousarray(X, dtype=np.float32)
