"""Abgeleitete Verhaeltnis-Merkmale auf Paar-Ebene.

Der Motor liefert ZAEHLER je gemeinsamer Hand. Die diskriminierende Groesse ist
aber eine RATE und ihr Kontrast, z.B.

    Flat-Call-Ueberschuss = P(call | praeflop, Partner noch im Pot)
                          - P(call | praeflop, Partner bereits gefoldet)

Beides sind Aktionen desselben Spielers in denselben Haenden — eine interne
Kontrolle, die Spielerstil, Tischdynamik und Blindstufe automatisch herausrechnet.
Da `mean` fuer alle Merkmale eines Paares durch dieselbe Handzahl teilt, ist
mean(a)/mean(b) exakt sum(a)/sum(b); die Raten sind also aus den __mean-Spalten
exakt rekonstruierbar.
"""
from __future__ import annotations

import numpy as np

# (Zaehler, Nenner, Kurzname) — je Paar wird alive/dead und die Differenz gebildet
RATIOS = [
    ("pre_call", "pre_n", "preCall"),
    ("pre_fold", "pre_n", "preFold"),
    ("pre_aggr", "pre_n", "preAggr"),
    ("prew_call", "prew_n", "preWeakCall"),
    ("prew_fold", "prew_n", "preWeakFold"),
    ("post_check", "post_n", "postCheck"),
    ("post_call", "post_n", "postCall"),
    ("post_aggr", "post_n", "postAggr"),
    ("post_fold", "post_n", "postFold"),
    ("bestpass", "bestn", "bestPassive"),
]
EPS = 1e-6


def add_ratio_features(F: np.ndarray, names: list[str]):
    idx = {n: i for i, n in enumerate(names)}
    cols, new = [], []

    def col(nm):
        return F[:, idx[nm]]

    for num, den, tag in RATIOS:
        for side in ("pa", "pd"):
            n_ = f"{side}_{num}__mean"
            d_ = f"{side}_{den}__mean"
            if n_ not in idx or d_ not in idx:
                break
        else:
            ra = col(f"pa_{num}__mean") / np.maximum(col(f"pa_{den}__mean"), EPS)
            rd = col(f"pd_{num}__mean") / np.maximum(col(f"pd_{den}__mean"), EPS)
            cols += [ra, rd, ra - rd]
            new += [f"rate_{tag}_alive", f"rate_{tag}_dead", f"excess_{tag}"]
            # Exposition: wie belastbar ist der Kontrast?
            cols += [col(f"pa_{den}__mean"), col(f"pd_{den}__mean")]
            new += [f"opp_{tag}_alive", f"opp_{tag}_dead"]

    # Equity beim Fold/Call gegen den Partner gegen dieselbe Groesse gegen alle
    # uebrigen Gegner — der eigentliche Kontrast, nicht die Rohhoehe.
    for nm, tag in (("eqf", "eqFold"), ("eqc", "eqCall")):
        a, b = f"{nm}_sum_p__mean", f"{nm}_n_p__mean"
        c, d = f"{nm}_sum_o__mean", f"{nm}_n_o__mean"
        if all(x in idx for x in (a, b, c, d)):
            rp = col(a) / np.maximum(col(b), EPS)
            ro = col(c) / np.maximum(col(d), EPS)
            cols += [rp, ro, rp - ro, col(b), col(d)]
            new += [f"rate_{tag}_partner", f"rate_{tag}_others", f"excess_{tag}",
                    f"opp_{tag}_partner", f"opp_{tag}_others"]

    # Konfrontationsraten gegen den Partner, normiert auf die Gelegenheiten
    for num, den, tag in [("foldto_sum", "faced_sum", "foldToPartner"),
                          ("callto_sum", "faced_sum", "callToPartner"),
                          ("reraise_sum", "faced_sum", "reraisePartner"),
                          ("q_foldbest_sum", "faced_sum", "foldBestPartner"),
                          ("squeeze_fold", "squeeze_count", "squeezeFold")]:
        a, b = f"{num}__mean", f"{den}__mean"
        if a in idx and b in idx:
            cols.append(col(a) / np.maximum(col(b), EPS))
            new.append(f"rate_{tag}")

    if not cols:
        return F, names
    return np.concatenate([F, np.stack(cols, axis=1).astype(np.float32)], axis=1), names + new
