"""Exakte Kopie der offiziellen Metrik (Kernel florianderoofr/slash-poker-competition-metric).

NICHT durch eine aehnlich benannte Bibliotheksmetrik ersetzen. `score()` ist
woertlich uebernommen; darunter stehen schnelle Hilfsfunktionen fuer die lokale
Validierung, die gegen `score()` verifiziert werden (siehe tests/).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class ParticipantVisibleError(Exception):
    """An error message that Kaggle may safely show to participants."""


ALLOWED_BEHAVIORS = {
    "none", "directed_transfer", "soft_play", "coordinated_isolation", "other_coordination",
}
TARGET_BEHAVIORS = ("directed_transfer", "soft_play", "coordinated_isolation")
EVIDENCE_COLUMNS = tuple(f"evidence_hand_{rank}" for rank in range(1, 6))
NO_EVIDENCE = "NO_EVIDENCE"
REQUIRED_COLUMNS = {"pair_id", "risk_score", "predicted_behavior", *EVIDENCE_COLUMNS}


def _average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    positives = int(y_true.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    ranked = y_true[order]
    true_positives = np.cumsum(ranked)
    ranks = np.arange(1, len(ranked) + 1)
    return float(np.sum((true_positives / ranks) * ranked) / positives)


def _clean_evidence(values: list[object]) -> list[str]:
    cleaned = []
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text and text != NO_EVIDENCE:
            cleaned.append(text)
    return cleaned


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    if row_id_column_name != "pair_id":
        raise ParticipantVisibleError("The row ID column must be pair_id.")
    if not REQUIRED_COLUMNS.issubset(solution.columns):
        raise ValueError("The private solution has an invalid schema.")
    if not REQUIRED_COLUMNS.issubset(submission.columns):
        missing = sorted(REQUIRED_COLUMNS - set(submission.columns))
        raise ParticipantVisibleError(f"submission.csv is missing columns: {missing}")
    if solution["pair_id"].duplicated().any():
        raise ValueError("The private solution contains duplicate pair IDs.")
    if submission["pair_id"].duplicated().any():
        raise ParticipantVisibleError("pair_id values must be unique.")

    solution_ids = set(solution["pair_id"].astype(str))
    submission_ids = set(submission["pair_id"].astype(str))
    if solution_ids != submission_ids:
        missing = len(solution_ids - submission_ids)
        extra = len(submission_ids - solution_ids)
        raise ParticipantVisibleError(f"pair_id coverage mismatch: {missing} missing and {extra} extra.")

    truth = solution.set_index("pair_id").sort_index()
    predictions = submission.set_index("pair_id").loc[truth.index]

    risk = pd.to_numeric(predictions["risk_score"], errors="coerce")
    if risk.isna().any() or not risk.between(0, 1).all():
        raise ParticipantVisibleError("risk_score must be numeric and between 0 and 1.")

    predicted_behavior = predictions["predicted_behavior"].astype(str)
    invalid = set(predicted_behavior) - ALLOWED_BEHAVIORS
    if invalid:
        raise ParticipantVisibleError(f"Invalid predicted_behavior values: {sorted(invalid)}")

    for row in predictions.loc[:, EVIDENCE_COLUMNS].itertuples(index=False, name=None):
        evidence = _clean_evidence(list(row))
        if len(evidence) != len(set(evidence)):
            raise ParticipantVisibleError("Evidence hand IDs must not repeat within a pair.")

    y_true = pd.to_numeric(truth["risk_score"], errors="raise").to_numpy(dtype=int)
    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("Private risk_score values must be binary labels.")
    risk_values = risk.to_numpy(dtype=float)
    pair_ap = _average_precision(y_true, risk_values)

    true_behavior = truth["predicted_behavior"].astype(str).to_numpy()
    predicted_behavior_values = predicted_behavior.to_numpy()
    behavior_scores = []
    for behavior in TARGET_BEHAVIORS:
        behavior_truth = (true_behavior == behavior).astype(int)
        if behavior_truth.sum() == 0:
            behavior_scores.append(0.0)
            continue
        behavior_risk = np.where(predicted_behavior_values == behavior, risk_values, 0.0)
        behavior_scores.append(_average_precision(behavior_truth, behavior_risk))
    behavior_map = float(np.mean(behavior_scores))

    evidence_scores = []
    for position in np.flatnonzero(y_true == 1):
        relevant = set(_clean_evidence(truth.iloc[position].loc[list(EVIDENCE_COLUMNS)].tolist()))
        submitted = _clean_evidence(predictions.iloc[position].loc[list(EVIDENCE_COLUMNS)].tolist())
        if not relevant:
            evidence_scores.append(0.0)
            continue
        hits = 0
        precision_sum = 0.0
        for rank, hand_id in enumerate(submitted[:5], start=1):
            if hand_id in relevant:
                hits += 1
                precision_sum += hits / rank
        evidence_scores.append(precision_sum / min(len(relevant), 5))
    evidence_map = float(np.mean(evidence_scores)) if evidence_scores else 0.0

    final_score = 0.70 * pair_ap + 0.20 * evidence_map + 0.10 * behavior_map
    if not np.isfinite(final_score):
        raise ParticipantVisibleError("The metric produced a non-finite score.")
    return float(final_score)


# --------------------------------------------------------------------------
# Schnelle Komponenten fuer die lokale Validierung (gegen score() verifiziert)
# --------------------------------------------------------------------------

def fast_ap(y_true: np.ndarray, scores: np.ndarray) -> float:
    """AP mit exakt der offiziellen Tie-Behandlung (stabile Sortierung nach
    absteigendem Score; Gleichstaende in der uebergebenen Reihenfolge)."""
    return _average_precision(np.asarray(y_true, dtype=int), np.asarray(scores, dtype=float))


def fast_evidence_map(relevant_sets: list[set], submitted_lists: list[list]) -> float:
    """MAP@5 ueber die echten Positiv-Paare."""
    out = []
    for rel, sub in zip(relevant_sets, submitted_lists):
        if not rel:
            out.append(0.0)
            continue
        hits, psum = 0, 0.0
        for rank, h in enumerate(sub[:5], start=1):
            if h in rel:
                hits += 1
                psum += hits / rank
        out.append(psum / min(len(rel), 5))
    return float(np.mean(out)) if out else 0.0


def fast_behavior_map(true_behavior: np.ndarray, pred_behavior: np.ndarray, risk: np.ndarray) -> float:
    sc = []
    for b in TARGET_BEHAVIORS:
        bt = (true_behavior == b).astype(int)
        if bt.sum() == 0:
            sc.append(0.0)
            continue
        br = np.where(pred_behavior == b, risk, 0.0)
        sc.append(_average_precision(bt, br))
    return float(np.mean(sc))


def components(y_true, risk, true_behavior, pred_behavior, relevant_sets, submitted_lists):
    """Alle drei Komponenten + Gesamtscore in einem Aufruf."""
    pair_ap = fast_ap(y_true, risk)
    beh = fast_behavior_map(np.asarray(true_behavior), np.asarray(pred_behavior), np.asarray(risk, float))
    ev = fast_evidence_map(relevant_sets, submitted_lists)
    return {
        "pair_ap": pair_ap, "evidence_map": ev, "behavior_map": beh,
        "total": 0.70 * pair_ap + 0.20 * ev + 0.10 * beh,
    }
