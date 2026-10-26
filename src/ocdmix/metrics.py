"""Evaluation metrics against ground truth for change-point detection."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class StreamEvaluation:
    """Per-stream evaluation record."""
    stream_id: int
    b: Optional[int]
    alarm_N: Optional[int]
    false_alarm: bool
    detected: bool
    delay: Optional[int]
    nu_hat: Optional[int]
    abs_error: Optional[int]
    ci_covers: Optional[bool]
    ci_width: Optional[int]
    support_precision: Optional[float]
    support_recall: Optional[float]
    support_jaccard: Optional[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_single_stream(
    stream_id: int,
    ground_truth: dict[str, Any],
    alarm_N: Optional[int],
    nu_hat: Optional[int] = None,
    CI: Optional[tuple[int, int]] = None,
    support: Optional[list[int]] = None,
) -> StreamEvaluation:
    """Evaluate a single stream against ground truth."""
    b = ground_truth.get("changepoint_index_zero_based")
    s_true_list = ground_truth.get("affected_coordinates_zero_based") or []
    s_true = set(s_true_list)

    if b is None:
        # Null stream
        is_false_alarm = alarm_N is not None
        return StreamEvaluation(
            stream_id=stream_id,
            b=None,
            alarm_N=alarm_N,
            false_alarm=is_false_alarm,
            detected=False,
            delay=None,
            nu_hat=nu_hat,
            abs_error=None,
            ci_covers=None,
            ci_width=None,
            support_precision=None,
            support_recall=None,
            support_jaccard=None,
        )

    # Changed stream
    if alarm_N is None:
        # No alarm before horizon T
        return StreamEvaluation(
            stream_id=stream_id,
            b=b,
            alarm_N=None,
            false_alarm=False,
            detected=False,
            delay=None,
            nu_hat=None,
            abs_error=None,
            ci_covers=False,
            ci_width=None,
            support_precision=None,
            support_recall=None,
            support_jaccard=None,
        )

    if alarm_N < b:
        # False alarm before changepoint
        return StreamEvaluation(
            stream_id=stream_id,
            b=b,
            alarm_N=alarm_N,
            false_alarm=True,
            detected=False,
            delay=None,
            nu_hat=nu_hat,
            abs_error=None,
            ci_covers=None,
            ci_width=None,
            support_precision=None,
            support_recall=None,
            support_jaccard=None,
        )

    # Detected: alarm_N >= b
    delay = alarm_N - b
    abs_err = abs(nu_hat - b) if nu_hat is not None else None

    if CI is not None:
        covers = bool(CI[0] <= b <= CI[1])
        ci_w = int(CI[1] - CI[0])
    else:
        covers = None
        ci_w = None

    if support is not None and len(s_true) > 0:
        s_pred = set(support)
        tp = len(s_pred & s_true)
        precision = float(tp / len(s_pred)) if len(s_pred) > 0 else 0.0
        recall = float(tp / len(s_true))
        union = len(s_pred | s_true)
        jaccard = float(tp / union) if union > 0 else 0.0
    elif len(s_true) == 0:
        s_pred = set(support or [])
        precision = 1.0 if len(s_pred) == 0 else 0.0
        recall = 1.0
        jaccard = 1.0 if len(s_pred) == 0 else 0.0
    else:
        precision = None
        recall = None
        jaccard = None

    return StreamEvaluation(
        stream_id=stream_id,
        b=b,
        alarm_N=alarm_N,
        false_alarm=False,
        detected=True,
        delay=delay,
        nu_hat=nu_hat,
        abs_error=abs_err,
        ci_covers=covers,
        ci_width=ci_w,
        support_precision=precision,
        support_recall=recall,
        support_jaccard=jaccard,
    )


def summarize_evaluations(
    records: list[StreamEvaluation],
    is_null: bool,
    m0: int,
    T: int,
) -> dict[str, Any]:
    """Compute aggregate summary metrics over stream records."""
    n = len(records)
    if n == 0:
        return {}

    if is_null:
        # False-alarm probability over the horizon (null)
        n_fa = sum(1 for r in records if r.false_alarm)
        far = n_fa / n
        far_se = math.sqrt(far * (1.0 - far) / n) if n > 0 else 0.0

        # Censored ARL estimate: run length is N - m0 + 1 if alarm else T - m0
        run_lengths = [
            (r.alarm_N - m0 + 1) if r.alarm_N is not None else (T - m0)
            for r in records
        ]
        censored_arl = float(np.mean(run_lengths))

        return {
            "n_streams": n,
            "false_alarm_probability": far,
            "false_alarm_probability_se": far_se,
            "censored_arl": censored_arl,
        }

    # Changed streams
    n_fa = sum(1 for r in records if r.false_alarm)
    far = n_fa / n
    far_se = math.sqrt(far * (1.0 - far) / n)

    n_det = sum(1 for r in records if r.detected)
    det_rate = n_det / n
    det_rate_se = math.sqrt(det_rate * (1.0 - det_rate) / n)

    delays = [r.delay for r in records if r.detected and r.delay is not None]
    if delays:
        mean_delay = float(np.mean(delays))
        median_delay = float(np.median(delays))
        p90_delay = float(np.percentile(delays, 90))
    else:
        mean_delay = None
        median_delay = None
        p90_delay = None

    # Coverage among detected streams with valid CI
    cov_list = [r.ci_covers for r in records if r.detected and r.ci_covers is not None]
    if cov_list:
        coverage = float(np.mean(cov_list))
        coverage_se = math.sqrt(coverage * (1.0 - coverage) / len(cov_list))
    else:
        coverage = None
        coverage_se = None

    ci_widths = [r.ci_width for r in records if r.detected and r.ci_width is not None]
    mean_ci_width = float(np.mean(ci_widths)) if ci_widths else None

    precisions = [r.support_precision for r in records if r.detected and r.support_precision is not None]
    recalls = [r.support_recall for r in records if r.detected and r.support_recall is not None]
    jaccards = [r.support_jaccard for r in records if r.detected and r.support_jaccard is not None]

    mean_precision = float(np.mean(precisions)) if precisions else None
    mean_recall = float(np.mean(recalls)) if recalls else None
    mean_jaccard = float(np.mean(jaccards)) if jaccards else None

    abs_errors = [r.abs_error for r in records if r.detected and r.abs_error is not None]
    mean_abs_error = float(np.mean(abs_errors)) if abs_errors else None
    median_abs_error = float(np.median(abs_errors)) if abs_errors else None

    return {
        "n_streams": n,
        "n_false_alarm": n_fa,
        "false_alarm_rate": far,
        "false_alarm_rate_se": far_se,
        "n_detected": n_det,
        "detection_rate": det_rate,
        "detection_rate_se": det_rate_se,
        "mean_delay": mean_delay,
        "median_delay": median_delay,
        "p90_delay": p90_delay,
        "coverage": coverage,
        "coverage_se": coverage_se,
        "mean_ci_width": mean_ci_width,
        "mean_support_precision": mean_precision,
        "mean_support_recall": mean_recall,
        "mean_support_jaccard": mean_jaccard,
        "mean_abs_error": mean_abs_error,
        "median_abs_error": median_abs_error,
    }
