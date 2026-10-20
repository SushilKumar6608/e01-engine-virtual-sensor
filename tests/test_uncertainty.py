"""Conformal step, OOD score and abstain rule (SIMULATION PROJECT)."""
import numpy as np
import pandas as pd

from vsensor.models.uncertainty import AbstainRule, OODScorer, OOD_FEATURES, conformal_correction


def test_conformal_correction_matches_definition_and_coverage():
    rng = np.random.default_rng(0)
    n = 20000
    y = rng.normal(0, 1, n); lo = np.full(n, -0.5); hi = np.full(n, 0.5)
    q = conformal_correction(lo, hi, y, 0.1)
    score = np.sort(np.maximum(lo - y, y - hi))
    assert q == score[int(np.ceil((n + 1) * 0.9)) - 1]
    yt = rng.normal(0, 1, 200000)
    cov = np.mean((yt >= -0.5 - q) & (yt <= 0.5 + q))
    assert abs(cov - 0.9) < 0.01


def test_conformal_skips_missing_and_too_few_points():
    assert conformal_correction(np.zeros(3), np.zeros(3), np.array([1.0, np.nan, 2.0]), 0.1) == np.inf


def _F(n, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({c: rng.normal(0, 1, n) + shift for c in OOD_FEATURES})


def test_ood_score_grows_away_from_training():
    tr = _F(3000); code = np.zeros(3000, int); code[1500:] = 1
    s = OODScorer(k=5).fit(tr, code, np.ones(3000, bool))
    near = s.score(_F(500, 0, 1), np.zeros(500, int)); far = s.score(_F(500, 8, 2), np.zeros(500, int))
    assert far.mean() > 3 * near.mean()


def test_abstain_reasons():
    F = pd.DataFrame(dict(f_oor=[0, 1, 0, 0, 0], f_stuck=[0, 0, 1, 0, 0]))
    r = AbstainRule(ood_threshold=1.0, width_tolerance=30.0).decide(np.array([0.5, 0.5, 0.5, 2.0, 0.5]), np.array([10, 10, 10, 10, 40.0]), F)
    assert r["abstain"].tolist() == [False, True, True, True, True]
    assert r.loc[1, "input_out_of_range"] and r.loc[2, "input_stuck"] and r.loc[3, "far_from_training"] and r.loc[4, "interval_too_wide"]
