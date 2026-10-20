"""Feature tests: causality, no hidden states, one function for all variants (SIMULATION PROJECT)."""
import numpy as np
import pandas as pd

from vsensor.features.build import DYN, FLAGS, PHYS, RAW, VARIANTS, build_features
from vsensor.params import load_registry

REG = load_registry()


def _df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return pd.DataFrame(dict(
        episode_id=np.repeat(["a", "b"], n // 2), t_s=np.tile(t[: n // 2], 2), fuel_code=np.repeat([0, 1], n // 2),
        n_rpm=100 + rng.normal(0, 1, n), m_fuel=0.2 + 0.02 * rng.normal(size=n) + np.where(t % 150 > 75, 0.05, 0),
        p_scav=2.8 + 0.01 * rng.normal(size=n), t_scav=300 + rng.normal(0, 0.5, n), n_tc=np.where(t % 50 == 3, np.nan, 12000 + rng.normal(0, 20, n)),
        t_amb=290 + rng.normal(0, 0.2, n), p_amb=1.013 + 0 * t))


def test_variants_are_nested_and_complete():
    assert VARIANTS["B"][: len(RAW)] == RAW and set(VARIANTS["D"]) == set(RAW + PHYS + DYN + FLAGS)
    F = build_features(_df(), np.full(300, 700.0), REG)
    assert list(F.columns) == RAW + PHYS + DYN + FLAGS and np.isfinite(F.to_numpy()).all()


def test_features_are_causal():
    """Changing inputs after time T must not change any feature at or before T."""
    df = _df(); T = 100
    a = build_features(df, np.full(300, 700.0), REG)
    df2 = df.copy(); m = (df2.episode_id == "a") & (df2.t_s > T)
    df2.loc[m, ["m_fuel", "n_rpm", "p_scav", "t_scav", "n_tc"]] *= 1.3
    b = build_features(df2, np.full(300, 700.0), REG)
    early = ((df.episode_id == "a") & (df.t_s <= T)).to_numpy()
    assert np.allclose(a[early].to_numpy(), b[early].to_numpy())
    assert np.allclose(a[(df.episode_id == "b").to_numpy()].to_numpy(), b[(df.episode_id == "b").to_numpy()].to_numpy())


def test_no_hidden_or_target_columns_used():
    bad = ("foul", "true", "steady", "load_fraction", "t_exh", "fault", "delivery")
    assert not [c for c in sum(VARIANTS.values(), []) if any(b in c for b in bad)]


def test_dropout_is_forward_filled_and_flagged():
    df = _df(); F = build_features(df, np.full(300, 700.0), REG)
    i = np.where(df.n_tc.isna())[0][1]
    assert F.f_n_tc_missing.iloc[i] == 1 and F.n_tc.iloc[i] == df.n_tc.iloc[i - 1]
