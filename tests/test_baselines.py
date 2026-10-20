"""Stage 3 tests: unit adapter, sealed splits, baselines (SIMULATION PROJECT)."""
import numpy as np
import pandas as pd
import pytest

from vsensor.eval import harness
from vsensor.models.baselines import BAR, LoadLookup, M0, M0b, PerFuelConstant, to_si
from vsensor.params import load_registry
from vsensor.sim.nominal import NominalModel

REG = load_registry()


def _frame(n=50):
    rng = np.random.default_rng(1)
    return pd.DataFrame(dict(
        episode_id=np.repeat(["a", "b"], n // 2), t_s=np.tile(np.arange(n // 2), 2),
        fuel_code=np.repeat([0, 1], n // 2), n_rpm=rng.uniform(100, 120, n), m_fuel=rng.uniform(0.15, 0.25, n),
        p_scav=rng.uniform(2.6, 3.2, n), t_scav=rng.uniform(290, 320, n), p_amb=np.full(n, 1.013),
        t_exh_meas=rng.uniform(650, 780, n)))


def test_unit_adapter_bar_to_pa():
    df = _frame()
    s = to_si(df)
    assert np.allclose(s["p_scav"], df["p_scav"] * 1e5) and BAR == 1e5
    direct = NominalModel(REG).predict_by_code(df.n_rpm, df.m_fuel, df.p_scav * 1e5, df.t_scav, df.fuel_code)
    via = M0(REG)
    via.theta = np.array([REG["nominal.lambda_del"]])
    assert np.allclose(via.predict(df), direct)
    # forgetting the conversion (bar passed as Pa) starves the engine of air: the model must fail loudly
    with pytest.raises(ValueError):
        NominalModel(REG).predict_by_code(df.n_rpm, df.m_fuel, df.p_scav, df.t_scav, df.fuel_code)


def test_m0b_constant_equals_m0():
    df = _frame()
    b = M0b(REG); b.theta = np.array([0.95, 0.0])
    m = M0(REG); m.theta = np.array([0.95])
    assert np.allclose(b.predict(df), m.predict(df))


def test_sealed_splits_refused_and_logged(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "UNSEAL_LOG", tmp_path / "log.csv")
    for s in ("test_in_distribution", "stress_low_load"):
        with pytest.raises(harness.SealedSplitError):
            harness.load_eval("does-not-matter", s)
    assert not (tmp_path / "log.csv").exists()


def test_constant_and_lookup_clamp():
    df = _frame(); tr = df.copy()
    c = PerFuelConstant().fit(tr)
    assert set(np.unique(c.predict(df))) == set(tr.groupby("fuel_code")["t_exh_meas"].mean())
    lk = LoadLookup(REG, bins=3).fit(tr)
    far = df.copy(); far["m_fuel"] = 50.0
    assert np.all(np.isfinite(lk.predict(far))) and lk.predict(far).max() <= tr.t_exh_meas.max()


def test_bootstrap_is_by_episode_and_zero_for_perfect():
    df = _frame()
    df["t_exh_true"] = df["t_exh_meas"]
    m = harness.metrics(df, df["t_exh_meas"].to_numpy())
    assert m["rmse"] == 0 and m["episodes"] == 2 and m["rows"] == len(df)
