"""Tests for the dataset configuration, sensors, episode generator and hashing (SIMULATED DATA)."""
import numpy as np
import pytest

from vsensor.data.config import load_dataset_config
from vsensor.data.episodes import simulate_episode
from vsensor.data.generate import MODEL_INPUT_COLUMNS, build_dataset, episode_digest, load_manifest, load_split, plant_for, registry_hash
from vsensor.params import load_registry
from vsensor.sim.sensors import SensorSpecs, first_order_lag, quantise

CFG, REG = load_dataset_config(), load_registry()
SP = SensorSpecs.from_registry(REG)


def episode(split="train", i=0, cfg=CFG):
    return simulate_episode(split, cfg.splits[split], i, cfg, REG, plant_for(cfg.splits[split], REG, {}), SP)


def test_config_rules():
    assert CFG.sampling_hz == 1.0
    for s in CFG.splits.values():
        assert abs(sum(s.fuel_mix.values()) - 1.0) < 1e-9
    assert len({s.seed_id for s in CFG.splits.values()}) == len(CFG.splits)


def test_lag_is_exact_first_order():
    y = first_order_lag(np.concatenate([np.zeros(10), np.ones(200)]), 10.0)
    assert y[-1] == pytest.approx(1.0, abs=1e-6)
    assert y[10 + 10] == pytest.approx(1 - np.exp(-1.0), abs=0.06)


def test_quantise_snaps_to_grid():
    q = quantise(np.array([1.234, 1.236]), 0.01)
    assert np.allclose(q, [1.23, 1.24])


def test_episode_reproducible_and_seed_dependent():
    a, b, c = episode(), episode(), episode(i=1)
    assert episode_digest(a) == episode_digest(b)
    assert episode_digest(a) != episode_digest(c)


def test_model_inputs_exclude_truth_and_target():
    ep = episode()
    assert set(MODEL_INPUT_COLUMNS) <= set(ep.signals)
    assert "t_exh_true" not in MODEL_INPUT_COLUMNS and "t_exh_meas" not in MODEL_INPUT_COLUMNS
    assert not any("foul" in c for c in MODEL_INPUT_COLUMNS)


def test_measured_close_to_truth_but_not_equal():
    ep = episode()
    d = ep.signals["t_exh_meas"] - ep.truth["t_exh_true"]
    assert 0.1 < np.nanstd(d) < 20.0


def test_fault_episode_marks_fault():
    ep = episode("test_sensor_fault", 2)
    assert ep.truth["fault_active"].any() or ep.meta.get("fault_kind") == "none"


def test_registry_hash_ignores_nominal_fit():
    reg2 = REG.with_overrides(nominal__lambda_del=REG["nominal.lambda_del"] * 1.01)
    assert registry_hash(REG) == registry_hash(reg2)


def test_build_one_split_twice_gives_same_hash(tmp_path):
    name = "stress_methanol_efficiency_plus"
    a = build_dataset(CFG, REG, tmp_path / "a", splits=[name], progress=False)
    b = build_dataset(CFG, REG, tmp_path / "b", splits=[name], progress=False)
    assert a["dataset_hash"] == b["dataset_hash"]
    assert len(load_split(tmp_path / "a", name)) == CFG.splits[name].episodes * int(CFG.episode_duration_s)
    assert load_manifest(tmp_path / "a")["dataset_hash"] == a["dataset_hash"]


# ---------------------------------------------------------------- fouling regimes (dataset_v2)
def _legacy_split(split):
    return split.model_copy(update={"foul_tc_range": (0.0, 0.8 - CFG.fouling.clip_margin), "foul_cool_range": (0.0, 0.8 - CFG.fouling.clip_margin)})


def test_fouling_rescale_changes_only_fouling_dependent_signals():
    split = CFG.splits["train"]
    a = simulate_episode("train", split, 3, CFG, REG, plant_for(split, REG, {}), SP)
    s2 = _legacy_split(split)
    b = simulate_episode("train", s2, 3, CFG, REG, plant_for(s2, REG, {}), SP)
    for key in ("fuel", "t_amb0", "p_amb0", "foul_tc_rate_per_h", "foul_cool_rate_per_h", "bias_t_exh_k", "bias_m_fuel_rel"):
        assert a.meta[key] == b.meta[key], key
    for key in ("load_fraction", "speed_factor", "t_amb_true", "p_amb_true", "n_rpm_true", "m_fuel_true"):
        assert np.array_equal(a.truth[key], b.truth[key]), key
    assert a.meta["foul_tc0"] <= 0.45 and b.meta["foul_tc0"] >= a.meta["foul_tc0"]
    assert not np.array_equal(a.truth["t_exh_true"], b.truth["t_exh_true"])


def test_capped_splits_never_exceed_the_cap():
    cap = CFG.fouling.default_start_range[1] + CFG.fouling.clip_margin
    for i in range(12):
        ep = episode("train", i)
        assert max(ep.truth["foul_tc"].max(), ep.truth["foul_cool"].max()) <= cap + 1e-12


def test_high_fouling_cohorts():
    for i in range(6):
        ep = episode("test_regime_high_fouling", i)
        hi, lo = ("foul_tc", "foul_cool") if ep.meta["foul_cohort"] == "tc_high" else ("foul_cool", "foul_tc")
        assert ep.truth[hi].min() >= 0.55 - 1e-12
        assert ep.truth[lo].max() <= 0.50 + 1e-12
    assert {episode("test_regime_high_fouling", i).meta["foul_cohort"] for i in range(2)} == {"tc_high", "cooler_high"}
