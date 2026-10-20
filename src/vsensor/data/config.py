"""Dataset configuration schema (SIMULATED DATA).

The YAML file config/dataset.yaml holds every sampling decision (how many episodes, which load
bands, which seeds). This module checks it, so a typo fails loudly before any data is generated.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "dataset.yaml"
FAULT_TYPES = ("none", "bias", "drift", "dropout")
FUELS = ("diesel_like", "methanol")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Profile(_Strict):
    hold_s: tuple[float, float]
    ramp_s: tuple[float, float]
    speed_factor_range: tuple[float, float]
    smoothing_sigma_s: float
    wiggle_sd: float
    wiggle_tau_s: float


class FaultInjection(_Strict):
    start_s: tuple[float, float]
    bias_k: tuple[float, float]
    drift_k_per_min: tuple[float, float]
    dropout_s: tuple[float, float]


class Coverage(_Strict):
    bins: int
    required_prior_mass: float
    min_rows: int
    min_episodes: int
    small_split_episodes: int
    small_bins: int
    small_min_rows: int
    small_min_episodes: int


class Fouling(_Strict):
    """Range of the starting fouling of the turbocharger and the air cooler (0 clean, 1 fully fouled).

    The two Beta draws per episode are unchanged; their result (0 to 1) is rescaled linearly into the
    range of the split, so the number of random draws per episode never changes.
    """
    default_start_range: tuple[float, float]
    clip_margin: float = Field(ge=0.0)   # fouling may drift up to this far above the top of its start range


class FoulCohort(_Strict):
    name: str
    foul_tc_range: tuple[float, float] | None = None
    foul_cool_range: tuple[float, float] | None = None


class Split(_Strict):
    seed_id: int
    episodes: int = Field(gt=0)
    purpose: str
    fuel_mix: dict[str, float]
    load_band: dict[str, tuple[float, float]]
    t_amb_k: tuple[float, float]
    eta_scale_methanol: float = 1.0
    fault_cycle: list[str] | None = None
    foul_tc_range: tuple[float, float] | None = None      # None: Fouling.default_start_range
    foul_cool_range: tuple[float, float] | None = None
    foul_cohorts: list[FoulCohort] | None = None           # episode i belongs to cohort i modulo the number of cohorts

    @model_validator(mode="after")
    def _check(self):
        if abs(sum(self.fuel_mix.values()) - 1.0) > 1e-9:
            raise ValueError("fuel_mix must sum to 1")
        for f in self.fuel_mix:
            if f not in FUELS:
                raise ValueError(f"unknown fuel {f}")
            if f not in self.load_band:
                raise ValueError(f"load_band missing for {f}")
        for lo, hi in self.load_band.values():
            if not (0.0 < lo < hi <= 1.0):
                raise ValueError("load_band must satisfy 0 < lo < hi <= 1")
        if self.fault_cycle is not None and not set(self.fault_cycle) <= set(FAULT_TYPES):
            raise ValueError("unknown fault type")
        return self


class DatasetConfig(_Strict):
    name: str
    schema_version: int
    base_seed: int
    sampling_hz: float
    episode_duration_s: int
    profile: Profile
    fault_injection: FaultInjection
    coverage: Coverage
    fouling: Fouling
    splits: dict[str, Split]

    @model_validator(mode="after")
    def _unique_seed_ids(self):
        ids = [s.seed_id for s in self.splits.values()]
        if len(set(ids)) != len(ids):
            raise ValueError("seed_id must be unique per split")
        for sname, s in self.splits.items():
            ranges = [r for r in (s.foul_tc_range, s.foul_cool_range, self.fouling.default_start_range) if r]
            ranges += [r for c in (s.foul_cohorts or []) for r in (c.foul_tc_range, c.foul_cool_range) if r]
            for lo, hi in ranges:
                if not (0.0 <= lo < hi and hi + self.fouling.clip_margin <= 1.0 + 1e-12):
                    raise ValueError(f"bad fouling range in {sname}: {(lo, hi)}")
        if self.sampling_hz != 1.0:
            raise ValueError("this version of the generator supports 1 Hz only")
        return self

    def content_hash(self) -> str:
        canon = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canon.encode()).hexdigest()


def foul_ranges(cfg: "DatasetConfig", split: Split, index: int):
    """Cohort name and the (low, high) start ranges for the turbocharger and the cooler of episode ``index``."""
    tc = split.foul_tc_range or cfg.fouling.default_start_range
    cl = split.foul_cool_range or cfg.fouling.default_start_range
    name = "all"
    if split.foul_cohorts:
        c = split.foul_cohorts[index % len(split.foul_cohorts)]
        name, tc, cl = c.name, c.foul_tc_range or tc, c.foul_cool_range or cl
    return name, tuple(tc), tuple(cl)


def load_dataset_config(path: Path | str = CONFIG_PATH) -> DatasetConfig:
    return DatasetConfig(**yaml.safe_load(Path(path).read_text()))
