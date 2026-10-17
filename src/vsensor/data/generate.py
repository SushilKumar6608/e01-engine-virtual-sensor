"""Build the simulated dataset: episodes -> tables -> hashes -> manifest (SIMULATED DATA).

Plain-language summary
----------------------
Reads config/dataset.yaml and the parameter registry, simulates every episode of every split,
writes the tables, and records a content hash. Same config, same parameters and same seed give
the same hash, which is quality gate 8.

Files written to the output directory
  signals_<split>.parquet   what a data logger records: the model inputs and the measured
                            exhaust temperature (the physical sensor, which is the target)
  truth_<split>.parquet     true values and hidden states; evaluation only, never a model input
  episodes.parquet          one row per episode: fuel, ambient, health, sensor biases, fault
  events.parquet            injected faults and sensor dropouts with exact times
  manifest.json             hashes, provenance, split sizes

Hashing: measured values are already rounded to the sensor resolution, so they are hashed as
integers. That keeps the hash stable between computers even if the last bits of a floating
point result differ. True values are hashed after rounding to 0.001 and reported separately.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from vsensor.data.config import DatasetConfig, Split
from vsensor.data.episodes import Episode, simulate_episode
from vsensor.params import Registry
from vsensor.sim.plant import Plant
from vsensor.sim.sensors import SensorSpecs

ROOT = Path(__file__).resolve().parents[3]
NO_FIT_PREFIXES = ("nominal.", "m0b.")   # parameters of models that consume the data
MODEL_INPUT_COLUMNS = ["fuel_code", "n_rpm", "m_fuel", "p_scav", "t_scav", "n_tc", "t_amb", "p_amb"]
TARGET_COLUMN = "t_exh_meas"
SIGNAL_COLUMNS = ["t_s", *MODEL_INPUT_COLUMNS, TARGET_COLUMN]
INT_SENTINEL = np.iinfo(np.int64).min


# ------------------------------------------------------------------------- hashing helpers
def _to_int(a: np.ndarray, scale: float) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    out = np.full(a.shape, INT_SENTINEL, dtype=np.int64)
    ok = np.isfinite(a)
    out[ok] = np.round(a[ok] * scale).astype(np.int64)
    return out


def signals_digest(signals: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for col in SIGNAL_COLUMNS:
        h.update(col.encode())
        h.update(_to_int(signals[col], 1e6).tobytes())
    return h.hexdigest()


def truth_digest(truth: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for col in sorted(truth):
        a = truth[col]
        h.update(col.encode())
        h.update(_to_int(a.astype(float) if a.dtype != bool else a.astype(np.int8), 1e3).tobytes())
    return h.hexdigest()


def _canon(obj) -> str:
    def conv(o):
        if isinstance(o, (np.floating, float)):
            return float(f"{float(o):.9g}")
        if isinstance(o, (np.integer,)):
            return int(o)
        raise TypeError(type(o))
    return json.dumps(obj, sort_keys=True, default=conv)


def episode_digest(ep: Episode) -> str:
    h = hashlib.sha256()
    h.update(signals_digest(ep.signals).encode())
    h.update(_canon(ep.meta).encode())
    h.update(_canon(ep.events).encode())
    return h.hexdigest()


def generator_code_hash() -> str:
    """Hash of the code that generates the data (line endings normalised, so Windows and Linux agree)."""
    files = sorted(
        [*(ROOT / "src" / "vsensor" / "sim").glob("*.py"), *(ROOT / "src" / "vsensor" / "data").glob("*.py")]
    )
    files = [f for f in files if f.name not in ("quality.py", "card.py")]
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode())
        h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    h.update((ROOT / "src" / "vsensor" / "sim" / "thermo_nasa7.json").read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def registry_hash(reg: Registry) -> str:
    return reg.content_hash(exclude_prefixes=NO_FIT_PREFIXES)


def dataset_digest(cfg: DatasetConfig, reg_hash: str, episode_hashes: dict[str, list[str]]) -> str:
    h = hashlib.sha256()
    h.update(f"schema{cfg.schema_version}".encode())
    h.update(cfg.content_hash().encode())
    h.update(reg_hash.encode())
    for split in cfg.splits:
        for d in episode_hashes[split]:
            h.update(d.encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------- build a split
def plant_for(split: Split, reg: Registry, cache: dict) -> Plant:
    key = split.eta_scale_methanol
    if key not in cache:
        cache[key] = Plant(reg if key == 1.0 else reg.with_overrides(engine__eta_scale_methanol=key))
    return cache[key]


def build_split(name: str, cfg: DatasetConfig, reg: Registry, sp: SensorSpecs, cache: dict, indices=None):
    split = cfg.splits[name]
    plant = plant_for(split, reg, cache)
    eps = [simulate_episode(name, split, i, cfg, reg, plant, sp) for i in (range(split.episodes) if indices is None else indices)]
    return eps


def _stack(eps: list[Episode], which: str) -> pd.DataFrame:
    cols = list(getattr(eps[0], which))
    data = {c: np.concatenate([getattr(e, which)[c] for e in eps]) for c in cols}
    df = pd.DataFrame(data)
    df.insert(0, "episode_id", pd.Categorical(np.repeat([e.episode_id for e in eps], [len(e.signals["t_s"]) for e in eps])))
    return df


def build_dataset(cfg: DatasetConfig, reg: Registry, out_dir: Path, splits: list[str] | None = None, progress: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    sp = SensorSpecs.from_registry(reg)
    cache: dict = {}
    names = list(cfg.splits) if splits is None else splits
    ep_hashes: dict[str, list[str]] = {}
    truth_hashes: list[str] = []
    meta_rows, event_rows = [], []
    sizes: dict[str, dict] = {}
    t0 = time.time()
    for name in names:
        eps = build_split(name, cfg, reg, sp, cache)
        sig, tru = _stack(eps, "signals"), _stack(eps, "truth")
        sig.to_parquet(out_dir / f"signals_{name}.parquet", index=False)
        tru.to_parquet(out_dir / f"truth_{name}.parquet", index=False)
        ep_hashes[name] = [episode_digest(e) for e in eps]
        truth_hashes += [truth_digest(e.truth) for e in eps]
        for e in eps:
            meta_rows.append({**e.meta, "episode_digest": episode_digest(e)})
            event_rows += [{"episode_id": e.episode_id, **ev} for ev in e.events]
        sizes[name] = {"episodes": len(eps), "rows": int(len(sig))}
        if progress:
            print(f"  {name:36s} {len(eps):4d} episodes {len(sig):8d} rows   [{time.time() - t0:6.1f} s]", flush=True)
    pd.DataFrame(meta_rows).to_parquet(out_dir / "episodes.parquet", index=False)
    ev = pd.DataFrame(event_rows, columns=["episode_id", "kind", "start_s", "end_s", "magnitude"])
    ev.to_parquet(out_dir / "events.parquet", index=False)

    rh = registry_hash(reg)
    th = hashlib.sha256("".join(truth_hashes).encode()).hexdigest()
    manifest = {
        "name": cfg.name,
        "schema_version": cfg.schema_version,
        "all_data_is_simulated": True,
        "dataset_hash": dataset_digest(cfg, rh, ep_hashes) if splits is None else None,
        "truth_hash": th,
        "config_hash": cfg.content_hash(),
        "registry_hash": rh,
        "registry_hash_note": f"computed without parameters starting with {NO_FIT_PREFIXES}",
        "generator_code_hash": generator_code_hash(),
        "base_seed": cfg.base_seed,
        "splits": sizes,
        "episode_digests": ep_hashes,
        "model_input_columns": MODEL_INPUT_COLUMNS,
        "target_column": TARGET_COLUMN,
        "environment": {
            "python": sys.version.split()[0], "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__,
        },
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or None if r.returncode == 0 else None
    except Exception:
        return None


def load_manifest(data_dir: Path) -> dict:
    return json.loads((Path(data_dir) / "manifest.json").read_text())


def load_split(data_dir: Path, name: str, kind: str = "signals") -> pd.DataFrame:
    return pd.read_parquet(Path(data_dir) / f"{kind}_{name}.parquet")
