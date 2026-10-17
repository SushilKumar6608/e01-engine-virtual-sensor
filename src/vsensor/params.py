"""Parameter registry with a provenance gate (SIMULATION PROJECT).

Plain-language summary
----------------------
Every number that the simulators use lives in config/params.yaml. Each entry
must say where it came from:

  sourced     - taken from a cited source (the source id must exist)
  derived     - computed from sourced numbers (the source id must exist)
  assumed     - my own engineering assumption; a written rationale is required
  calibrated  - tuned so that the model hits a stated target; the target and
                the test that enforces it must be named

If an entry lacks that evidence, loading the registry fails. This is quality
gate 1 ("provenance") from the design document: no anonymous numbers.

The registry has a content hash, so a dataset can record exactly which
parameter set produced it.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "params.yaml"

Status = Literal["sourced", "derived", "assumed", "calibrated"]
_ID_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z0-9_]+)+$")


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")

    citation: str
    url: str | None = None
    #: How the content was checked in this project - be explicit about limits.
    verification: str


class Param(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: float
    unit: str = Field(min_length=1)
    description: str = Field(min_length=3)
    status: Status
    source: str | None = None
    rationale: str | None = None
    calibration: str | None = None
    plausible_range: tuple[float, float] | None = None

    @model_validator(mode="after")
    def _evidence_required(self) -> "Param":
        if self.status in ("sourced", "derived") and not self.source:
            raise ValueError(f"status '{self.status}' needs a source id")
        if self.status == "assumed" and not (self.rationale and len(self.rationale) >= 15):
            raise ValueError("status 'assumed' needs a written rationale")
        if self.status == "calibrated" and not self.calibration:
            raise ValueError("status 'calibrated' needs a calibration note (target + test)")
        if self.plausible_range is not None:
            lo, hi = self.plausible_range
            if not lo <= self.value <= hi:
                raise ValueError(f"value {self.value} outside plausible_range {self.plausible_range}")
        return self


class RegistryFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sources: dict[str, Source]
    params: dict[str, Param]

    @field_validator("params")
    @classmethod
    def _ids_well_formed(cls, v: dict[str, Param]) -> dict[str, Param]:
        bad = [k for k in v if not _ID_RE.match(k)]
        if bad:
            raise ValueError(f"malformed parameter ids: {bad}")
        return v

    @model_validator(mode="after")
    def _sources_exist(self) -> "RegistryFile":
        missing = sorted(
            {p.source for p in self.params.values() if p.source and p.source not in self.sources}
        )
        if missing:
            raise ValueError(f"parameters cite unknown source ids: {missing}")
        return self


class Registry:
    """Read-only access to validated parameters."""

    def __init__(self, data: RegistryFile):
        self._data = data

    # -- access -----------------------------------------------------------
    def __getitem__(self, pid: str) -> float:
        try:
            return self._data.params[pid].value
        except KeyError:
            raise KeyError(f"parameter '{pid}' is not in the registry (gate 1: no anonymous numbers)") from None

    def param(self, pid: str) -> Param:
        return self._data.params[pid]

    @property
    def ids(self) -> list[str]:
        return sorted(self._data.params)

    @property
    def sources(self) -> dict[str, Source]:
        return self._data.sources

    # -- provenance -------------------------------------------------------
    def with_overrides(self, **overrides: float) -> "Registry":
        """New registry with some values replaced (ids use '__' for '.').

        Used for sensitivity checks and calibration only. The overridden
        registry has a different content hash, so it cannot masquerade as
        the original.
        """
        data = self._data.model_copy(deep=True)
        for key, val in overrides.items():
            pid = key.replace("__", ".")
            if pid not in data.params:
                raise KeyError(pid)
            data.params[pid] = data.params[pid].model_copy(update={"value": float(val)})
        return Registry(data)

    def content_hash(self, exclude_prefixes: tuple[str, ...] = ()) -> str:
        """SHA-256 over ids, values, units and status (not over prose).

        ``exclude_prefixes`` leaves out parameters of models that consume the data (for example
        ``nominal.``), so fitting them later does not change the identity of the simulated data.
        """
        canon = {
            pid: [p.value, p.unit, p.status, p.source]
            for pid, p in sorted(self._data.params.items())
            if not pid.startswith(exclude_prefixes)
        }
        return hashlib.sha256(json.dumps(canon, sort_keys=True).encode()).hexdigest()

    def status_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for p in self._data.params.values():
            out[p.status] = out.get(p.status, 0) + 1
        return out


def load_registry(path: Path | str = DEFAULT_PATH) -> Registry:
    raw = yaml.safe_load(Path(path).read_text())
    return Registry(RegistryFile.model_validate(raw))
