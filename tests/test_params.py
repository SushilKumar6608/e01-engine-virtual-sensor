"""Gate 1 - provenance: no anonymous numbers (SIMULATION PROJECT)."""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from vsensor.params import DEFAULT_PATH, Param, RegistryFile, load_registry

ROOT = Path(__file__).resolve().parents[1]


class _StrictLoader(yaml.SafeLoader):
    """YAML loader that refuses duplicate keys (PyYAML silently keeps the last one)."""


def _no_duplicates(loader, node, deep=False):
    keys = [loader.construct_object(k, deep=deep) for k, _ in node.value]
    dup = {k for k in keys if keys.count(k) > 1}
    if dup:
        raise ValueError(f"duplicate YAML keys: {sorted(dup)}")
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicates)


def test_registry_file_has_no_duplicate_keys():
    yaml.load(DEFAULT_PATH.read_text(), Loader=_StrictLoader)


def test_registry_loads_and_every_param_has_evidence():
    reg = load_registry()
    assert len(reg.ids) > 30
    for pid in reg.ids:
        p = reg.param(pid)
        if p.status in ("sourced", "derived"):
            assert p.source in reg.sources, pid
        if p.status == "assumed":
            assert p.rationale and len(p.rationale) >= 15, pid
        if p.status == "calibrated":
            assert p.calibration, pid


def test_every_source_says_how_it_was_verified():
    reg = load_registry()
    for sid, s in reg.sources.items():
        assert len(s.verification) > 20, sid


def test_sources_not_cited_by_a_parameter_are_the_known_reference_only_ones():
    reg = load_registry()
    cited = {reg.param(p).source for p in reg.ids if reg.param(p).source}
    # S05 = Cantera thermo table (feeds src/vsensor/sim/thermo_nasa7.json, not a registry value)
    # S06 = marine gas oil heating value, used only to judge the n-dodecane surrogate in the Cantera check
    assert set(reg.sources) - cited == {"S05", "S06"}


@pytest.mark.parametrize(
    "bad",
    [
        dict(value=1.0, unit="m", description="x" * 5, status="sourced"),  # no source
        dict(value=1.0, unit="m", description="x" * 5, status="assumed"),  # no rationale
        dict(value=1.0, unit="m", description="x" * 5, status="assumed", rationale="short"),
        dict(value=1.0, unit="m", description="x" * 5, status="calibrated"),  # no calibration note
        dict(value=5.0, unit="m", description="x" * 5, status="assumed", rationale="a long enough reason here", plausible_range=(0, 1)),
        dict(value=1.0, unit="m", description="x" * 5, status="invented", rationale="a long enough reason here"),
    ],
)
def test_param_without_evidence_is_rejected(bad):
    with pytest.raises(ValidationError):
        Param(**bad)


def test_unknown_source_id_is_rejected():
    data = {
        "sources": {"S1": {"citation": "c", "verification": "v" * 25}},
        "params": {"a.b": {"value": 1.0, "unit": "m", "description": "desc", "status": "sourced", "source": "S9"}},
    }
    with pytest.raises(ValidationError):
        RegistryFile.model_validate(data)


def test_malformed_id_is_rejected():
    data = {"sources": {}, "params": {"Bad Id": {"value": 1.0, "unit": "m", "description": "desc", "status": "assumed", "rationale": "long enough reason"}}}
    with pytest.raises(ValidationError):
        RegistryFile.model_validate(data)


def test_unregistered_lookup_fails_loudly():
    with pytest.raises(KeyError, match="not in the registry"):
        load_registry()["engine.not_a_parameter"]


def test_hash_is_stable_and_changes_with_values():
    a, b = load_registry(), load_registry()
    assert a.content_hash() == b.content_hash()
    assert a.with_overrides(engine__stroke=2.3).content_hash() != a.content_hash()


# --- code <-> registry consistency -------------------------------------------------------
_LITERAL_ID = re.compile(r"""\[\s*["']([a-z][a-z0-9_]*(?:\.[a-z0-9_]+)+)["']\s*\]""")


def _python_files():
    for sub in ("src", "scripts", "tests"):
        yield from (ROOT / sub).rglob("*.py")


def test_every_parameter_id_used_in_code_is_registered():
    reg = set(load_registry().ids)
    used = set()
    for f in _python_files():
        if f.name == "test_params.py":
            continue
        used |= set(_LITERAL_ID.findall(f.read_text()))
    missing = sorted(u for u in used if u not in reg)
    assert not missing, f"ids used in code but missing from registry: {missing}"


def test_every_registered_parameter_is_used_somewhere():
    reg = load_registry()
    text = "\n".join(f.read_text() for f in _python_files() if f.name != "test_params.py")
    # ids can appear as "a.b" or with '__' (override keyword form)
    unused = []
    for pid in reg.ids:
        if pid in text or pid.replace(".", "__") in text:
            continue
        # fuel latent heats are looked up with an f-string in sim/fuels.py
        if pid.startswith("fuel.") and pid.endswith(".latent_heat") and "latent_heat" in text:
            continue
        unused.append(pid)
    assert not unused, f"registered but never used: {unused}"
