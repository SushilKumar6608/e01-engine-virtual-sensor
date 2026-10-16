"""Generate docs/PARAMETERS.md from config/params.yaml (SIMULATION PROJECT).

Every number the simulators use, with unit, status and the evidence behind it. Run: make params-doc
"""
from __future__ import annotations

from pathlib import Path

from vsensor.params import load_registry

OUT = Path(__file__).resolve().parents[1] / "docs" / "PARAMETERS.md"
ORDER = ["sourced", "derived", "calibrated", "assumed"]


def fmt(v: float) -> str:
    return f"{v:.6g}"


def main() -> None:
    reg = load_registry()
    L = []
    L.append("# Parameter registry")
    L.append("")
    L.append("**All values describe a simulated, generic engine. Nothing is validated against a real engine.** "
             "Generated from `config/params.yaml` by `scripts/make_params_doc.py`; do not edit by hand.")
    L.append("")
    c = reg.status_counts()
    L.append(f"{len(reg.ids)} parameters: " + ", ".join(f"{c.get(k, 0)} {k}" for k in ORDER) + f". Registry hash `{reg.content_hash()[:16]}`.")
    L.append("")
    L.append("| Status | Meaning |")
    L.append("|---|---|")
    L.append("| sourced | taken from a cited source |")
    L.append("| derived | computed from sourced numbers |")
    L.append("| calibrated | tuned to hit a stated target; a test re-runs the calibration and fails if the stored value is stale |")
    L.append("| assumed | my own engineering assumption, with a written rationale; **not verified** |")
    L.append("")
    L.append("## Sources and how each was checked")
    L.append("")
    for sid, s in reg.sources.items():
        L.append(f"**{sid}** - {' '.join(s.citation.split())}" + (f" <{s.url}>" if s.url else ""))
        L.append(f"  - *Verification:* {' '.join(s.verification.split())}")
        L.append("")
    for status in ORDER:
        ids = [p for p in reg.ids if reg.param(p).status == status]
        if not ids:
            continue
        L.append(f"## {status.capitalize()} parameters")
        L.append("")
        L.append("| Id | Value | Unit | What it is | Evidence |")
        L.append("|---|---|---|---|---|")
        for pid in ids:
            p = reg.param(pid)
            ev = {
                "sourced": f"source {p.source}",
                "derived": f"derived from {p.source}",
                "calibrated": " ".join((p.calibration or "").split()),
                "assumed": " ".join((p.rationale or "").split()),
            }[status]
            L.append(f"| `{pid}` | {fmt(p.value)} | {p.unit} | {p.description} | {ev} |")
        L.append("")
    OUT.write_text("\n".join(L) + "\n")
    print("wrote", OUT, f"({len(reg.ids)} parameters)")


if __name__ == "__main__":
    main()
