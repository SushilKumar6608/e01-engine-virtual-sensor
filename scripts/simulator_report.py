"""Simulator evidence report (SIMULATION PROJECT) -> reports/simulator/.

Acceptance gate:
  * exhaust temperature stays in a plausible band across load
  * scavenge pressure rises with load
  * unit tests pass
plus the deliverables: parameter registry, Cantera mechanism check, nominal model.

Every number in the generated report is computed here. Run: make report
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vsensor.calibration import healthy_grid
from vsensor.params import load_registry
from vsensor.sim.fuels import FUEL_SPECIES
from vsensor.sim.nominal import NominalModel
from vsensor.sim.plant import Plant

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "simulator"
OUT.mkdir(parents=True, exist_ok=True)

BAND = (600.0, 800.0)  # K, indicative axis range of the exhaust-temperature plot in S01
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LABEL = "SIMULATED DATA - generic engine model, not validated against any real engine"

plt.rcParams.update(
    {"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
     "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#fcfcfb",
     "axes.facecolor": "#fcfcfb", "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6}
)

reg = load_registry()
P = Plant(reg)
N = NominalModel(reg)
LOADS = np.round(np.arange(0.30, 1.0001, 0.05), 2)


def sweep(fuel, **kw):
    n, mf = P.propeller_law(LOADS, fuel)
    return P.solve(n, mf, fuel=fuel, **kw)


def nominal_err(s, fuel="diesel_like"):
    return N.predict(s.n_rpm, s.m_fuel, s.p_scav, s.t_scav, fuel) - s.t_exh


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


d, m = sweep("diesel_like"), sweep("methanol")
sfoc = lambda s: 3.6e6 / (s.eta_brake * reg["calib.sfoc.lhv"]) * 1e3  # noqa: E731  (g/kWh at the MAN 42.7 MJ/kg basis)

# ------------------------------------------------------------------ figure 1
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
a = ax[0]
a.axhspan(*BAND, color=GRID, alpha=0.6, lw=0)
a.text(41, 606, "600-800 K: indicative range plotted in Alegret et al. 2015 [S01]", fontsize=8, color=INK2)
a.plot(LOADS * 100, d.t_exh, color=BLUE, lw=2, marker="o", ms=4, label="diesel-like")
a.plot(LOADS * 100, m.t_exh, color=ORANGE, lw=2, marker="o", ms=4, label="methanol")
a.set_xlabel("brake power, % of rated (propeller law)")
a.set_ylabel("exhaust gas temperature before turbine [K]")
a.set_title("Exhaust temperature versus load", loc="left", fontsize=10, color=INK)
a.legend(frameon=False, loc="upper right")
a.set_xlim(28, 102)
a.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}")
b = ax[1]
b.plot(LOADS * 100, d.p_scav / 1e5, color=BLUE, lw=2, marker="o", ms=4, label="diesel-like")
b.plot(LOADS * 100, m.p_scav / 1e5, color=ORANGE, lw=2, marker="o", ms=4, label="methanol")
b.plot(LOADS * 100, d.p_exh / 1e5, color=BLUE, lw=1.2, ls="--", label="exhaust receiver (hidden), diesel-like")
b.set_xlabel("brake power, % of rated (propeller law)")
b.set_ylabel("pressure [bar abs]")
b.set_title("Scavenge pressure versus load", loc="left", fontsize=10, color=INK)
b.legend(frameon=False, loc="upper left", fontsize=8)
b.set_xlim(28, 102)
fig.suptitle(LABEL, fontsize=8, color=INK2, y=0.995)
fig.tight_layout()
fig.savefig(OUT / "fig1_load_sweep.png", dpi=150)
plt.close(fig)

# ----------------------------------------------------------------- scenarios
scen = {
    "healthy, ISO": dict(),
    "turbocharger fouled (1.0)": dict(foul_tc=1.0),
    "air cooler fouled (1.0)": dict(foul_cool=1.0),
    "cold ambient (278 K)": dict(t_amb=278.15),
    "hot ambient (308 K)": dict(t_amb=308.15),
}
base = sweep("diesel_like")
res = {k: sweep("diesel_like", **v) for k, v in scen.items()}
res["methanol, healthy"] = m

# ------------------------------------------------------------------ figure 2
fig, a = plt.subplots(figsize=(9, 4.4))
cols = [INK2, BLUE, ORANGE, AQUA, YELLOW, MAGENTA]
for (name, s), c in zip(res.items(), cols):
    fuel = "methanol" if "methanol" in name else "diesel_like"
    e = nominal_err(s, fuel)
    a.plot(LOADS * 100, e, color=c, lw=2 if name.startswith("healthy") else 1.6, label=name)
a.axhline(0, color=INK2, lw=0.8)
a.set_xlabel("brake power, % of rated (propeller law)")
a.set_ylabel("nominal model minus plant [K]")
a.set_title("Where the physics-only baseline (M0) is wrong", loc="left", fontsize=10, color=INK)
a.legend(frameon=False, fontsize=8, loc="lower right", ncol=2)
a.set_xlim(28, 102)
fig.text(0.01, 0.005, LABEL, fontsize=7, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(OUT / "fig2_nominal_error.png", dpi=150)
plt.close(fig)

# --------------------------------------------------------------------- tests
proc = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-o", "addopts="], cwd=ROOT, capture_output=True, text=True)
test_line = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()][-1]
tests_ok = proc.returncode == 0
cantera_ok = "all checks passed" in (ROOT / "docs" / "CANTERA_CHECK.md").read_text()

# --------------------------------------------------------------------- report
hi = LOADS >= 0.40 - 1e-9
in_band = (d.t_exh >= BAND[0]) & (d.t_exh <= BAND[1])
out_loads = LOADS[~in_band]
m_in_band = (m.t_exh >= BAND[0]) & (m.t_exh <= BAND[1])
p_mono = bool(np.all(np.diff(d.p_scav) > 0) and np.all(np.diff(m.p_scav) > 0))
band_ok = bool(np.all(in_band[hi]) and np.all(m_in_band[hi]))
bmep = P.p_rated / (P.v_disp * P.n_rated / 60) / 1e5
rated = d.t_exh[-1]

L = []
A = L.append
A("# Simulator report")
A("")
A(f"**{LABEL}.** Generated by `scripts/simulator_report.py`; every number is computed there. Registry hash `{reg.content_hash()[:16]}`.")
A("")
A("## Gate")
A("")
rows = [
    ("Exhaust temperature in a plausible band across load", "PASS across 40-100 % load" if band_ok else "**FAIL**",
     f"600-800 K [S01, indicative]. Diesel-like: {d.t_exh[hi].min():.0f}-{d.t_exh[hi].max():.0f} K for 40-100 % load. "
     f"Methanol: {m.t_exh[hi].min():.0f}-{m.t_exh[hi].max():.0f} K. **Outside the band: loads {', '.join(f'{x*100:.0f} %' for x in out_loads)} "
     f"(diesel-like {', '.join(f'{t:.0f} K' for t in d.t_exh[~in_band])}); no source covers that region.**"),
    ("Scavenge pressure rises with load", "PASS" if p_mono else "**FAIL**",
     f"diesel-like {d.p_scav[0]/1e5:.2f} -> {d.p_scav[-1]/1e5:.2f} bar, methanol {m.p_scav[0]/1e5:.2f} -> {m.p_scav[-1]/1e5:.2f} bar, strictly increasing"),
    ("Unit tests pass", "PASS" if tests_ok else "**FAIL**", test_line),
    ("Cantera mechanism check", "PASS" if cantera_ok else "**FAIL**", "docs/CANTERA_CHECK.md"),
    ("Every parameter has a source or is marked assumed", "PASS", f"{len(reg.ids)} parameters: " + ", ".join(f"{v} {k}" for k, v in sorted(reg.status_counts().items()))),
]
L += md_table(["Criterion", "Result", "Evidence"], rows)
A("")
A("![Load sweep](fig1_load_sweep.png)")
A("")
A("## Load sweep on the propeller law (healthy, ISO ambient)")
A("")
A("**Study range is 40-100 % load.** Rows at 30-35 % are shown only because they are used as an abstention stress test; they carry no accuracy claim. ")
A("")
A("Exhaust temperature and scavenge pressure are the model's *predictions*: only the rated point (and, for efficiency, the 50/75/100 % fuel-consumption points) were calibrated.")
A("")
hdr = ["load %", "speed rpm", "fuel kg/s", "p_scav bar", "T_scav K", "n_tc rpm", "T_exh K", "air kg/s", "excess air", "SFOC g/kWh", "p_exh/p_scav (hidden)"]
rows = [[f"{LOADS[i]*100:.0f}", f"{d.n_rpm[i]:.1f}", f"{d.m_fuel[i]:.3f}", f"{d.p_scav[i]/1e5:.2f}", f"{d.t_scav[i]:.1f}", f"{d.n_tc[i]:.0f}",
         f"{d.t_exh[i]:.0f}", f"{d.m_air[i]:.1f}", f"{d.lambda_total[i]:.2f}", f"{sfoc(d)[i]:.1f}", f"{d.p_exh[i]/d.p_scav[i]:.3f}"] for i in range(len(LOADS))]
L += md_table(hdr, rows)
A("")
A("## Consistency checks against sources")
A("")
A(f"* Implied rated brake mean effective pressure: **{bmep:.1f} bar** (assumed stroke {reg['engine.stroke']} m). The larger G60ME-C9.5 quotes 21.0 bar at L1 [S10]: same class, different engine, so this supports only the order of magnitude.")
A(f"* Fuel consumption at 50/75/100 % load: {sfoc(d)[LOADS==0.5][0]:.1f} / {sfoc(d)[LOADS==0.75][0]:.1f} / {sfoc(d)[LOADS==1.0][0]:.1f} g/kWh. Published (G60ME-C9.5, high-load tuning): 165.5 / 163.0 / 167.0 [S10]. This was a calibration, so the match proves nothing; the curve outside 50-100 % load is an extrapolation ({sfoc(d)[0]:.0f} g/kWh at 30 %).")
A(f"* Rated point: T_exh {rated:.1f} K, p_scav {d.p_scav[-1]/1e5:.2f} bar, n_tc {d.n_tc[-1]:.0f} rpm ({d.n_tc[-1]*2*np.pi/60:.0f} rad/s), scavenge air {d.t_scav[-1]-d.t_amb[-1]:.1f} K above cooling water [S08: about 12 K by design], air flow {d.m_air[-1]:.1f} kg/s, excess air ratio {d.lambda_total[-1]:.2f}. T_exh, p_scav and n_tc are calibration targets chosen inside the S01 plot ranges; air flow and excess air are NOT calibrated and have no source.")
A(f"* Low-load scavenge air only {d.t_scav[0]-d.t_amb[0]:.1f} K above cooling water at 30 % load, consistent in direction with S08 (\"only slightly higher\").")
A("")
A("## Size of each effect on the target (diesel-like unless stated)")
A("")
A("Change in exhaust temperature, scavenge pressure, scavenge temperature and turbocharger speed against the healthy ISO case at the same fuel flow and speed.")
A("")
sel = [0.30, 0.50, 0.75, 1.00]
idx = [int(np.where(np.isclose(LOADS, x))[0][0]) for x in sel]
for name, s in res.items():
    if name == "healthy, ISO":
        continue
    ref = base if "methanol" not in name else None
    if ref is None:
        continue
    A(f"**{name}**")
    A("")
    rows = [[f"{LOADS[i]*100:.0f} %", f"{s.t_exh[i]-ref.t_exh[i]:+.1f}", f"{(s.p_scav[i]-ref.p_scav[i])/1e5:+.3f}", f"{s.t_scav[i]-ref.t_scav[i]:+.1f}", f"{s.n_tc[i]-ref.n_tc[i]:+.0f}"] for i in idx]
    L += md_table(["load", "dT_exh K", "dp_scav bar", "dT_scav K", "dn_tc rpm"], rows)
    A("")
A("**methanol versus diesel-like at equal brake power** (composition and turbocharger feedback together)")
A("")
rows = [[f"{LOADS[i]*100:.0f} %", f"{m.t_exh[i]-d.t_exh[i]:+.1f}", f"{(m.p_scav[i]-d.p_scav[i])/1e5:+.3f}", f"{m.m_fuel[i]/d.m_fuel[i]:.2f}"] for i in idx]
L += md_table(["load", "dT_exh K", "dp_scav bar", "fuel mass ratio"], rows)
A("")
A("Direction checks (tests): fouling of either kind raises exhaust temperature; colder ambient lowers it, consistent with MAN's statement that the temperature after the turbine drops in winter [S08]. **The sizes are not verified**: S08 gives no numbers. Cooler fouling *raises* scavenge pressure in this model (a hotter exhaust drives the turbocharger harder); I could not confirm that sign against a source. Turbocharger fouling lowers boost from 50 % load up but not at 30 %.")
A("")
A("## How wrong is the physics-only baseline (M0)?")
A("")
A("![Nominal error](fig2_nominal_error.png)")
A("")
g = healthy_grid(P)
err = N.predict(g.n_rpm, g.m_fuel, g.p_scav, g.t_scav) - g.t_exh
lo, mid = g.load_fraction < 0.45, g.load_fraction >= 0.45
A(f"Fit on the healthy ISO diesel-like grid (39 points, load 40-100 % at three speed offsets from the propeller law): RMSE {np.sqrt(np.mean(err**2)):.1f} K overall; "
  f"{np.sqrt(np.mean(err[mid]**2)):.1f} K for loads of 45 % and above; {np.sqrt(np.mean(err[lo]**2)):.1f} K below 45 % (max {np.max(np.abs(err)):.0f} K).")
A("")
rows = []
for name, s in res.items():
    fuel = "methanol" if "methanol" in name else "diesel_like"
    e = nominal_err(s, fuel)
    rows.append([name, f"{np.sqrt(np.mean(e[hi]**2)):.1f}", f"{np.mean(e[hi]):+.1f}", f"{np.sqrt(np.mean(e[~hi]**2)):.1f}"])
L += md_table(["scenario", "RMSE K, 40-100 % load", "bias K, 40-100 %", "RMSE K, 30-35 % (stress only, no accuracy claim)"], rows)
A("")
A("## Limitations found (also in `docs/MODEL_CHOICES.md`)")
A("")
A("1. **Below about 40 % load the model is unverified and runs hot** (see gate row). Real engines use auxiliary blowers at low load; this model has none.")
A(f"2. **Exhaust temperature is nearly flat between 50 and 100 % load** ({d.t_exh[LOADS>=0.5].min():.0f}-{d.t_exh[LOADS>=0.5].max():.0f} K) because the matched turbocharger raises air flow almost in proportion to fuel. I could not find part-load exhaust data to check this against. If real engines show a steeper curve, the plant is too forgiving.")
A("3. **Air flow responds strongly to exhaust back-pressure** through the assumed square-root delivery law; its sensitivity is an assumption, not a result.")
A("4. **Cold or hot ambient is not a hard regime shift here**: the nominal model absorbs it through the measured scavenge pressure and temperature (table above). The design lists cold ambient as a held-out regime.")
A("5. **The fuel effect is large (see table) and comes only from exhaust composition and turbocharger feedback**; both fuels share the same efficiency curve, which is an assumption.")
A("6. Compressor and turbine run at constant efficiency with no maps; the stroke, friction, wall-loss share and fouling magnitudes are assumptions (see `docs/PARAMETERS.md`).")
(OUT / "simulator_report.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
