"""Run all three calibrations in dependency order (SIMULATION PROJECT).

  1. efficiency curve  - exact fit to three published fuel-consumption points
  2. rated point       - five air-path constants hit five rated-point targets
  3. nominal model     - exhaust share from the rated heat balance, delivery ratio by least squares

Usage:  python scripts/calibrate_all.py [--write]     (--write edits config/params.yaml in place)
"""
from __future__ import annotations

import argparse

import numpy as np

from vsensor.calibration import (
    ALL_IDS, YAML_PATH, brake_efficiency_on_propeller_law, calibrate_all, healthy_grid,
    rated_state, sfoc_target_efficiencies, write_yaml_values,
)
from vsensor.params import load_registry
from vsensor.sim.nominal import NominalModel
from vsensor.sim.plant import Plant


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="edit config/params.yaml in place")
    args = ap.parse_args()
    reg0 = load_registry()
    reg = calibrate_all(reg0)
    print("calibrated values (new / old):")
    for pid in ALL_IDS:
        print(f"  {pid:22s} {reg[pid]:<12.6g} {reg0[pid]:.6g}")
    plant, s = rated_state(reg)
    print("\\nefficiency fit: model", np.round(brake_efficiency_on_propeller_law(plant), 5),
          "target", np.round(sfoc_target_efficiencies(reg), 5))
    print(f"rated point (clean, ISO, diesel-like): T_exh {float(s.t_exh):.2f} K | p_scav {float(s.p_scav)/1e5:.4f} bar | "
          f"n_tc {float(s.n_tc):.1f} rpm | cooler dp {float(s.dp_cooler):.1f} Pa | T_scav-T_amb {float(s.t_scav - s.t_amb):.2f} K | "
          f"m_air {float(s.m_air):.2f} kg/s")
    g = healthy_grid(Plant(reg))
    err = NominalModel(reg).predict(g.n_rpm, g.m_fuel, g.p_scav, g.t_scav) - g.t_exh
    print(f"nominal model on healthy grid ({err.size} pts): RMSE {np.sqrt(np.mean(err**2)):.2f} K, max |err| {np.max(np.abs(err)):.2f} K")
    if args.write:
        write_yaml_values({pid: reg[pid] for pid in ALL_IDS})
        print("wrote", YAML_PATH)


if __name__ == "__main__":
    main()
