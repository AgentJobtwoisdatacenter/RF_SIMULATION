#!/usr/bin/env python3
"""Distanz-Sweeps ueber alle 4 Umgebungsstufen, fuer beide Konfigurationen.

Schreibt PNGs nach output/ (gitignored):
    output/sweep_default.png       -- Verifikations-Baseline (single)
    output/sweep_operational.png   -- reales Setup (dual_diversity, kombiniert)

    python scripts/run_sweeps.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rf_linksim import config, plotting, sweep

DISTANCES_M = np.array([200.0, 500.0, 1000.0, 2000.0, 3000.0, 5000.0, 6000.0])
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    default_cfg = config.load_config("config/default.yaml")
    results_default = sweep.distance_sweep_all_stages(
        DISTANCES_M, default_cfg.scenario, default_cfg.transmitter, default_cfg.rx_single
    )
    fig = plotting.plot_distance_sweep_by_stage(
        DISTANCES_M, results_default, quantity="cn0_db_hz",
        title="Verifikations-Baseline (config/default.yaml, single-Modus)",
    )
    out = OUTPUT_DIR / "sweep_default.png"
    plotting.save_figure(fig, out)
    print(f"geschrieben: {out}")

    operational_cfg = config.load_config("config/operational.yaml")
    results_operational = sweep.distance_sweep_dual_diversity_all_stages(
        DISTANCES_M, operational_cfg.scenario, operational_cfg.transmitter,
        operational_cfg.rx_dual_a, operational_cfg.rx_dual_b,
    )
    # plot_distance_sweep_by_stage braucht LinkResult-artige Objekte mit dem
    # per quantity benannten Feld -- DualDiversityResult hat cn0_combined_db_hz
    # statt cn0_db_hz, das quantity-Argument traegt dem einfach Rechnung.
    fig = plotting.plot_distance_sweep_by_stage(
        DISTANCES_M, results_operational, quantity="cn0_combined_db_hz",
        title="Reales Setup (config/operational.yaml, dual_diversity, kombiniert)",
    )
    out = OUTPUT_DIR / "sweep_operational.png"
    plotting.save_figure(fig, out)
    print(f"geschrieben: {out}")


if __name__ == "__main__":
    main()
