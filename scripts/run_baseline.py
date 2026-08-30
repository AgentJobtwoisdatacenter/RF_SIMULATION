#!/usr/bin/env python3
"""Rechnet die Basisszenarien aus config/default.yaml und config/operational.yaml
und druckt P_rx, C/N0 und alle Zwischenwerte.

    python scripts/run_baseline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rf_linksim import config


def _print_single(label, result):
    print(f"\n{label}")
    print("-" * len(label))
    print(f"  P_rx        = {result.p_rx_dbm:9.3f} dBm")
    print(f"  C/N0        = {result.cn0_db_hz:9.3f} dB-Hz")
    print(f"  d_slant     = {result.d_slant_m:9.1f} m")
    print(f"  Elevation   = {result.elevation_deg:9.3f} deg")
    print(f"  theta_TX    = {result.theta_tx_deg:9.3f} deg")
    print(f"  TX-Term     = {result.tx_power_term_dbm:9.3f} dBm  (Direktivitaet {result.tx_directivity_dbi:.2f} dBi, Musterverlust {result.tx_pattern_loss_db:.3f} dB)")
    print(f"  RX-Gewinn   = {result.rx_gain_dbi:9.3f} dBi")
    print(f"  FSPL        = {result.fspl_db:9.3f} dB")
    print(f"  Clutter     = {result.clutter_loss_db:9.3f} dB")
    print(f"  Vegetation  = {result.vegetation_loss_db:9.3f} dB")
    print(f"  Two-Ray     = {result.two_ray_extra_db:9.3f} dB")
    print(f"  Atmosphaere = {result.atmospheric_loss_db:9.3f} dB")
    print(f"  Regen       = {result.rain_loss_db:9.3f} dB")
    print(f"  PLF         = {result.polarization_loss_db:9.3f} dB")
    print(f"  Speiseverl. = {result.feed_loss_db:9.3f} dB")


def main():
    default_cfg = config.load_config("config/default.yaml")
    default_result = config.compute(default_cfg)
    _print_single("Verifikations-Baseline (config/default.yaml)", default_result)

    operational_cfg = config.load_config("config/operational.yaml")
    operational_result = config.compute(operational_cfg)
    print(f"\nReales Zielszenario (config/operational.yaml), rx_mode={operational_cfg.rx_mode}")
    print("-" * 60)
    _print_single("  Kanal A", operational_result.branch_a)
    _print_single("  Kanal B", operational_result.branch_b)
    print(f"\n  Kombiniert: P_rx = {operational_result.p_rx_combined_dbm:.3f} dBm, "
          f"C/N0 = {operational_result.cn0_combined_db_hz:.3f} dB-Hz")


if __name__ == "__main__":
    main()
