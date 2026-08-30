#!/usr/bin/env python3
"""Leitet aus dem Sweep-Raum die drei requirements.py-Kenngroessen ab:
maximal zulaessige Rauschzahl, Dynamikbereich, spektrale Leistungsdichte.

Reine Ableitung aus C/N0 -- keine Empfehlung, keine Pass/Fail-Bewertung
(siehe INSTRUCITONS.md). Nutzt config/operational.yaml als Sweep-Raum.

    python scripts/run_sdr_requirements.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rf_linksim import config, requirements, sweep

DISTANCES_M = np.array([200.0, 500.0, 1000.0, 2000.0, 3000.0, 5000.0, 6000.0])
BANDWIDTHS_HZ = [20e6, 27e6, 56e6]
SNR_TARGETS_DB = [3.0, 6.0, 10.0]
SIGNAL_BANDWIDTH_HZ = 27e6  # RSGB-Messung, Basisszenario -- Sender-Bandbreite, keine Empfaengerbandbreite


def main():
    cfg = config.load_config("config/operational.yaml")
    results = sweep.distance_sweep_dual_diversity_all_stages(
        DISTANCES_M, cfg.scenario, cfg.transmitter, cfg.rx_dual_a, cfg.rx_dual_b
    )

    all_cn0 = np.concatenate([r.cn0_combined_db_hz for r in results.values()])
    all_p_rx = np.concatenate([r.p_rx_combined_dbm for r in results.values()])

    print("1) Maximal zulaessige Rauschzahl NF_max [dB], ueber Bandbreiten x Ziel-SNR")
    print("   (staerkstes Szenario im Sweep-Raum: C/N0_max = %.1f dB-Hz)" % all_cn0.max())
    print("   (schwaechstes Szenario im Sweep-Raum: C/N0_min = %.1f dB-Hz)" % all_cn0.min())
    header = "   B \\ SNR_ziel" + "".join(f"{s:>10.0f} dB" for s in SNR_TARGETS_DB)
    for label, cn0 in (("staerkstes", all_cn0.max()), ("schwaechstes", all_cn0.min())):
        print(f"\n   -- {label} Szenario (C/N0 = {cn0:.1f} dB-Hz) --")
        print(header)
        table = requirements.max_noise_figure_table(cn0, BANDWIDTHS_HZ, SNR_TARGETS_DB)
        for bw, row in zip(BANDWIDTHS_HZ, table):
            print(f"   {bw/1e6:9.0f} MHz" + "".join(f"{v:13.2f}" for v in row))

    print("\n2) Erforderlicher Dynamikbereich")
    dr_db = requirements.required_dynamic_range_db(all_p_rx)
    dr_bits = requirements.required_dynamic_range_bits(all_p_rx)
    print(f"   P_rx: {all_p_rx.min():.1f} .. {all_p_rx.max():.1f} dBm")
    print(f"   Spanne: {dr_db:.1f} dB  =>  {dr_bits:.2f} ADC-Bit (dB / 6,02)")

    print("\n3) Spektrale Leistungsdichte (Sendesignal-Bandbreite, NICHT Empfaenger-Bandbreite)")
    for label, p_rx in (("staerkstes", all_p_rx.max()), ("schwaechstes", all_p_rx.min())):
        psd = requirements.spectral_power_density_dbm_per_hz(p_rx, SIGNAL_BANDWIDTH_HZ)
        print(f"   {label} Szenario: P_rx = {p_rx:7.1f} dBm  =>  PSD = {psd:7.1f} dBm/Hz "
              f"(bei {SIGNAL_BANDWIDTH_HZ/1e6:.0f} MHz Signalbandbreite)")


if __name__ == "__main__":
    main()
