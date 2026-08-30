"""YAML -> Objekte: config/*.yaml in Transmitter/Scenario/ReceiveAntenna(-Paare).

Zwei mitgelieferte Konfigurationen mit unterschiedlichem Zweck:

- `config/default.yaml` -- reproduziert das Basisszenario aus
  INSTRUCITONS.md 1:1 (Tilt 0, rx_mode "single", 1 lineare 2-dBi-Antenne).
  Das ist die Konfiguration, gegen die die Verifikationstabelle geprueft
  wird -- NICHT aendern, ohne die Tests in tests/test_physics.py
  mitzuziehen.
- `config/operational.yaml` -- das reale Zielszenario (Tilt 20 Grad,
  rx_mode "dual_diversity", zwei identische Breitband-Rundstrahler). Alle
  Annahmewerte sind im YAML mit `# ANNAHME` markiert.

Dieses Modul kennt nur die Uebersetzung YAML -> Dataclass, keine eigene
Physik -- die liegt vollstaendig in link.py.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import yaml

from rf_linksim import antenna, link

PATTERN_REGISTRY = {
    "pagoda": antenna.pagoda_pattern,
    "dipole": antenna.dipole_pattern,
    "isotropic": antenna.isotropic_pattern,
}


@dataclass
class AppConfig:
    """Alles, was aus einer config/*.yaml geladen wurde."""

    transmitter: link.Transmitter
    scenario: link.Scenario
    rx_mode: str  # "single" oder "dual_diversity"
    rx_single: Optional[link.ReceiveAntenna] = None
    rx_dual_a: Optional[link.ReceiveAntenna] = None
    rx_dual_b: Optional[link.ReceiveAntenna] = None


def _build_transmitter(raw: dict) -> link.Transmitter:
    raw = dict(raw)  # Kopie, damit wir Schluessel entfernen koennen
    pattern_name = raw.pop("antenna_pattern", "pagoda")
    tilt_deg = raw.pop("tilt_deg", 0.0)

    return link.Transmitter(
        power_dbm=float(raw.pop("power_dbm")),
        power_is_eirp=raw.pop("power_is_eirp", False),
        frequency_hz=float(raw.pop("frequency_hz", 5.8e9)),
        height_m=float(raw.pop("height_m", 100.0)),
        tilt_rad=np.radians(float(tilt_deg)),
        antenna_pattern=PATTERN_REGISTRY[pattern_name],
        antenna_pattern_kwargs=raw.pop("antenna_pattern_kwargs", {}) or {},
        polarization_r=raw.pop("polarization_r", 1.0),
    )


def _build_receive_antenna(raw: dict) -> link.ReceiveAntenna:
    return link.ReceiveAntenna(
        gain_dbi=float(raw["gain_dbi"]),
        height_m=float(raw.get("height_m", 2.0)),
        feed_loss_db=float(raw.get("feed_loss_db", 0.0)),
        polarization_r=float(raw.get("polarization_r", float("inf"))),
        polarization_delta_tau_rad=float(raw.get("polarization_delta_tau_rad", 0.0)),
    )


def _build_scenario(raw: dict) -> link.Scenario:
    clutter_fixed_db = raw.get("clutter_fixed_db")
    atmospheric = raw.get("atmospheric_specific_attenuation_db_per_km", 0.0)
    rain_rate = raw.get("rain_rate_mm_per_hr", 0.0)
    rain_k = raw.get("rain_k")
    rain_alpha = raw.get("rain_alpha")
    return link.Scenario(
        d_ground_m=float(raw["d_ground_m"]),
        environment_stage=int(raw["environment_stage"]),
        clutter_model=raw.get("clutter_model", "al_hourani"),
        clutter_fixed_db=None if clutter_fixed_db is None else float(clutter_fixed_db),
        apply_frequency_correction=raw.get("apply_frequency_correction", False),
        vegetation_depth_m=float(raw.get("vegetation_depth_m", 0.0)),
        two_ray_enabled=raw.get("two_ray_enabled", False),
        two_ray_kwargs=raw.get("two_ray_kwargs", {}) or {},
        atmospheric_specific_attenuation_db_per_km=float(atmospheric),
        rain_rate_mm_per_hr=float(rain_rate),
        rain_k=None if rain_k is None else float(rain_k),
        rain_alpha=None if rain_alpha is None else float(rain_alpha),
    )


def load_config(path) -> AppConfig:
    """Laedt eine config/*.yaml-Datei zu einem AppConfig."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    transmitter = _build_transmitter(raw["transmitter"])
    scenario = _build_scenario(raw["scenario"])
    rx_mode = raw["rx_mode"]

    rx_single = None
    rx_dual_a = None
    rx_dual_b = None
    if rx_mode == "single":
        rx_single = _build_receive_antenna(raw["rx_single"])
    elif rx_mode == "dual_diversity":
        rx_dual_a = _build_receive_antenna(raw["rx_dual"]["branch_a"])
        rx_dual_b = _build_receive_antenna(raw["rx_dual"]["branch_b"])
    else:
        raise ValueError(f"unbekannter rx_mode: {rx_mode!r}")

    return AppConfig(
        transmitter=transmitter,
        scenario=scenario,
        rx_mode=rx_mode,
        rx_single=rx_single,
        rx_dual_a=rx_dual_a,
        rx_dual_b=rx_dual_b,
    )


def compute(config: AppConfig):
    """Fuehrt config direkt zu link.compute_link_budget(...) bzw.
    ..._dual_diversity(...) aus, je nach rx_mode."""
    if config.rx_mode == "single":
        return link.compute_link_budget(config.scenario, config.transmitter, config.rx_single)
    return link.compute_link_budget_dual_diversity(
        config.scenario, config.transmitter, config.rx_dual_a, config.rx_dual_b
    )
