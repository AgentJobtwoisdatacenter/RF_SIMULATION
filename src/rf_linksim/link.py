"""Die Kette: Sender + Szenario + Empfangsantenne(n) -> P_rx und C/N0.

Dies ist der Zusammenfuehrungs-Modul. Er kennt constants/geometry/antenna/
pathloss/environment und setzt sie zu einem vollstaendigen Leistungsbudget
zusammen -- **und hoert dort auf**. Kein B, keine Rauschzahl, kein ADC, kein
SDR. `LinkResult.cn0_db_hz` ist die einzige Groesse, die Schritt 2 braucht;
`link.py` weiss nichts von Empfaengern (siehe INSTRUCITONS.md, "Der Schnitt
zu Schritt 2").

**Zwei Empfangsmodi**, ueber `rx_mode`:

- "single": eine Empfangsantenne (Default in der Verifikations-Testsuite,
  reproduziert INSTRUCITONS.md exakt -- linear polarisiert, 2 dBi).
- "dual_diversity": zwei phasenkohaerente Empfangskanaele (Default fuer den
  *operativen* Betrieb, siehe config/default.yaml -- ein bladeRF 2.0 micro
  xA9 mit RHCP+LHCP-Antennenpaar). Kombiniert wird per idealem
  Maximum-Ratio-Combining: C/N0_kombiniert (LINEAR) = C/N0_a + C/N0_b. Das
  ist ein exaktes Standardresultat der Diversity-Combining-Theorie, keine
  Naeherung -- setzt aber voraus, dass beide Kanaele auf denselben
  Rauschpegel (kT0-Referenz) bezogen sind und die Kombination phasenrichtig
  und praktisch verlustfrei erfolgt (reale Kombinier-/Kalibrierverluste sind
  Schritt-2-Hardwaredetails).

**power_is_eirp**: Wenn true, ist `tx.power_dbm` bereits die EIRP zum
Antennenmaximum -- der TX-Antennengewinn darf dann NICHT noch einmal addiert
werden (INSTRUCITONS.md, strukturelle Tests). Was aber weiterhin gilt: die
Richtcharakteristik F(theta) reduziert die Leistung, wenn der Empfaenger
nicht exakt in Gewinnrichtung liegt -- EIRP ist ja nur fuer eine Richtung
definiert. Deshalb wird bei power_is_eirp=True nur der Musterverlust
10*log10(F(theta)) angewendet, nicht die volle Direktivitaet.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from rf_linksim import antenna, constants, environment, geometry, pathloss


# --- Dataclasses -------------------------------------------------------------


@dataclass
class Transmitter:
    """Sendeseite: Leistung, Frequenz, Antennenmuster, Polarisation, Einbaulage."""

    power_dbm: float
    power_is_eirp: bool = False
    frequency_hz: float = 5.8e9
    height_m: float = 100.0
    tilt_rad: float = 0.0
    antenna_pattern: Callable = antenna.pagoda_pattern
    antenna_pattern_kwargs: dict = field(default_factory=dict)
    polarization_r: float = 1.0  # +1 = RHCP (Referenz-Drehsinn)


@dataclass
class ReceiveAntenna:
    """Empfangsseite: eine einzelne Antenne (ein Kanal von ggf. mehreren)."""

    gain_dbi: float
    height_m: float = 2.0
    feed_loss_db: float = 0.0
    polarization_r: float = float("inf")  # inf = linear
    polarization_delta_tau_rad: float = 0.0


@dataclass
class Scenario:
    """Alles, was nicht zu Sender oder Empfangsantenne gehoert: Strecke und Umwelt."""

    d_ground_m: float
    environment_stage: int
    clutter_model: str = "al_hourani"  # oder "fixed"
    clutter_fixed_db: Optional[float] = None  # nur fuer clutter_model="fixed"
    apply_frequency_correction: bool = False  # unbelegter Zusatzterm, siehe environment.py
    vegetation_depth_m: float = 0.0
    two_ray_enabled: bool = False
    two_ray_kwargs: dict = field(default_factory=dict)
    atmospheric_specific_attenuation_db_per_km: float = 0.0
    rain_rate_mm_per_hr: float = 0.0
    rain_k: Optional[float] = None
    rain_alpha: Optional[float] = None


@dataclass
class LinkResult:
    """P_rx, C/N0 und JEDER Zwischenwert der Kette -- nichts wird verschluckt."""

    p_rx_dbm: float
    cn0_db_hz: float

    d_slant_m: float
    elevation_deg: float
    theta_tx_deg: float

    tx_power_term_dbm: float  # power_dbm (+ Gewinn oder + Musterverlust, je nach power_is_eirp)
    tx_directivity_dbi: float
    tx_pattern_loss_db: float  # 10*log10(F(theta)), <=0
    rx_gain_dbi: float
    fspl_db: float
    clutter_loss_db: float
    vegetation_loss_db: float
    two_ray_extra_db: float
    atmospheric_loss_db: float
    rain_loss_db: float
    polarization_loss_db: float
    feed_loss_db: float


# --- Kern-Berechnung (ein einzelner Empfangskanal) --------------------------


def compute_link_budget(scenario: Scenario, tx: Transmitter, rx: ReceiveAntenna) -> LinkResult:
    """Vollstaendiges Leistungsbudget fuer EINEN Empfangskanal, an der Antennenklemme."""
    wavelength_m = constants.wavelength(tx.frequency_hz)

    d_slant = geometry.slant_range(scenario.d_ground_m, tx.height_m, rx.height_m)
    elevation_rad = geometry.elevation_angle(scenario.d_ground_m, tx.height_m, rx.height_m)
    elevation_deg = np.degrees(elevation_rad)
    theta = antenna.angle_off_boresight(elevation_rad, tx.tilt_rad)

    tx_directivity_dbi = antenna.directivity_dbi(tx.antenna_pattern, **tx.antenna_pattern_kwargs)
    f_theta = np.asarray(tx.antenna_pattern(theta, **tx.antenna_pattern_kwargs), dtype=float)
    with np.errstate(divide="ignore"):
        tx_pattern_loss_db = 10.0 * np.log10(f_theta)

    if tx.power_is_eirp:
        # EIRP ist P_tx*G_peak in EINER Referenzrichtung -- der Peak-Gewinn
        # steckt schon in power_dbm, nur die Musterabweichung von diesem
        # Peak (<=0 dB) kommt noch dazu. Direktivitaet NICHT addieren, sonst
        # doppelt gezaehlt (INSTRUCITONS.md, struktureller Test).
        tx_power_term_dbm = tx.power_dbm + tx_pattern_loss_db
    else:
        tx_power_term_dbm = tx.power_dbm + tx_directivity_dbi + tx_pattern_loss_db

    fspl_db = pathloss.free_space_path_loss(d_slant, wavelength_m)

    stage = environment.STAGES[scenario.environment_stage]
    if scenario.clutter_model == "al_hourani":
        clutter_loss_db = environment.clutter_loss_al_hourani_db(elevation_deg, stage)
    elif scenario.clutter_model == "fixed":
        if scenario.clutter_fixed_db is None:
            raise ValueError("clutter_model='fixed' braucht scenario.clutter_fixed_db")
        clutter_loss_db = environment.clutter_loss_fixed_db(scenario.clutter_fixed_db)
    else:
        raise ValueError(f"unbekanntes clutter_model: {scenario.clutter_model!r}")

    if scenario.apply_frequency_correction:
        clutter_loss_db = clutter_loss_db + environment.frequency_correction_db(tx.frequency_hz)

    if scenario.vegetation_depth_m > 0.0:
        vegetation_loss_db = environment.vegetation_loss_weissberger_db(
            scenario.vegetation_depth_m, tx.frequency_hz
        )
    else:
        vegetation_loss_db = 0.0

    if scenario.two_ray_enabled:
        two_ray_extra_db = pathloss.two_ray_extra_loss_db(
            scenario.d_ground_m, tx.height_m, rx.height_m, wavelength_m, **scenario.two_ray_kwargs
        )
    else:
        two_ray_extra_db = 0.0

    if scenario.atmospheric_specific_attenuation_db_per_km > 0.0:
        atmospheric_loss_db = pathloss.atmospheric_attenuation_db(
            d_slant, scenario.atmospheric_specific_attenuation_db_per_km
        )
    else:
        atmospheric_loss_db = 0.0

    if scenario.rain_rate_mm_per_hr > 0.0:
        if scenario.rain_k is None or scenario.rain_alpha is None:
            raise ValueError("rain_rate_mm_per_hr > 0 braucht scenario.rain_k und scenario.rain_alpha")
        rain_loss_db = pathloss.rain_attenuation_db(
            d_slant, scenario.rain_rate_mm_per_hr, scenario.rain_k, scenario.rain_alpha
        )
    else:
        rain_loss_db = 0.0

    polarization_loss_db = antenna.polarization_mismatch_loss_db(
        tx.polarization_r, rx.polarization_r, rx.polarization_delta_tau_rad
    )

    p_rx_dbm = (
        tx_power_term_dbm
        + rx.gain_dbi
        - fspl_db
        - clutter_loss_db
        - vegetation_loss_db
        + two_ray_extra_db  # zwei_ray_extra_db traegt sein Vorzeichen schon: negativ=Verlust
        - atmospheric_loss_db
        - rain_loss_db
        - polarization_loss_db
        - rx.feed_loss_db
    )
    cn0_db_hz = p_rx_dbm - constants.kt0_dbm_per_hz()

    return LinkResult(
        p_rx_dbm=p_rx_dbm,
        cn0_db_hz=cn0_db_hz,
        d_slant_m=d_slant,
        elevation_deg=elevation_deg,
        theta_tx_deg=np.degrees(theta),
        tx_power_term_dbm=tx_power_term_dbm,
        tx_directivity_dbi=tx_directivity_dbi,
        tx_pattern_loss_db=tx_pattern_loss_db,
        rx_gain_dbi=rx.gain_dbi,
        fspl_db=fspl_db,
        clutter_loss_db=clutter_loss_db,
        vegetation_loss_db=vegetation_loss_db,
        two_ray_extra_db=two_ray_extra_db,
        atmospheric_loss_db=atmospheric_loss_db,
        rain_loss_db=rain_loss_db,
        polarization_loss_db=polarization_loss_db,
        feed_loss_db=rx.feed_loss_db,
    )


# --- Dual-Diversity-Combining (2 phasenkohaerente Kanaele, bladeRF-Default) -


@dataclass
class DualDiversityResult:
    """Ergebnis fuer rx_mode='dual_diversity': beide Einzelkanaele plus Kombination."""

    branch_a: LinkResult
    branch_b: LinkResult
    p_rx_combined_dbm: float
    cn0_combined_db_hz: float


def compute_link_budget_dual_diversity(
    scenario: Scenario, tx: Transmitter, rx_a: ReceiveAntenna, rx_b: ReceiveAntenna
) -> DualDiversityResult:
    """Zwei phasenkohaerente Empfangskanaele (z. B. RHCP+LHCP), ideal per MRC kombiniert.

    C/N0_kombiniert (linear) = C/N0_a (linear) + C/N0_b (linear) -- exaktes
    MRC-Ergebnis fuer zwei Kanaele mit unabhaengigem, gleich starkem Rauschen
    (siehe Modul-Docstring). P_rx_kombiniert folgt aus derselben Rechnung in
    Watt (beide Kanaele beziehen sich auf dieselbe kT0-Referenz, deshalb ist
    die Watt-Summe der beiden Einzel-P_rx physikalisch die kombinierte
    Empfangsleistung, keine Naeherung).
    """
    branch_a = compute_link_budget(scenario, tx, rx_a)
    branch_b = compute_link_budget(scenario, tx, rx_b)

    cn0_a_lin = 10.0 ** (branch_a.cn0_db_hz / 10.0)
    cn0_b_lin = 10.0 ** (branch_b.cn0_db_hz / 10.0)
    cn0_combined_db_hz = 10.0 * np.log10(cn0_a_lin + cn0_b_lin)

    p_rx_a_w = constants.dbm_to_watt(branch_a.p_rx_dbm)
    p_rx_b_w = constants.dbm_to_watt(branch_b.p_rx_dbm)
    p_rx_combined_dbm = constants.watt_to_dbm(p_rx_a_w + p_rx_b_w)

    return DualDiversityResult(
        branch_a=branch_a,
        branch_b=branch_b,
        p_rx_combined_dbm=p_rx_combined_dbm,
        cn0_combined_db_hz=cn0_combined_db_hz,
    )
