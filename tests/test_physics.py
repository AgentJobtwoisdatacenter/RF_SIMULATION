"""Verifikation gegen unabhaengig nachgerechnete Sollwerte (siehe INSTRUCITONS.md).

Jeder Test prueft eine Groesse, deren Sollwert nicht aus diesem Code stammt,
sondern separat (von Hand oder mit Standardformeln) nachgerechnet wurde. Die
Toleranzen sind bewusst eng (rtol 1e-3 .. 1e-4), damit ein Vorzeichen- oder
Rundungsfehler nicht durchrutscht.

Baustein-weise nach INSTRUCITONS.md "Vorgehen": aktuell abgedeckt sind
constants.py und geometry.py.
"""

import numpy as np
import pytest

from rf_linksim import constants, geometry


# --- constants.py ------------------------------------------------------------


def test_kt0_dbm_per_hz():
    assert constants.kt0_dbm_per_hz() == pytest.approx(-173.98, abs=0.01)


def test_wavelength_5_8_ghz():
    lam = constants.wavelength(5.8e9)
    assert lam == pytest.approx(0.05169, abs=1e-5)  # 51,69 mm


def test_db_lin_roundtrip():
    x_db = 17.3
    assert constants.lin_to_db(constants.db_to_lin(x_db)) == pytest.approx(x_db)


def test_dbm_watt_roundtrip():
    # 33,0 dBm ist die im Dokument *gerundete* Angabe fuer 2 W; exakt sind es
    # 10*log10(2000) = 33,0103 dBm. Roundtrip muss exakt sein, und die
    # Rundung darf nicht mehr als 0,02 dB Fehler einbringen.
    p_w = 2.0
    p_dbm = constants.watt_to_dbm(p_w)
    assert p_dbm == pytest.approx(33.0, abs=0.02)
    assert constants.dbm_to_watt(p_dbm) == pytest.approx(p_w, rel=1e-9)


# --- geometry.py ---------------------------------------------------------


def test_slant_range_baseline():
    d = geometry.slant_range(d_ground=3000.0, h_tx=100.0, h_rx=2.0)
    assert d == pytest.approx(3001.6, abs=0.1)


def test_elevation_angle_baseline():
    eps_deg = geometry.elevation_angle_deg(d_ground=3000.0, h_tx=100.0, h_rx=2.0)
    assert eps_deg == pytest.approx(1.871, abs=0.001)


def test_fspl_uses_slant_not_ground_range():
    """100 m Flughoehe / 200 m Grundentfernung (h_rx = 0, wie im Dokument als
    Faustzahl-Beispiel genannt): Schraegentfernung ist rund 12 % laenger als
    die Grundentfernung, das kostet knapp 1 dB zusaetzliche Freiraum-
    daempfung -- der Effekt, der bei kurzen/steilen Strecken nicht
    vernachlaessigt werden darf."""
    d_ground = 200.0
    d_slant = geometry.slant_range(d_ground=d_ground, h_tx=100.0, h_rx=0.0)
    assert d_slant / d_ground == pytest.approx(1.118, abs=0.01)
    extra_loss_db = 20.0 * np.log10(d_slant / d_ground)
    assert extra_loss_db == pytest.approx(0.92, abs=0.05)


def test_radio_horizon_k_4_3():
    # d[km] = 4,12 * sqrt(h[m])
    for h in (10.0, 100.0, 1000.0):
        d_km = geometry.radio_horizon_km(h)
        assert d_km == pytest.approx(4.12 * np.sqrt(h), rel=1e-3)


def test_fresnel_zone_midpoint_3km():
    lam = constants.wavelength(5.8e9)
    r1 = geometry.fresnel_zone_radius_midpoint(d_total=3000.0, wavelength_m=lam)
    assert r1 == pytest.approx(6.23, abs=0.02)


def test_fresnel_zone_scales_with_sqrt_n():
    lam = constants.wavelength(5.8e9)
    r1 = geometry.fresnel_zone_radius_midpoint(3000.0, lam, n=1)
    r2 = geometry.fresnel_zone_radius_midpoint(3000.0, lam, n=2)
    assert r2 == pytest.approx(r1 * np.sqrt(2.0), rel=1e-6)


def test_geometry_vectorizes_over_numpy_arrays():
    """Sweeps duerfen nicht mit Python-Schleifen rechnen -- geometry.py muss
    mit numpy-Arrays genauso funktionieren wie mit Skalaren."""
    d_ground = np.array([200.0, 500.0, 3000.0])
    d = geometry.slant_range(d_ground, h_tx=100.0, h_rx=2.0)
    assert d.shape == d_ground.shape
    assert np.all(d >= d_ground)
