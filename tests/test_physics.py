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

from rf_linksim import antenna, constants, geometry


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


# --- antenna.py: Winkel zur Antennenachse -----------------------------------


def test_angle_off_boresight_horizon_case():
    """epsilon = 0, tilt = 0 (Sender am Horizont, Antenne senkrecht):
    theta = pi/2 -- der Empfaenger liegt am Horizont der Antenne, dort hat
    ein Rundstrahler sein Maximum."""
    theta = antenna.angle_off_boresight(elevation_rad=0.0, tilt_rad=0.0)
    assert theta == pytest.approx(np.pi / 2.0)


def test_angle_off_boresight_nadir_case():
    """epsilon = 90 Grad, tilt = 0 (Empfaenger direkt unter der Drohne):
    theta = 0 -- der Empfaenger liegt auf der Antennenachse, dort hat ein
    Rundstrahler seine Null."""
    theta = antenna.angle_off_boresight(elevation_rad=np.pi / 2.0, tilt_rad=0.0)
    assert theta == pytest.approx(0.0, abs=1e-12)


def test_angle_off_boresight_baseline_scenario():
    """Basisszenario (3 km/100 m/2 m, Antenne senkrecht): epsilon = 1,871 Grad,
    also fast am Horizont -> theta nahe 90 Grad -> fast voller Antennengewinn.
    Genau der Fallstrick-6-Effekt aus INSTRUCITONS.md."""
    eps = geometry.elevation_angle(d_ground=3000.0, h_tx=100.0, h_rx=2.0)
    theta_deg = np.degrees(antenna.angle_off_boresight(eps, tilt_rad=0.0))
    assert theta_deg == pytest.approx(88.129, abs=0.01)


# --- antenna.py: Richtcharakteristiken und Direktivitaet -------------------


def test_directivity_isotropic():
    assert antenna.directivity_dbi(antenna.isotropic_pattern) == pytest.approx(
        0.0, abs=1e-6
    )


def test_directivity_dipole():
    assert antenna.directivity_dbi(antenna.dipole_pattern) == pytest.approx(
        2.15, abs=0.01
    )


def test_directivity_pagoda():
    assert antenna.directivity_dbi(antenna.pagoda_pattern) == pytest.approx(
        1.75, abs=0.02
    )


def test_dipole_true_null_on_axis():
    assert antenna.dipole_pattern(0.0) == pytest.approx(0.0, abs=1e-12)
    assert antenna.dipole_pattern(np.pi) == pytest.approx(0.0, abs=1e-12)


def test_dipole_maximum_broadside():
    assert antenna.dipole_pattern(np.pi / 2.0) == pytest.approx(1.0)


def test_pagoda_fill_level_on_axis():
    """Auf der Achse (theta=0) faellt sin^n auf 0, der Floor haelt das Muster
    aber auf dem konfigurierten Fill-Level (-12 dB), nicht bei -inf."""
    f_axis = antenna.pagoda_pattern(0.0, floor_db=-12.0)
    assert 10.0 * np.log10(f_axis) == pytest.approx(-12.0, abs=1e-6)


def test_pagoda_pattern_peaks_at_90_degrees():
    """'Rundstrahler-Maximum bei theta = 90 Grad', numerisch ueber ein
    feines Winkelraster bestaetigt, nicht nur aus der Formel abgelesen."""
    theta_grid = np.linspace(0.0, np.pi, 100_001)
    f = antenna.pagoda_pattern(theta_grid)
    theta_at_max = theta_grid[np.argmax(f)]
    assert theta_at_max == pytest.approx(np.pi / 2.0, abs=1e-3)


def test_pattern_functions_vectorize():
    # 51 Punkte (ungerade), damit die Rastermitte exakt auf theta = pi/2
    # faellt -- sonst wird das Maximum der Muster knapp verfehlt, weil sie
    # dort ihr Maximum haben, nicht weil die Funktion falsch waere.
    theta_grid = np.linspace(0.0, np.pi, 51)
    for pattern in (antenna.isotropic_pattern, antenna.dipole_pattern, antenna.pagoda_pattern):
        f = pattern(theta_grid)
        assert f.shape == theta_grid.shape
        assert np.all(f >= 0.0)
        assert np.max(f) == pytest.approx(1.0)


# --- antenna.py: Polarisations-Mismatch (PLF) -------------------------------


def test_plf_cp_to_cp_same_sense():
    loss_db = antenna.polarization_mismatch_loss_db(r1=1.0, r2=1.0, delta_tau_rad=0.0)
    assert loss_db == pytest.approx(0.0, abs=1e-9)


def test_plf_cp_to_linear():
    # r1 = 1 (perfekt zirkular), r2 -> unendlich (perfekt linear): 3,01 dB,
    # unabhaengig von delta_tau, weil ein zirkularer Sender keine
    # bevorzugte Achse hat.
    for delta_tau in (0.0, 0.3, np.pi / 4.0, 1.7):
        loss_db = antenna.polarization_mismatch_loss_db(
            r1=1.0, r2=np.inf, delta_tau_rad=delta_tau
        )
        assert loss_db == pytest.approx(3.0103, abs=0.001)


def test_plf_rhcp_to_lhcp_total_loss():
    # Gegenlaeufiger Drehsinn ueber das Vorzeichen von r2.
    loss_db = antenna.polarization_mismatch_loss_db(r1=1.0, r2=-1.0, delta_tau_rad=0.0)
    assert np.isinf(loss_db) and loss_db > 0


def test_plf_linear_to_linear_crossed_total_loss():
    loss_db = antenna.polarization_mismatch_loss_db(
        r1=np.inf, r2=np.inf, delta_tau_rad=np.pi / 2.0
    )
    assert np.isinf(loss_db) and loss_db > 0


def test_plf_linear_to_linear_aligned_no_loss():
    loss_db = antenna.polarization_mismatch_loss_db(
        r1=np.inf, r2=np.inf, delta_tau_rad=0.0
    )
    assert loss_db == pytest.approx(0.0, abs=1e-6)


def test_plf_symmetric_in_branches():
    """PLF darf nicht davon abhaengen, welche Antenne als 'Sender' (r1) und
    welche als 'Empfaenger' (r2) benannt wird -- physikalisch ist das
    Mismatch symmetrisch."""
    a = antenna.polarization_loss_factor(r1=1.0, r2=np.inf, delta_tau_rad=0.4)
    b = antenna.polarization_loss_factor(r1=np.inf, r2=1.0, delta_tau_rad=0.4)
    assert a == pytest.approx(b)


def test_plf_never_exceeds_unity():
    """PLF ist ein Leistungsanteil, darf nie > 1 werden -- ueber ein Raster
    aus Achsenverhaeltnissen und Winkeln geprueft."""
    r_values = [1.0, 1.5, 2.0, 5.0, np.inf]
    for r1 in r_values:
        for r2 in r_values:
            for delta_tau in np.linspace(0.0, np.pi, 5):
                plf = antenna.polarization_loss_factor(r1, r2, delta_tau)
                assert plf <= 1.0 + 1e-9
                assert plf >= -1e-9
