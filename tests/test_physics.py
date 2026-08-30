"""Verifikation gegen unabhaengig nachgerechnete Sollwerte (siehe INSTRUCITONS.md).

Jeder Test prueft eine Groesse, deren Sollwert nicht aus diesem Code stammt,
sondern separat (von Hand oder mit Standardformeln) nachgerechnet wurde. Die
Toleranzen sind bewusst eng (rtol 1e-3 .. 1e-4), damit ein Vorzeichen- oder
Rundungsfehler nicht durchrutscht.

Baustein-weise nach INSTRUCITONS.md "Vorgehen": aktuell abgedeckt sind
constants.py, geometry.py, antenna.py und pathloss.py.
"""

import numpy as np
import pytest

from rf_linksim import antenna, constants, environment, geometry, pathloss


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


# --- pathloss.py: Freiraumdaempfung -----------------------------------------


def test_fspl_5km_5_8ghz():
    lam = constants.wavelength(5.8e9)
    assert pathloss.free_space_path_loss(5000.0, lam) == pytest.approx(121.70, abs=0.01)


def test_fspl_6km_5_8ghz():
    lam = constants.wavelength(5.8e9)
    assert pathloss.free_space_path_loss(6000.0, lam) == pytest.approx(123.28, abs=0.01)


def test_fspl_doubling_distance_costs_6_02_db():
    lam = constants.wavelength(5.8e9)
    for d in (500.0, 1500.0, 3000.0):
        l1 = pathloss.free_space_path_loss(d, lam)
        l2 = pathloss.free_space_path_loss(2.0 * d, lam)
        assert l2 - l1 == pytest.approx(6.0206, abs=1e-4)


def test_fspl_vectorizes():
    lam = constants.wavelength(5.8e9)
    d = np.array([1000.0, 2000.0, 3000.0])
    l = pathloss.free_space_path_loss(d, lam)
    assert l.shape == d.shape
    assert np.all(np.diff(l) > 0)  # monoton steigend mit der Entfernung


# --- pathloss.py: Two-Ray-Bodenreflexion (Default: aus, hier nur Formel) ---


def test_two_ray_breakpoint_baseline():
    lam = constants.wavelength(5.8e9)
    d_bp = pathloss.two_ray_breakpoint_distance(h_tx=100.0, h_rx=2.0, wavelength_m=lam)
    assert d_bp == pytest.approx(15_500.0, rel=0.005)  # ~15,5 km


def test_fresnel_reflection_zero_when_epsilon_equals_one():
    """eps_r = 1 heisst 'Boden hat dieselbe Permittivitaet wie Luft' -- es
    gibt dann physikalisch keine Grenzflaeche und der Reflexionskoeffizient
    muss exakt 0 sein, unabhaengig vom Winkel."""
    eps_g = pathloss.complex_relative_permittivity(
        epsilon_r=1.0, conductivity_s_per_m=0.0, wavelength_m=0.05169
    )
    for psi_deg in (5.0, 20.0, 45.0, 80.0):
        psi = np.radians(psi_deg)
        gamma_h = pathloss.fresnel_reflection_coefficient(psi, eps_g, "horizontal")
        gamma_v = pathloss.fresnel_reflection_coefficient(psi, eps_g, "vertical")
        assert abs(gamma_h) == pytest.approx(0.0, abs=1e-9)
        assert abs(gamma_v) == pytest.approx(0.0, abs=1e-9)


def test_fresnel_reflection_magnitude_bounded_for_lossless_ground():
    """|Gamma| darf fuer einen verlustfreien, passiven Boden nie > 1 werden
    -- eine Grenzflaeche kann keine Leistung erzeugen."""
    eps_g = pathloss.complex_relative_permittivity(
        epsilon_r=15.0, conductivity_s_per_m=0.0, wavelength_m=0.05169
    )
    for psi_deg in np.linspace(1.0, 89.0, 20):
        psi = np.radians(psi_deg)
        for pol in ("horizontal", "vertical"):
            gamma = pathloss.fresnel_reflection_coefficient(psi, eps_g, pol)
            assert abs(gamma) <= 1.0 + 1e-9


def test_ament_roughness_smooth_ground_is_lossless():
    """h_rms = 0 (perfekt glatter Boden) darf die Reflexion nicht zusaetzlich
    daempfen -- rho_s muss exakt 1 sein."""
    rho = pathloss.ament_roughness_factor(
        grazing_angle_rad=np.radians(10.0), height_rms_m=0.0, wavelength_m=0.05169
    )
    assert rho == pytest.approx(1.0)


def test_ament_roughness_decreases_with_roughness():
    psi = np.radians(10.0)
    lam = 0.05169
    rho_smooth = pathloss.ament_roughness_factor(psi, height_rms_m=0.01, wavelength_m=lam)
    rho_rough = pathloss.ament_roughness_factor(psi, height_rms_m=0.5, wavelength_m=lam)
    assert 0.0 <= rho_rough < rho_smooth <= 1.0


def test_two_ray_extra_loss_vanishes_without_reflection():
    """Mit eps_r = 1 (kein Reflexionskoeffizient) darf der Two-Ray-Zusatzterm
    exakt 0 dB sein -- die Formel muss sich dann auf reines FSPL reduzieren."""
    lam = constants.wavelength(5.8e9)
    extra_db = pathloss.two_ray_extra_loss_db(
        d_ground=3000.0, h_tx=100.0, h_rx=2.0, wavelength_m=lam,
        epsilon_r=1.0, conductivity_s_per_m=0.0, height_rms_m=0.0,
    )
    assert extra_db == pytest.approx(0.0, abs=1e-6)


def test_two_ray_path_loss_matches_fspl_without_reflection():
    lam = constants.wavelength(5.8e9)
    d_direct = geometry.slant_range(3000.0, 100.0, 2.0)
    fspl = pathloss.free_space_path_loss(d_direct, lam)
    two_ray = pathloss.two_ray_path_loss_db(
        d_ground=3000.0, h_tx=100.0, h_rx=2.0, wavelength_m=lam,
        epsilon_r=1.0, conductivity_s_per_m=0.0, height_rms_m=0.0,
    )
    assert two_ray == pytest.approx(fspl, abs=1e-6)


# --- pathloss.py: Atmosphaere/Regen (Formelstruktur, keine eingebauten Koeff.) --


def test_atmospheric_attenuation_zero_coefficient_is_zero_loss():
    assert pathloss.atmospheric_attenuation_db(6000.0, specific_attenuation_db_per_km=0.0) == 0.0


def test_atmospheric_attenuation_scales_linearly_with_distance():
    loss_3km = pathloss.atmospheric_attenuation_db(3000.0, specific_attenuation_db_per_km=0.02)
    loss_6km = pathloss.atmospheric_attenuation_db(6000.0, specific_attenuation_db_per_km=0.02)
    assert loss_6km == pytest.approx(2.0 * loss_3km)


def test_rain_attenuation_zero_rain_is_zero_loss():
    # Ergebnis muss unabhaengig von k/alpha exakt 0 sein, wenn es nicht regnet.
    assert pathloss.rain_attenuation_db(6000.0, rain_rate_mm_per_hr=0.0, k=0.01, alpha=1.3) == 0.0


def test_rain_attenuation_power_law_in_rain_rate():
    l1 = pathloss.rain_attenuation_db(1000.0, rain_rate_mm_per_hr=10.0, k=0.01, alpha=1.3)
    l2 = pathloss.rain_attenuation_db(1000.0, rain_rate_mm_per_hr=20.0, k=0.01, alpha=1.3)
    assert l2 / l1 == pytest.approx(2.0**1.3)


# --- environment.py: Al-Hourani-Clutter, Vegetation, Stufenparameter -------


def test_weissberger_10m_5_8ghz():
    loss = environment.vegetation_loss_weissberger_db(depth_m=10.0, frequency_hz=5.8e9)
    assert loss == pytest.approx(7.4, abs=0.05)


def test_weissberger_vectorizes_and_increases_with_depth():
    depths = np.array([5.0, 10.0, 20.0, 100.0, 300.0])
    losses = environment.vegetation_loss_weissberger_db(depths, frequency_hz=5.8e9)
    assert losses.shape == depths.shape
    assert np.all(np.diff(losses) > 0)


def test_weissberger_branches_roughly_continuous_at_14m():
    """Bekannte kleine Unstetigkeit des Weissberger-Modells am 14-m-Uebergang
    (Modified Exponential Decay) -- muss klein bleiben (<10 %), nicht exakt 0."""
    just_below = environment.vegetation_loss_weissberger_db(13.99, 5.8e9)
    just_above = environment.vegetation_loss_weissberger_db(14.01, 5.8e9)
    assert abs(just_above - just_below) / just_below < 0.10


def test_los_probability_approaches_one_at_zenith():
    """Sender senkrecht ueber dem Empfaenger (epsilon=90 Grad): P_LOS -> 1
    fuer jede Stufe. Ein Sigmoid erreicht 1 nur asymptotisch, deshalb keine
    zu enge Toleranz -- "nur eta_LOS bleibt uebrig" heisst "praktisch nur",
    nicht "auf Maschinengenauigkeit exakt"."""
    for stage in environment.STAGES.values():
        p_los = environment.los_probability(90.0, stage)
        assert p_los == pytest.approx(1.0, abs=5e-3)


def test_clutter_loss_only_eta_los_remains_at_zenith():
    """Direkt aus INSTRUCITONS.md: 'Senkrecht ueber dem Empfaenger bleibt nur
    eta_LOS uebrig.' (asymptotisch, siehe test_los_probability_approaches_one_at_zenith)."""
    for stage in environment.STAGES.values():
        loss = environment.clutter_loss_al_hourani_db(90.0, stage)
        assert loss == pytest.approx(stage.eta_los_db, abs=0.1)


def test_clutter_loss_monotonic_across_stages_and_elevation():
    """Strukturtest aus INSTRUCITONS.md: eine dichtere Umgebungsstufe darf das
    Signal nie verbessern -- L_clutter(Stufe k) <= L_clutter(Stufe k+1) fuer
    JEDEN Elevationswinkel, nicht nur im Basisszenario."""
    elevations = np.linspace(0.1, 90.0, 900)
    losses_by_stage = [
        environment.clutter_loss_al_hourani_db(elevations, environment.STAGES[k])
        for k in (1, 2, 3, 4)
    ]
    for lower, higher in zip(losses_by_stage, losses_by_stage[1:]):
        assert np.all(higher >= lower - 1e-9)


def test_clutter_loss_baseline_scenario_stage1():
    """3 km/100 m/2 m, Stufe 1 -- Zwischenwert aus der eigenen Nachrechnung
    (nicht direkt in INSTRUCITONS.md tabelliert, aber Teil der Kette, die auf
    die Verifikationstabelle fuehrt)."""
    eps_deg = geometry.elevation_angle_deg(d_ground=3000.0, h_tx=100.0, h_rx=2.0)
    loss = environment.clutter_loss_al_hourani_db(eps_deg, environment.STAGES[1])
    assert loss == pytest.approx(5.631, abs=0.01)


def test_clutter_loss_fixed_is_passthrough():
    assert environment.clutter_loss_fixed_db(12.3) == 12.3


def test_frequency_correction_zero_at_reference():
    assert environment.frequency_correction_db(2.0e9, reference_hz=2.0e9) == pytest.approx(0.0)


def test_frequency_correction_positive_above_reference():
    corr = environment.frequency_correction_db(5.8e9, reference_hz=2.0e9)
    assert corr > 0.0
    assert corr == pytest.approx(10.0 * np.log10(5.8e9 / 2.0e9))


def test_stage_2_eta_nlos_corrected_for_monotonicity():
    """Fallstrick 3: 18,0 dB statt der publizierten 21,0 dB, damit die Skala
    monoton bleibt -- Regressionsschutz gegen versehentliches Zuruecksetzen
    auf den publizierten Wert."""
    assert environment.STAGES[2].eta_nlos_db == 18.0
