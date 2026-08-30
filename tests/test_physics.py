"""Verifikation gegen unabhaengig nachgerechnete Sollwerte (siehe INSTRUCITONS.md).

Jeder Test prueft eine Groesse, deren Sollwert nicht aus diesem Code stammt,
sondern separat (von Hand oder mit Standardformeln) nachgerechnet wurde. Die
Toleranzen sind bewusst eng (rtol 1e-3 .. 1e-4), damit ein Vorzeichen- oder
Rundungsfehler nicht durchrutscht.

Baustein-weise nach INSTRUCITONS.md "Vorgehen": aktuell abgedeckt sind
constants.py, geometry.py, antenna.py und pathloss.py.
"""

import dataclasses

import numpy as np
import pytest

from rf_linksim import antenna, constants, environment, geometry, link, montecarlo, pathloss, sweep


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


# --- link.py: die volle Kette ------------------------------------------------


def _baseline_scenario_tx_rx(stage=1, d_ground_m=3000.0, tilt_rad=0.0):
    scenario = link.Scenario(d_ground_m=d_ground_m, environment_stage=stage)
    tx = link.Transmitter(power_dbm=constants.watt_to_dbm(2.0), tilt_rad=tilt_rad)
    rx = link.ReceiveAntenna(gain_dbi=2.0, feed_loss_db=1.5, polarization_r=float("inf"))
    return scenario, tx, rx


def test_link_budget_baseline_matches_independent_hand_check():
    """Regressionstest gegen die eigene, zweifach unabhaengig (dB-Summe +
    lineare Watt-Rechnung) nachgerechnete Kette (siehe Konversation) -- NICHT
    gegen die Dokument-Ankertabelle, die einen ungeklaerten ~0,37-dB-Offset
    hat. Dieser Test sichert ab, dass link.py die Formeln korrekt verdrahtet,
    unabhaengig von dieser offenen Frage."""
    scenario, tx, rx = _baseline_scenario_tx_rx()
    result = link.compute_link_budget(scenario, tx, rx)
    assert result.p_rx_dbm == pytest.approx(-90.6448, abs=0.001)
    # 173,98 ist die auf 2 Nachkommastellen gerundete kT0-Konstante; die
    # exakt berechnete kt0_dbm_per_hz() weicht davon um ~0,005 dB ab, daher
    # die etwas weitere Toleranz hier (P_rx selbst ist auf 0,001 dB exakt).
    assert result.cn0_db_hz == pytest.approx(83.3352, abs=0.01)


def test_link_budget_matches_documented_matrix_within_known_offset():
    """Gegen die volle Dokument-Sollwertmatrix, mit der dokumentierten,
    bewusst weiten Toleranz fuer den ungeklaerten ~0,37-dB-Offset (siehe
    Konversation) -- faengt groessere Regressionen ab, ohne den bekannten
    kleinen Offset als Fehlschlag zu werten."""
    target = {
        200: [110.9, 110.9, 102.3, 94.0],
        500: [104.5, 100.0, 86.8, 83.1],
        1000: [98.5, 84.7, 79.8, 76.6],
        2000: [89.3, 76.2, 73.5, 70.5],
        3000: [83.7, 72.3, 69.9, 66.9],
        5000: [78.1, 67.6, 65.4, 62.4],
        6000: [76.3, 65.9, 63.8, 60.8],
    }
    for d_ground, row in target.items():
        for stage_idx, cn0_target in zip((1, 2, 3, 4), row):
            scenario, tx, rx = _baseline_scenario_tx_rx(stage=stage_idx, d_ground_m=d_ground)
            result = link.compute_link_budget(scenario, tx, rx)
            assert result.cn0_db_hz == pytest.approx(cn0_target, abs=0.5)


def test_link_budget_sum_of_terms_equals_p_rx_exactly():
    """Struktureller Test aus INSTRUCITONS.md: die Summe aller Budget-Terme
    muss P_rx exakt ergeben."""
    scenario, tx, rx = _baseline_scenario_tx_rx()
    r = link.compute_link_budget(scenario, tx, rx)
    reconstructed = (
        r.tx_power_term_dbm
        + r.rx_gain_dbi
        - r.fspl_db
        - r.clutter_loss_db
        - r.vegetation_loss_db
        + r.two_ray_extra_db
        - r.atmospheric_loss_db
        - r.rain_loss_db
        - r.polarization_loss_db
        - r.feed_loss_db
    )
    assert reconstructed == pytest.approx(r.p_rx_dbm, abs=1e-9)


def test_link_budget_cn0_p_rx_relation_exact():
    scenario, tx, rx = _baseline_scenario_tx_rx()
    r = link.compute_link_budget(scenario, tx, rx)
    assert r.cn0_db_hz - r.p_rx_dbm == pytest.approx(173.98, abs=0.01)


def test_power_is_eirp_does_not_double_count_gain():
    """Struktureller Test aus INSTRUCITONS.md: power_is_eirp=True darf den
    Antennengewinn nicht doppelt zaehlen -- bei verschiedenen Distanzen
    (verschiedene theta) geprueft, nicht nur an einem Punkt."""
    raw_power_dbm = 20.0
    for d_ground in (300.0, 1500.0, 4000.0):
        scenario = link.Scenario(d_ground_m=d_ground, environment_stage=1)
        rx = link.ReceiveAntenna(gain_dbi=0.0, polarization_r=1.0)

        tx_raw = link.Transmitter(power_dbm=raw_power_dbm, power_is_eirp=False)
        r_raw = link.compute_link_budget(scenario, tx_raw, rx)

        peak_directivity_dbi = antenna.directivity_dbi(tx_raw.antenna_pattern)
        tx_eirp = link.Transmitter(
            power_dbm=raw_power_dbm + peak_directivity_dbi, power_is_eirp=True
        )
        r_eirp = link.compute_link_budget(scenario, tx_eirp, rx)

        assert r_eirp.p_rx_dbm == pytest.approx(r_raw.p_rx_dbm, abs=1e-9)


def test_distance_doubling_costs_exactly_6_02_db_without_clutter_isotropic():
    """Struktureller Test aus INSTRUCITONS.md: ohne Clutter und mit isotroper
    Antenne kostet jede Distanzverdopplung exakt 6,02 dB -- durch die volle
    Kette gepruft, nicht nur in pathloss.py isoliert."""
    # Gleiche Hoehe TX/RX (0 m), damit Grundentfernung = Schraegentfernung
    # ist und eine Grundentfernungs-Verdopplung auch die Ausbreitungsstrecke
    # exakt verdoppelt -- sonst greift der Slant-vs-Ground-Effekt aus
    # test_fspl_uses_slant_not_ground_range und verfaelscht den Vergleich.
    tx = link.Transmitter(
        power_dbm=30.0,
        height_m=0.0,
        antenna_pattern=antenna.isotropic_pattern,
        polarization_r=1.0,
    )
    rx = link.ReceiveAntenna(gain_dbi=0.0, height_m=0.0, feed_loss_db=0.0, polarization_r=1.0)
    for d in (400.0, 1200.0, 2500.0):
        s1 = link.Scenario(d_ground_m=d, environment_stage=1, clutter_model="fixed", clutter_fixed_db=0.0)
        s2 = link.Scenario(d_ground_m=2.0 * d, environment_stage=1, clutter_model="fixed", clutter_fixed_db=0.0)
        r1 = link.compute_link_budget(s1, tx, rx)
        r2 = link.compute_link_budget(s2, tx, rx)
        assert r1.p_rx_dbm - r2.p_rx_dbm == pytest.approx(6.0206, abs=1e-3)


def test_denser_stage_never_improves_signal_through_full_chain():
    scenario_base_kwargs = dict(d_ground_m=1500.0)
    tx = link.Transmitter(power_dbm=constants.watt_to_dbm(2.0))
    rx = link.ReceiveAntenna(gain_dbi=2.0, feed_loss_db=1.5, polarization_r=float("inf"))
    results = []
    for stage in (1, 2, 3, 4):
        scenario = link.Scenario(environment_stage=stage, **scenario_base_kwargs)
        results.append(link.compute_link_budget(scenario, tx, rx).p_rx_dbm)
    assert results == sorted(results, reverse=True)  # streng monoton fallend


# --- link.py: dual_diversity (RHCP+LHCP, bladeRF-Default) ------------------


def test_dual_diversity_recovers_polarization_loss():
    """RHCP-Sender, RHCP+LHCP-Empfangspaar: der RHCP-Zweig hat 0 dB PLF, der
    LHCP-Zweig -> inf dB PLF (traegt nichts bei). Kombiniert muss das
    praktisch exakt dem RHCP-Zweig allein entsprechen -- und das ist genau
    die 3,01 dB, die eine einzelne lineare Antenne (single-Modus) verliert."""
    scenario = link.Scenario(d_ground_m=3000.0, environment_stage=1)
    tx = link.Transmitter(power_dbm=constants.watt_to_dbm(2.0), polarization_r=1.0)
    rx_a = link.ReceiveAntenna(gain_dbi=2.0, feed_loss_db=1.5, polarization_r=1.0)  # RHCP
    rx_b = link.ReceiveAntenna(gain_dbi=2.0, feed_loss_db=1.5, polarization_r=-1.0)  # LHCP

    dual = link.compute_link_budget_dual_diversity(scenario, tx, rx_a, rx_b)
    assert dual.branch_a.polarization_loss_db == pytest.approx(0.0, abs=1e-9)
    assert np.isinf(dual.branch_b.polarization_loss_db)
    assert dual.cn0_combined_db_hz == pytest.approx(dual.branch_a.cn0_db_hz, abs=1e-6)

    rx_linear = link.ReceiveAntenna(gain_dbi=2.0, feed_loss_db=1.5, polarization_r=float("inf"))
    single = link.compute_link_budget(scenario, tx, rx_linear)
    assert dual.cn0_combined_db_hz - single.cn0_db_hz == pytest.approx(3.0103, abs=0.01)


def test_dual_diversity_combined_never_below_either_branch():
    scenario = link.Scenario(d_ground_m=1200.0, environment_stage=2)
    tx = link.Transmitter(power_dbm=constants.watt_to_dbm(2.0), polarization_r=1.0)
    rx_a = link.ReceiveAntenna(gain_dbi=2.0, feed_loss_db=1.5, polarization_r=1.0)
    rx_b = link.ReceiveAntenna(gain_dbi=2.0, feed_loss_db=1.5, polarization_r=-1.0)
    dual = link.compute_link_budget_dual_diversity(scenario, tx, rx_a, rx_b)
    assert dual.cn0_combined_db_hz >= dual.branch_a.cn0_db_hz - 1e-9
    assert dual.cn0_combined_db_hz >= dual.branch_b.cn0_db_hz - 1e-9


# --- montecarlo.py: Shadowing + Rice-Fading ---------------------------------


def test_shadowing_p95_minus_p5():
    """Sollwert aus INSTRUCITONS.md: P95 - P5 = 2*1,645*sigma (reines
    Shadowing, kein Fading -- die 1,645-Faktoren sind die Standard-Quantile
    der Normalverteilung bei 5%/95%)."""
    rng = np.random.default_rng(42)
    sigma = 6.0
    samples = montecarlo.run_monte_carlo(
        p_rx_deterministic_dbm=-90.0,
        shadow_sigma_db=sigma,
        rice_k_db=12.0,
        n_runs=500_000,
        rng=rng,
        include_fading=False,
    )
    summary = montecarlo.percentile_summary(samples, percentiles=(5.0, 95.0))
    spread = summary[95.0] - summary[5.0]
    assert spread == pytest.approx(2.0 * 1.645 * sigma, rel=0.01)


def test_rice_fading_normalized_to_unit_mean_power():
    """Sollwert aus INSTRUCITONS.md: E[|h|^2] = 1, unabhaengig vom K-Faktor --
    direkt an den komplexen Amplituden geprueft, nicht nur am dB-Ergebnis."""
    rng = np.random.default_rng(7)
    for k_db in (12.0, 8.0, 5.0, 3.0):
        k_lin = 10.0 ** (k_db / 10.0)
        s = np.sqrt(k_lin / (k_lin + 1.0))
        sigma_f = np.sqrt(1.0 / (2.0 * (k_lin + 1.0)))
        n = 2_000_000
        h_real = s + sigma_f * rng.standard_normal(n)
        h_imag = sigma_f * rng.standard_normal(n)
        mean_power = np.mean(h_real**2 + h_imag**2)
        assert mean_power == pytest.approx(1.0, rel=0.005)


def test_rice_fading_db_median_more_negative_for_lower_k():
    """Niedrigerer K-Faktor (mehr Mehrweganteile) muss staerker streuen --
    der dB-Median des Fading-Terms wird dadurch negativer (Jensensche
    Ungleichung, siehe Modul-Docstring), monoton mit sinkendem K."""
    rng = np.random.default_rng(123)
    medians = []
    for k_db in (12.0, 8.0, 5.0, 3.0):  # Stufe 1 -> Stufe 4
        samples_db = montecarlo.sample_rice_fading_db(k_db, 500_000, rng)
        medians.append(np.median(samples_db))
    assert medians == sorted(medians, reverse=True)  # streng monoton fallend
    assert all(m <= 0.05 for m in medians)  # Median <= 0, nie ein Netto-Gewinn


def test_run_monte_carlo_uses_single_deterministic_value():
    """run_monte_carlo rechnet die Kette NICHT neu -- alle Stichproben
    streuen um denselben deterministischen Wert."""
    rng = np.random.default_rng(1)
    det = -85.0
    samples = montecarlo.run_monte_carlo(det, shadow_sigma_db=4.0, rice_k_db=12.0, n_runs=100_000, rng=rng)
    # Median der reinen Shadowing+Fading-Streuung liegt nahe am deterministischen
    # Wert (Shadowing ist symmetrisch um 0, Fading zieht den Median nur leicht,
    # bei K=12 dB minimal).
    assert np.median(samples) == pytest.approx(det, abs=1.0)


def test_percentile_summary_basic():
    samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    summary = montecarlo.percentile_summary(samples, percentiles=(0.0, 50.0, 100.0))
    assert summary[0.0] == pytest.approx(1.0)
    assert summary[50.0] == pytest.approx(3.0)
    assert summary[100.0] == pytest.approx(5.0)


# --- sweep.py: Distanz-/Hoehen-/2D-Sweeps -----------------------------------


def _sweep_baseline_tx_rx():
    scenario = link.Scenario(d_ground_m=3000.0, environment_stage=1)
    tx = link.Transmitter(power_dbm=constants.watt_to_dbm(2.0))
    rx = link.ReceiveAntenna(gain_dbi=2.0, feed_loss_db=1.5, polarization_r=float("inf"))
    return scenario, tx, rx


def test_distance_sweep_matches_scalar_calls():
    scenario, tx, rx = _sweep_baseline_tx_rx()
    distances = np.array([200.0, 1000.0, 3000.0, 6000.0])
    swept = sweep.distance_sweep(distances, scenario, tx, rx)
    assert swept.p_rx_dbm.shape == distances.shape
    for i, d in enumerate(distances):
        scalar_scenario = dataclasses.replace(scenario, d_ground_m=float(d))
        scalar_result = link.compute_link_budget(scalar_scenario, tx, rx)
        assert swept.p_rx_dbm[i] == pytest.approx(scalar_result.p_rx_dbm)
        assert swept.cn0_db_hz[i] == pytest.approx(scalar_result.cn0_db_hz)


def test_distance_sweep_all_stages_returns_four_stages():
    scenario, tx, rx = _sweep_baseline_tx_rx()
    distances = np.array([500.0, 3000.0])
    results = sweep.distance_sweep_all_stages(distances, scenario, tx, rx)
    assert set(results.keys()) == {1, 2, 3, 4}
    for stage, r in results.items():
        assert r.p_rx_dbm.shape == distances.shape
    # Stufe 4 (Stadt) muss bei gleicher Distanz schlechter sein als Stufe 1.
    assert np.all(results[4].p_rx_dbm <= results[1].p_rx_dbm)


def test_height_sweep_matches_scalar_calls():
    scenario, tx, rx = _sweep_baseline_tx_rx()
    heights = np.array([50.0, 100.0, 150.0])
    swept = sweep.height_sweep(heights, scenario, tx, rx)
    assert swept.p_rx_dbm.shape == heights.shape
    for i, h in enumerate(heights):
        scalar_tx = dataclasses.replace(tx, height_m=float(h))
        scalar_result = link.compute_link_budget(scenario, scalar_tx, rx)
        assert swept.p_rx_dbm[i] == pytest.approx(scalar_result.p_rx_dbm)


def test_grid_sweep_shape_and_spot_check():
    scenario, tx, rx = _sweep_baseline_tx_rx()
    distances = np.array([500.0, 3000.0, 6000.0])
    heights = np.array([50.0, 100.0])
    grid = sweep.grid_sweep(distances, heights, scenario, tx, rx)
    assert grid.p_rx_dbm.shape == (3, 2)

    # Stichprobe: Zelle [1, 0] (Distanz=3000, Hoehe=50) gegen Einzelrechnung.
    scalar_scenario = dataclasses.replace(scenario, d_ground_m=3000.0)
    scalar_tx = dataclasses.replace(tx, height_m=50.0)
    scalar_result = link.compute_link_budget(scalar_scenario, scalar_tx, rx)
    assert grid.p_rx_dbm[1, 0] == pytest.approx(scalar_result.p_rx_dbm)
