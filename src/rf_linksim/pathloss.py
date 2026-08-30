"""Streckendaempfung: Freiraum (FSPL), Two-Ray-Bodenreflexion, Atmosphaere/Regen.

**Freiraumdaempfung (FSPL)** ist der einzige Term, der im Basisszenario ohne
weitere Annahmen greift -- reine Kugelwellen-Ausbreitung, ITU-R P.525.

**Two-Ray-Bodenreflexion** ist implementiert, aber per Vorgabe aus
INSTRUCITONS.md standardmaessig **aus**: bei 5,8 GHz ist die Lobing-Struktur
extrem feinstufig (eine Interferenzperiode entspricht wenigen Metern
Flughoehe), und ob sie sich real ausbildet, haengt am Rayleigh-Kriterium fuer
Bodenrauigkeit (glatter als ~20 cm RMS noetig). Ob dieses Modell fuer eine
gegebene Szene ueberhaupt zutrifft, ist eine Annahme, keine Messung -- deshalb
bleibt es ein bewusst zuschaltbarer Modus, kein Default.

**Atmosphaerische Gasdaempfung und Regendaempfung** sind hier nur als
Formelstruktur hinterlegt (spezifische Daempfung * Streckenlaenge, bzw. das
ITU-R-P.838-Potenzgesetz fuer Regen), OHNE eingebaute Koeffiziententabelle.
Der Grund: die exakten ITU-R-P.676/P.838-Tabellenwerte bei 5,8 GHz konnten in
dieser Session nicht gegen eine lesbare Quelle verifiziert werden -- ein
stillschweigend eingebauter, ungeprüfter Zahlenwert waere hier riskanter als
ein Parameter, den der Aufrufer explizit und nachvollziehbar setzen muss.
Siehe die docstrings der beiden Funktionen fuer die Fundstellen, die noch
geprueft werden muessen, bevor daraus ein Default wird.
"""

import numpy as np

from rf_linksim.geometry import slant_range

# --- Freiraumdaempfung -------------------------------------------------------


def free_space_path_loss(d_m, wavelength_m):
    """Freiraumdaempfung (ITU-R P.525), dB.

    L = 20*log10(4*pi*d / lambda)

    d ist die tatsaechliche Ausbreitungsstrecke (Schraegentfernung fuer den
    Direktstrahl, die laengere Umwegstrecke fuer den reflektierten Strahl im
    Two-Ray-Modell) -- niemals die Grundentfernung.
    """
    return 20.0 * np.log10(4.0 * np.pi * d_m / wavelength_m)


# --- Two-Ray-Bodenreflexion (Default: aus) ----------------------------------


def two_ray_breakpoint_distance(h_tx, h_rx, wavelength_m):
    """Breakpoint-Entfernung des Two-Ray-Modells, m.

    d_bp = 4 * h_tx * h_rx / lambda

    Unterhalb dieser Distanz liegt die Szene in der Interferenzzone (Direkt-
    und Bodenstrahl interferieren sichtbar konstruktiv/destruktiv je nach
    Hoehe/Entfernung), oberhalb geht die Interferenz in einen glatten
    d^4-Abfall ueber. Bei 100 m/2 m/5,8 GHz liegt der Breakpoint bei 15,5 km
    -- der gesamte fuer FPV relevante Entfernungsbereich liegt also *innerhalb*
    der Interferenzzone, falls Two-Ray ueberhaupt zutrifft.
    """
    return 4.0 * h_tx * h_rx / wavelength_m


def complex_relative_permittivity(epsilon_r, conductivity_s_per_m, wavelength_m):
    """Komplexe relative Permittivitaet eines verlustbehafteten Bodens.

    eps_g = eps_r - j * 60 * lambda * sigma

    sigma in S/m, lambda in m. Die Konstante 60 = 1/(2*pi*c*eps_0) fasst die
    Umrechnung von Leitfaehigkeit in einen verlustbehafteten Imaginaerteil bei
    gegebener Wellenlaenge zusammen (Standardform, z. B. Balanis/Rappaport).
    Fuer sigma = 0 reduziert sich das auf reelles eps_r (verlustfreier Boden).
    """
    return epsilon_r - 1j * 60.0 * wavelength_m * conductivity_s_per_m


def fresnel_reflection_coefficient(grazing_angle_rad, epsilon_g, polarization="horizontal"):
    """Fresnel-Reflexionskoeffizient (komplex) an der Bodenebene.

    Mit dem *Grazing*-Winkel psi (Winkel zwischen Bodenebene und einfallendem/
    reflektiertem Strahl, NICHT der Winkel zur Flaechennormalen):

    Gamma(psi) = (sin psi - X) / (sin psi + X)

    horizontale Polarisation:  X = sqrt(eps_g - cos^2 psi)
    vertikale Polarisation:    X = sqrt(eps_g - cos^2 psi) / eps_g

    eps_g ist die (ggf. komplexe) relative Permittivitaet des Bodens, siehe
    complex_relative_permittivity(). Gegenprobe gegen die ueblichere Form mit
    Einfallswinkel theta_i (gemessen von der Flaechennormalen, psi = 90-theta_i)
    liefert dieselbe Formel -- beide Konventionen sind hier konsistent.
    """
    grazing_angle_rad = np.asarray(grazing_angle_rad, dtype=complex)
    sin_psi = np.sin(grazing_angle_rad)
    cos2_psi = np.cos(grazing_angle_rad) ** 2
    root = np.sqrt(epsilon_g - cos2_psi)

    if polarization == "horizontal":
        x = root
    elif polarization == "vertical":
        x = root / epsilon_g
    else:
        raise ValueError(f"polarization muss 'horizontal' oder 'vertical' sein, nicht {polarization!r}")

    return (sin_psi - x) / (sin_psi + x)


def ament_roughness_factor(grazing_angle_rad, height_rms_m, wavelength_m):
    """Ament-Streuverlustfaktor rho_s fuer raue Bodenreflexion, linear (0..1).

    rho_s = exp[-8 * (pi * h_rms * sin(psi) / lambda)^2]

    h_rms ist die RMS-Hoehenabweichung der Oberflaeche (Rauigkeit). h_rms = 0
    (perfekt glatt) ergibt rho_s = 1 (keine Zusatzdaempfung durch Rauigkeit,
    reine Fresnel-Reflexion). Das Rayleigh-Kriterium (glatt, wenn h_rms klein
    gegen lambda/(8*sin psi) ist) ist in dieser Formel implizit enthalten,
    nicht separat zu pruefen.
    """
    g = np.pi * height_rms_m * np.sin(grazing_angle_rad) / wavelength_m
    return np.exp(-8.0 * g**2)


def two_ray_extra_loss_db(
    d_ground,
    h_tx,
    h_rx,
    wavelength_m,
    epsilon_r=15.0,
    conductivity_s_per_m=0.005,
    polarization="horizontal",
    height_rms_m=0.0,
):
    """Zusatzterm (dB) zur Freiraumdaempfung des Direktstrahls durch Bodenreflexion.

    Positiv = zusaetzlicher Verlust (destruktive Interferenz), negativ =
    Gewinn gegenueber reinem Freiraumfall (konstruktive Interferenz, bis zu
    +6 dB bei perfekt kohaerenter Verdopplung der Feldstaerke). Zu addieren zu
    free_space_path_loss(d_direct, wavelength_m), NICHT zu ersetzen.

    epsilon_r/conductivity_s_per_m sind Boden-ANNAHMEN (hier: Richtwerte fuer
    mittelfeuchten Erdboden), height_rms_m = 0 bedeutet perfekt glatte
    Reflexionsflaeche (kein Ament-Abschlag). Alle drei sind Szenario-
    Annahmen, keine Messwerte -- im config.yaml entsprechend kennzeichnen.

    Geometrie (Spiegelmethode, ebene Erde):
      d_direkt    = sqrt(d_ground^2 + (h_tx - h_rx)^2)
      d_reflekt   = sqrt(d_ground^2 + (h_tx + h_rx)^2)
      psi (Grazing) = atan2(h_tx + h_rx, d_ground)
    """
    d_direct = slant_range(d_ground, h_tx, h_rx)
    d_reflected = np.sqrt(d_ground**2 + (h_tx + h_rx) ** 2)
    grazing_angle = np.arctan2(h_tx + h_rx, d_ground)

    epsilon_g = complex_relative_permittivity(epsilon_r, conductivity_s_per_m, wavelength_m)
    gamma = fresnel_reflection_coefficient(grazing_angle, epsilon_g, polarization)
    rho = ament_roughness_factor(grazing_angle, height_rms_m, wavelength_m)
    gamma_eff = rho * gamma

    delta_phase = 2.0 * np.pi / wavelength_m * (d_reflected - d_direct)
    field_ratio = 1.0 + gamma_eff * (d_direct / d_reflected) * np.exp(-1j * delta_phase)
    power_ratio = np.abs(field_ratio) ** 2

    with np.errstate(divide="ignore"):
        extra_gain_db = 10.0 * np.log10(power_ratio)
    return -extra_gain_db


def two_ray_path_loss_db(d_ground, h_tx, h_rx, wavelength_m, **two_ray_kwargs):
    """FSPL des Direktstrahls plus Two-Ray-Interferenzterm, dB. Bequemlichkeitsfunktion."""
    d_direct = slant_range(d_ground, h_tx, h_rx)
    return free_space_path_loss(d_direct, wavelength_m) + two_ray_extra_loss_db(
        d_ground, h_tx, h_rx, wavelength_m, **two_ray_kwargs
    )


# --- Atmosphaerische Gasdaempfung und Regendaempfung ------------------------
#
# Beide Funktionen unten sind absichtlich ohne eingebaute Koeffiziententabelle.
# NICHT verwenden, ohne den jeweiligen Koeffizienten vorher gegen die
# Originalquelle geprueft zu haben:
#   Atmosphaere: ITU-R P.676-13, Tabellen fuer spezifische Daempfung
#                (Sauerstoff + Wasserdampf) bei der Zielfrequenz.
#   Regen:       ITU-R P.838-3, Tabelle 1 (Koeffizienten k, alpha nach
#                Frequenz und Polarisation, log-log-Interpolation zwischen
#                Stuetzstellen).
# Beide Modelle sind im Basisszenario nicht referenziert (keine Sollwerte in
# INSTRUCITONS.md) und standardmaessig durch specific_attenuation_db_per_km=0
# bzw. rain_rate_mm_per_hr=0 wirkungslos, bis jemand gepruefte Werte eintraegt.


def atmospheric_attenuation_db(d_slant_m, specific_attenuation_db_per_km):
    """Atmosphaerische Gasdaempfung = spezifische Daempfung * Streckenlaenge.

    L = specific_attenuation_db_per_km * (d_slant_m / 1000)

    specific_attenuation_db_per_km ist bewusst ein Pflichtparameter ohne
    Default -- siehe Modul-Docstring. Bei 5,8 GHz und wenigen km Strecke ist
    der Effekt ohnehin klein (typischerweise << 1 dB), aber "klein" ist keine
    Rechtfertigung fuer eine ungeprüfte Zahl.
    """
    return specific_attenuation_db_per_km * (d_slant_m / 1000.0)


def rain_attenuation_db(d_slant_m, rain_rate_mm_per_hr, k, alpha):
    """Regendaempfung nach dem ITU-R-P.838-Potenzgesetz.

    gamma_R = k * R^alpha   [dB/km], R = Regenrate in mm/h
    L = gamma_R * (d_slant_m / 1000)

    k und alpha sind bewusst Pflichtparameter ohne Default -- siehe
    Modul-Docstring. Ohne Regen (rain_rate_mm_per_hr = 0) ist das Ergebnis
    unabhaengig von k/alpha exakt 0.
    """
    specific_attenuation = k * rain_rate_mm_per_hr**alpha
    return specific_attenuation * (d_slant_m / 1000.0)
