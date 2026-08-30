"""Antennen-Richtcharakteristiken, Direktivitaet und Polarisations-Mismatch.

Drei unabhaengige physikalische Effekte, die hier zusammenkommen:

1. **Winkel zur Antennenachse.** Die Sendeantenne haengt starr an der Drohne,
   ihre mechanische Achse (Boresight) zeigt nach unten (Nadir), ggf. um einen
   Nickwinkel `tilt` geneigt (Vorwaertsflug). Wie weit die Blickrichtung zum
   Empfaenger von dieser Achse absteht, bestimmt `angle_off_boresight()`.

2. **Richtcharakteristik F(theta).** Wie stark die Antenne in eine gegebene
   Richtung relativ zu ihrem Maximum abstrahlt, F auf 1 normiert. Zwei Muster
   sind hier hinterlegt: der ideale Halbwellendipol (exakte Lehrbuchformel,
   dient als Verifikationsanker) und ein Pagoda-Modell (sin^n mit Fill-Level),
   das die reale RHCP-Pagoda-Antenne am FPV-VTX approximiert.

3. **Direktivitaet D.** Der Gewinnfaktor, der aus der Buendelung der
   Abstrahlung folgt (verglichen mit einem Kugelstrahler gleicher
   Gesamtleistung). Wird numerisch aus F(theta) integriert, nicht als Zahl
   eingetragen -- das haelt sie konsistent, wenn Formparameter (n, Fill-Level)
   geaendert werden, und macht sie testbar.

4. **Polarisations-Mismatch (PLF).** Wie viel Leistung durch unterschiedliche
   Polarisation von Sender und Empfaenger verloren geht -- unabhaengig von
   Richtcharakteristik und Entfernung.

Alle Muster hier sind azimutal symmetrisch (haengen nur von theta ab, nicht
von phi) -- realistisch fuer eine senkrecht haengende Rundstrahl-/Pagoda-
Antenne und fuer den idealen Dipol. Ein Azimutwinkel taucht deshalb nirgends
auf.
"""

import numpy as np
from scipy import integrate

from rf_linksim.constants import lin_to_db

# --- Winkel zur Antennenachse -------------------------------------------


def angle_off_boresight(elevation_rad, tilt_rad=0.0):
    """Winkel theta zwischen Sendeantennenachse (Nadir) und Empfaengerrichtung, rad.

    theta = pi/2 - epsilon - tilt

    epsilon ist der Elevationswinkel des Senders ueber dem Empfaengerhorizont
    (aus geometry.elevation_angle), tilt der Nickwinkel, um den die Antenne
    aus der Senkrechten gekippt ist (0 = senkrecht haengend, >0 = nach vorne
    geneigt bei Vorwaertsflug). theta = 0 heisst "Empfaenger liegt genau auf
    der Antennenachse" (senkrecht unter der Drohne), theta = pi/2 heisst
    "Empfaenger liegt am Horizont der Antenne" (maximaler Gewinn eines
    Rundstrahlers).

    Das Ergebnis wird nicht auf [0, pi] geclippt -- die Richtcharakteristiken
    unten sind spiegelsymmetrisch um die Achse und werten |theta| aus, ein
    negativer Winkel (z. B. sehr nahe Distanz + grosser Tilt) ist also
    physikalisch sinnvoll und muss hier nicht behandelt werden.
    """
    return np.pi / 2.0 - elevation_rad - tilt_rad


# --- Richtcharakteristiken F(theta), auf Maximum 1 normiert ----------------


def isotropic_pattern(theta):
    """Kugelstrahler: F(theta) = 1 ueberall. Referenzfall, D = 1 (0 dBi)."""
    theta = np.asarray(theta, dtype=float)
    return np.ones_like(theta)


def dipole_pattern(theta):
    """Halbwellendipol-Richtcharakteristik, auf Maximum 1 normiert.

    F(theta) = [cos(pi/2 * cos theta) / sin theta]^2

    Maximum (F=1) liegt breitseits (theta = pi/2), echte Null auf der Achse
    (theta = 0, pi) -- dort steht 0/0, was hier explizit auf 0 gesetzt wird
    statt der numerischen Instabilitaet ueberlassen zu werden. Dient als
    Verifikationsanker (D = 2,15 dBi ist ein Lehrbuchwert), nicht als Modell
    fuer die tatsaechliche VTX-Antenne.
    """
    theta = np.asarray(theta, dtype=float)
    theta_eff = np.abs(theta)
    sin_t = np.sin(theta_eff)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = (np.cos(np.pi / 2.0 * np.cos(theta_eff)) / sin_t) ** 2
    return np.where(np.isclose(sin_t, 0.0), 0.0, f)


def pagoda_pattern(theta, n=2.0, floor_db=-12.0):
    """RHCP-Pagoda-Naeherung: F(theta) = max(sin^n(theta), floor), auf Max. 1.

    sin^n(theta) hat wie der Dipol Maximum breitseits (theta = pi/2) und
    Nullstellen auf der Achse -- reale Pagoda-Antennen strahlen wegen
    Streustrahlung (Leiterbahnen, Gehaeuse, endliche Elementzahl) aber auch
    axial noch mit typischerweise ~-12 dB relativ zum Maximum ab. `floor`
    deckelt die Formel deshalb nach unten, statt eine unendlich tiefe Null
    zuzulassen. n ist ein Formparameter (n=2 als Default, entspricht in etwa
    gemessenen Pagoda-Mustern), keine physikalische Konstante.
    """
    theta = np.asarray(theta, dtype=float)
    floor_lin = 10.0 ** (floor_db / 10.0)
    return np.maximum(np.abs(np.sin(theta)) ** n, floor_lin)


# --- Direktivitaet, numerisch integriert ------------------------------------


def directivity_linear(pattern_func, **pattern_kwargs):
    """Direktivitaet D (linear) eines azimutal symmetrischen Musters F(theta).

    D = 4*pi / Integral( F(theta) * sin(theta) dtheta dphi )

    Fuer azimutal symmetrische F (kein phi-Abhaengigkeit) reduziert sich das
    Flaechenintegral ueber die Kugel auf
    D = 2 / Integral_0^pi F(theta) * sin(theta) dtheta,
    numerisch ausgewertet mit scipy.integrate.quad. F muss dafuer bereits auf
    Maximum 1 normiert sein (das ist bei allen *_pattern-Funktionen oben der
    Fall).
    """

    def integrand(theta):
        return float(pattern_func(theta, **pattern_kwargs)) * np.sin(theta)

    integral, _abserr = integrate.quad(integrand, 0.0, np.pi)
    return 2.0 / integral


def directivity_dbi(pattern_func, **pattern_kwargs):
    """Direktivitaet in dBi. Siehe directivity_linear()."""
    return lin_to_db(directivity_linear(pattern_func, **pattern_kwargs))


def gain_dbi(pattern_func, theta, **pattern_kwargs):
    """Antennengewinn in Richtung theta, dBi.

    G(theta) = D * F(theta) linear, also in dB: D_dBi + 10*log10(F(theta)).
    F(theta) = 0 (echte Null) ergibt -inf dBi, das ist korrekt und kein Fehler
    -- an einer echten Nullstelle kommt keine Leistung an.
    """
    d_dbi = directivity_dbi(pattern_func, **pattern_kwargs)
    f = np.asarray(pattern_func(theta, **pattern_kwargs), dtype=float)
    with np.errstate(divide="ignore"):
        f_db = 10.0 * np.log10(f)
    return d_dbi + f_db


# --- Polarisations-Mismatch (PLF) -------------------------------------------
#
# r = lineares Achsenverhaeltnis der Polarisationsellipse, |r| >= 1.
# r = 1: perfekt zirkular. r -> unendlich: perfekt linear. Das Vorzeichen von
# r kodiert den Drehsinn bei zirkularer/elliptischer Polarisation relativ
# zueinander (z. B. r1 = +1 fuer RHCP, r2 = -1 fuer LHCP) -- bei linearer
# Polarisation (r = inf) ist das Vorzeichen bedeutungslos, weil es keinen
# Drehsinn gibt; die Ausrichtung steckt dort allein in delta_tau.


def polarization_loss_factor(r1, r2, delta_tau_rad):
    """PLF (linear, 0..1): Leistungsanteil, der trotz Polarisations-Mismatch ankommt.

    PLF = 1/2 + 1/2 * (4*r1*r2 + (r1^2-1)*(r2^2-1)*cos(2*delta_tau))
                       / ((r1^2+1)*(r2^2+1))

    r1, r2 sind die linearen Achsenverhaeltnisse (>= 1 im Betrag) von Sende-
    und Empfangsantenne, delta_tau der Winkel zwischen ihren Hauptachsen.

    Fuer perfekt lineare Polarisation (r -> unendlich) ist die Formel direkt
    numerisch instabil: (r^2-1) -> inf, und wenn gleichzeitig beim jeweils
    anderen r^2-1 = 0 ist (reiner Zirkularfall), steht dort inf*0 = nan. Statt
    dessen wird hier der analytische Grenzwert eingesetzt (Herleitung: fuehre
    den Grenzuebergang r -> unendlich in Zaehler/Nenner getrennt fuer
    Terme gleicher Ordnung in r durch):

        r2 -> inf, r1 endlich:  PLF -> 1/2 + 1/2 * (r1^2-1)/(r1^2+1) * cos(2dt)
        r1, r2 -> inf:          PLF -> 1/2 + 1/2 * cos(2*delta_tau)

    Die zweite Zeile ist ein Spezialfall der ersten mit r1 -> inf und
    bestaetigt sich gegenseitig: beide Grenzwerte sind stetig ineinander
    ueberfuehrbar, es gibt keinen Sprung an der "unendlich"-Grenze.
    """
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    delta_tau_rad = np.asarray(delta_tau_rad, dtype=float)
    cos2dt = np.cos(2.0 * delta_tau_rad)

    inf1 = np.isinf(r1)
    inf2 = np.isinf(r2)

    def axial_term(r):
        return (r**2 - 1.0) / (r**2 + 1.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        general = 0.5 + 0.5 * (
            4.0 * r1 * r2 + (r1**2 - 1.0) * (r2**2 - 1.0) * cos2dt
        ) / ((r1**2 + 1.0) * (r2**2 + 1.0))
        limit_r2_inf = 0.5 + 0.5 * axial_term(r1) * cos2dt
        limit_r1_inf = 0.5 + 0.5 * axial_term(r2) * cos2dt
        limit_both_inf = 0.5 + 0.5 * cos2dt

    result = np.where(
        inf1 & inf2,
        limit_both_inf,
        np.where(inf2, limit_r2_inf, np.where(inf1, limit_r1_inf, general)),
    )
    return result


def polarization_mismatch_loss_db(r1, r2, delta_tau_rad):
    """Polarisations-Mismatch als positiver dB-Verlust (0 = perfekt angepasst).

    loss_dB = -10*log10(PLF). Bei perfekter Kreuzpolarisation (PLF = 0, z. B.
    RHCP gegen LHCP oder zwei um 90 Grad verdrehte lineare Antennen) ergibt
    sich +inf dB -- kein Fehler, sondern der physikalisch korrekte Totalverlust.
    """
    plf = polarization_loss_factor(r1, r2, delta_tau_rad)
    with np.errstate(divide="ignore"):
        return -10.0 * np.log10(plf)
