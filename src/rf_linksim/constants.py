"""Naturkonstanten und die dB/linear-Umrechnungen, auf denen alles andere aufbaut.

Zwei Konstanten sind hier so definiert, wie sie seit der SI-Neudefinition 2019
exakt (nicht gemessen) festliegen: die Lichtgeschwindigkeit `SPEED_OF_LIGHT`
und die Boltzmann-Konstante `BOLTZMANN`. Beides sind Definitionswerte, keine
Messwerte mit Unsicherheit.

Die Rauschtemperatur `T0_KELVIN = 290 K` ist dagegen eine *Konvention*
(IEEE Std 686 / ITU-R), keine Naturkonstante -- sie ist der Referenzwert, auf
den sich Rauschzahlen (noise figure) und thermisches Grundrauschen beziehen,
unabhaengig von der tatsaechlichen physikalischen Temperatur der Antenne oder
des Empfaengers.

`kT0_dbm_per_hz()` ist die einzige Groesse in diesem Paket, die eine
Bandbreite *im Namen* traegt (Hz), ohne eine Empfaengereigenschaft zu sein --
kT0 ist die thermische Rauschleistungsdichte des Vakuums bei T0, Physik, keine
Hardware. Deshalb ist sie hier und nicht in einem Empfaenger-Modul erlaubt
(siehe Modul-Docstring von link.py). Der Wert wird aus Boltzmann-Konstante und
T0 *berechnet*, nicht als Literal -113,98-o.ae. eingetragen, damit er bei einer
abweichenden Referenztemperatur automatisch konsistent bleibt.
"""

import numpy as np

# Lichtgeschwindigkeit im Vakuum, m/s -- exakt seit SI-Neudefinition 2019.
SPEED_OF_LIGHT = 299_792_458.0

# Boltzmann-Konstante, J/K -- exakt seit SI-Neudefinition 2019.
BOLTZMANN = 1.380_649e-23

# Referenz-Rauschtemperatur, K -- IEEE-/ITU-R-Konvention, keine Messgroesse.
T0_KELVIN = 290.0


def kt0_dbm_per_hz(t_kelvin: float = T0_KELVIN) -> float:
    """Thermische Rauschleistungsdichte kT bei gegebener Temperatur, dBm/Hz.

    P = k * T ist eine Leistung pro Hertz Bandbreite (W/Hz). Umgerechnet in
    dBm/Hz bei T0 = 290 K ergibt das kT0 = -173,98 dBm/Hz -- die Konstante,
    die C/N0 = P_rx - kT0 definiert.
    """
    p_watt_per_hz = BOLTZMANN * t_kelvin
    return watt_to_dbm(p_watt_per_hz)


def wavelength(frequency_hz):
    """Freiraum-Wellenlaenge lambda = c / f, m. Nimmt auch numpy-Arrays."""
    return SPEED_OF_LIGHT / frequency_hz


# --- dB/linear-Umrechnungen -------------------------------------------------
#
# Konvention im ganzen Paket: Leistungen (nicht Feldgroessen) werden in dB
# umgerechnet, also immer 10*log10(...), nie 20*log10(...) -- auch fuer
# Groessen wie den Polarisations-Mismatch, die man aus einem Feldverhaeltnis
# herleitet. Wo ein Feldverhaeltnis (Spannung, E-Feld) gemeint ist, steht das
# im jeweiligen Modul-Docstring explizit dabei.


def db_to_lin(x_db):
    """dB (Leistungsverhaeltnis) -> linear. 10**(x/10)."""
    return 10.0 ** (x_db / 10.0)


def lin_to_db(x_lin):
    """Linear (Leistungsverhaeltnis) -> dB. 10*log10(x)."""
    return 10.0 * np.log10(x_lin)


def dbm_to_watt(p_dbm):
    """dBm -> Watt."""
    return 10.0 ** ((p_dbm - 30.0) / 10.0)


def watt_to_dbm(p_watt):
    """Watt -> dBm."""
    return 10.0 * np.log10(p_watt) + 30.0
