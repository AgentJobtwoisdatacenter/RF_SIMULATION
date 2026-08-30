"""Die Bruecke zu Schritt 2 -- drei reine Ableitungen aus C/N0, noch ohne Hardware.

Genau die Groessen, die INSTRUCITONS.md fuer diesen Modul verlangt, nicht
mehr: Schritt 1 endet bei C/N0, und alles hier bleibt eine *Ableitung* aus
C/N0, keine Bewertung eines konkreten Empfaengers. Wenn `bandwidth_hz` oder
`noise_figure_db` auftauchen, ist das kein Bruch der Schritt-1/2-Grenze --
es ist genau der Punkt, an dem Schritt 2 anfaengt, und dieses Modul ist die
Nahtstelle dazu.

Die Kernzeile aus der Dokument-Einleitung, hier als Formel:

    SNR(B, NF) = C/N0 - 10*log10(B) - NF

`snr_db()` ist diese Zeile direkt; `max_noise_figure_db()` ist dieselbe
Formel nach NF aufgeloest.
"""

import numpy as np


def snr_db(cn0_db_hz, bandwidth_hz, noise_figure_db):
    """SNR eines Empfaengers mit Bandbreite B und Rauschzahl NF, dB.

    SNR = C/N0 - 10*log10(B) - NF
    """
    return cn0_db_hz - 10.0 * np.log10(bandwidth_hz) - noise_figure_db


def max_noise_figure_db(cn0_db_hz, bandwidth_hz, snr_target_db):
    """Maximal zulaessige Rauschzahl fuer ein Ziel-SNR bei gegebener Bandbreite, dB.

    NF_max = C/N0 - 10*log10(B) - SNR_ziel

    Als Tabelle ueber mehrere Bandbreiten und Ziel-SNRs aufgerufen (siehe
    max_noise_figure_table), OHNE eine Kombination auszuwaehlen -- das ist
    die Spezifikation, gegen die spaeter SDRs geprueft werden, keine
    Empfehlung.
    """
    return cn0_db_hz - 10.0 * np.log10(bandwidth_hz) - snr_target_db


def max_noise_figure_table(cn0_db_hz, bandwidths_hz, snr_targets_db):
    """NF_max als 2D-Tabelle (Bandbreiten x Ziel-SNRs), dB.

    Rueckgabe hat Form (len(bandwidths_hz), len(snr_targets_db)) --
    Zeile = Bandbreite, Spalte = Ziel-SNR.
    """
    bw = np.asarray(bandwidths_hz, dtype=float)[:, None]
    snr = np.asarray(snr_targets_db, dtype=float)[None, :]
    return cn0_db_hz - 10.0 * np.log10(bw) - snr


def required_dynamic_range_db(p_rx_dbm_values):
    """Spanne zwischen staerkstem und schwaechstem Szenario im Sweep-Raum, dB.

    max(P_rx) - min(P_rx) ueber ein beliebig geformtes Array von P_rx-Werten
    (typischerweise ein voller Sweep ueber Distanz x Stufe x Hoehe). Das ist
    die Anforderung an den nutzbaren Eingangsleistungsbereich des Empfaengers.
    """
    p_rx_dbm_values = np.asarray(p_rx_dbm_values, dtype=float)
    return np.max(p_rx_dbm_values) - np.min(p_rx_dbm_values)


def required_dynamic_range_bits(p_rx_dbm_values):
    """Wie required_dynamic_range_db(), umgerechnet in ADC-Bits (dB / 6,02).

    6,02 dB pro Bit ist der Standard-ADC-Umrechnungsfaktor (20*log10(2)) --
    der Grund, warum die Wahl zwischen 8 und 12 Bit keine Geschmacksfrage
    ist, sondern direkt aus der Streckenphysik folgt.
    """
    return required_dynamic_range_db(p_rx_dbm_values) / 6.02


def spectral_power_density_dbm_per_hz(p_rx_dbm, signal_bandwidth_hz):
    """Spektrale Leistungsdichte PSD = P_rx - 10*log10(B_signal), dBm/Hz.

    ACHTUNG (INSTRUCITONS.md Fallstrick 2): B_signal ist die Bandbreite des
    SENDESIGNALS (27 MHz im Basisszenario), NICHT eine Empfaengerbandbreite
    -- das ist die einzige Stelle im ganzen Paket, an der eine Bandbreite
    auftaucht, ohne einen Empfaenger zu beschreiben. P_rx ist die gesamte
    einfallende Leistung; PSD ist die Dichte. Wer die beiden verwechselt,
    liegt bei 27 MHz um 10*log10(27e6) = 74,3 dB daneben.

    Mit PSD laesst sich fuer jede beliebige Empfaenger-Aufloesungsbandbreite
    B_res ausrechnen, wieviel Signal in einem Bin landet:
    P_bin = PSD + 10*log10(B_res).
    """
    return p_rx_dbm - 10.0 * np.log10(signal_bandwidth_hz)
