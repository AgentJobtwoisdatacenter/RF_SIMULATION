"""Schritt 2: C/N0 aus link.py + SDR-Kenndaten -> zwei Detektions-Szenarien.

Das hier ist bewusst ein NEUES Modul (nicht Teil von rf_linksim/Schritt 1) --
genau der Bruch, den INSTRUCITONS.md vorgesehen hat: "link.py gibt C/N0
zurueck und weiss nichts von Empfaengern... Schritt 2 wird spaeter ein
Modul receiver.py ergaenzen, das C/N0 plus SDR-Kenndaten entgegennimmt."

Zwei Szenarien, weil "detektierbar" zwei verschiedene Dinge heissen kann:

1. **Demodulation**: das Rohsignal-SNR in der vollen Signalbandbreite (27
   MHz) muss fuer eine Videodemodulation reichen. Einzelmessung, kein
   Integrationsgewinn.
2. **Detektion**: es reicht festzustellen "hier sendet etwas" -- durch
   nicht-kohaerente Integration ueber mehrere unabhaengige Zeit-Looks (z. B.
   aufeinanderfolgende Spektrum-Sweeps) laesst sich die Detektionsschwelle
   weit unter das Einzelschuss-SNR druecken. Formalisiert ueber Albersheims
   Gleichung (Radar-Detektionstheorie).

Kennwerte des bladeRF 2.0 micro xA9 (AD9361), aus dem ADI-AD9361-Datenblatt
Rev. G (vom Nutzer als PDF bereitgestellt, direkt aus Table 1, Seite 4/6
gelesen -- keine Schaetzung):
  - Rauschzahl bei Max-Gain, 5,5 GHz: 3,8 dB (RECEIVERS, 5.5 GHz)
  - Max. RX-Gain bei 5500 MHz (RX1A/RX2A): 65,5 dB; Min-Gain: 0 dB;
    Gain-Step: 1 dB (RECEIVERS, GENERAL)
  - ADC: 12 Bit (AUXILIARY CONVERTERS)
  - RSSI-Bereich: 100 dB, Genauigkeit +/-2 dB

**ANNAHME, klar markiert**: Eine explizite NF-vs-Gain-Index-Kurve oder ein
Vollausschlag-Eingangspegel (P1dB) steht im Datenblatt NICHT als Zahl,
nur als eingebettete Grafik (z. B. "RX Noise Figure vs. Interferer Power
Level... Gain Index = 64", nur EIN Gain-Punkt, keine Kurve) bzw. gar nicht
explizit tabelliert. `effective_noise_figure_db()` nutzt deshalb eine
einfache, konservative Naeherung (NF verschlechtert sich 1:1 mit der
AGC-Gain-Reduktion), aber mit dem ECHTEN Gain-Bereich (0..65,5 dB bei
5,5 GHz) als physikalische Grenze. Mit den echten NF-Stuetzpunkten (falls
irgendwo als Zahlentabelle auffindbar) laesst sich die Steigung direkt
ersetzen.
"""

import numpy as np

# --- AGC-Naeherung: effektive Rauschzahl bei reduziertem Gain --------------


MAX_GAIN_DB_5500MHZ = 65.5  # Table 1 (Rev. G), RX1A/RX2A bei 5500 MHz -- echter Datenblattwert
MIN_GAIN_DB = 0.0  # Table 1 (Rev. G) -- echter Datenblattwert


def effective_noise_figure_db(
    p_rx_dbm,
    nf_max_gain_db=3.8,
    agc_threshold_dbm=-20.0,
    gain_reduction_slope=1.0,
    max_gain_reduction_db=MAX_GAIN_DB_5500MHZ - MIN_GAIN_DB,
):
    """Effektive Rauschzahl unter AGC-Gain-Reduktion, dB.

    Solange P_rx unter agc_threshold_dbm bleibt, laeuft der AD9361 mit
    Max-Gain und nf_max_gain_db gilt unveraendert. Darueber muss die AGC den
    Frontend-Gain zurueckregeln, um den ADC (12 Bit) nicht zu uebersteuern.

    ANNAHME (nicht aus dem Datenblatt ablesbar, siehe Modul-Docstring): die
    effektive Rauschzahl verschlechtert sich naeherungsweise linear mit der
    Gain-Reduktion (gain_reduction_slope = 1,0 dB NF pro dB Gain-Rueckregelung
    als konservativer Ansatz). ECHT aus dem Datenblatt (Table 1, Rev. G) ist
    dagegen die physikalische Grenze: die Gain-Reduktion kann nicht groesser
    sein als der tatsaechliche Gain-Bereich bei 5,5 GHz (65,5 dB Max- minus
    0 dB Min-Gain = 65,5 dB) -- darueber hinaus gibt es keinen Gain mehr zum
    Zurueckregeln, das Signal wuerde den ADC unabhaengig von diesem Modell
    uebersteuern.

    agc_threshold_dbm = -20 dBm ist selbst eine ANNAHME (kein
    Vollausschlag-/P1dB-Wert im Datenblatt gefunden, siehe Modul-Docstring)
    fuer den Punkt, an dem die AGC ueberhaupt anfaengt, Gain zurueckzunehmen.
    """
    gain_reduction_db = np.clip(p_rx_dbm - agc_threshold_dbm, 0.0, max_gain_reduction_db)
    return nf_max_gain_db + gain_reduction_slope * gain_reduction_db


# --- Szenario 1: Demodulation (Einzelschuss-SNR, kein Integrationsgewinn) --


def demodulation_snr_db(cn0_db_hz, bandwidth_hz, p_rx_dbm, nf_max_gain_db=3.8, agc_threshold_dbm=-20.0):
    """SNR fuer Video-Demodulation: volle Bandbreite, ein einzelner Messwert.

    SNR = C/N0 - 10*log10(B) - NF_effektiv(P_rx)

    Referenzschwelle (NICHT ausgewertet, nur zur Einordnung): analoge
    FM-Videolinks zeigen typischerweise ab ~8-10 dB Traeger-Rausch-Verhaeltnis
    einen "FM-Schwelleneffekt" (zunehmendes Knistern/Bildstoerungen unterhalb
    dieses Bereichs) -- ein allgemeiner Richtwert aus der analogen FM-
    Uebertragungstheorie, keine Kenngroesse des konkreten hier verwendeten
    VTX-Standards.
    """
    nf_eff = effective_noise_figure_db(p_rx_dbm, nf_max_gain_db, agc_threshold_dbm)
    return cn0_db_hz - 10.0 * np.log10(bandwidth_hz) - nf_eff


# --- Szenario 2: Detektion (nicht-kohaerente Integration, Albersheim) ------


def albersheim_required_snr_db(pd, pfa, n_pulses):
    """Erforderliches Einzelschuss-SNR (dB) fuer nicht-kohaerente Integration
    von n_pulses unabhaengigen Looks, gegebene Detektionswahrscheinlichkeit
    Pd und Falschalarmrate Pfa.

    Albersheims Gleichung (Albersheim, W. J. (1981), "A closed-form
    approximation to Robertson's detection characteristics," Proc. IEEE,
    69(7), 839-840) -- Standardnaeherung der Radar-Detektionstheorie,
    gueltig fuer 0,1 <= Pd <= 0,9, 1e-7 <= Pfa <= 1e-3, 1 <= n_pulses <= 8096:

        A = ln(0,62 / Pfa)
        B = ln(Pd / (1 - Pd))
        SNR_dB = -5*log10(n) + (6,2 + 4,54/sqrt(n+0,44)) * log10(A + 0,12*A*B + 1,7*B)

    Verifiziert gegen den Standard-Lehrbuchfall Pd=0,9/Pfa=1e-6/n=1 ->
    ~13,1 dB (z. B. Richards, *Fundamentals of Radar Signal Processing*).
    """
    a = np.log(0.62 / pfa)
    b = np.log(pd / (1.0 - pd))
    return -5.0 * np.log10(n_pulses) + (6.2 + 4.54 / np.sqrt(n_pulses + 0.44)) * np.log10(
        a + 0.12 * a * b + 1.7 * b
    )


def detection_margin_db(
    cn0_db_hz,
    signal_bandwidth_hz,
    p_rx_dbm,
    n_looks,
    pd=0.9,
    pfa=1e-6,
    nf_max_gain_db=3.8,
    agc_threshold_dbm=-20.0,
):
    """(Einzelschuss-SNR, erforderliches SNR, Marge) fuer die Detektions-Szenario.

    Das Einzelschuss-SNR ist dieselbe Groesse wie in demodulation_snr_db()
    (volle Signalbandbreite als eine Messung) -- das ist kein Zufall: fuer
    ein Signal mit naeherungsweise flacher spektraler Leistungsdichte ist
    das SNR EINES einzelnen schmalbandigen FFT-Bins unabhaengig von der
    gewaehlten Aufloesungsbandbreite identisch mit dem Vollband-SNR (Signal-
    UND Rauschanteil im Bin skalieren gleich mit der Binbreite, kuerzen sich
    also). Verbessert wird die Detektion deshalb nicht durch schmalere Bins,
    sondern durch mehrere unabhaengige ZEIT-Looks (aufeinanderfolgende
    Sweeps), die nicht-kohaerent gemittelt werden -- genau das, was
    n_looks hier zaehlt.
    """
    single_look_snr_db = demodulation_snr_db(
        cn0_db_hz, signal_bandwidth_hz, p_rx_dbm, nf_max_gain_db, agc_threshold_dbm
    )
    required_snr_db = albersheim_required_snr_db(pd, pfa, n_looks)
    margin_db = single_look_snr_db - required_snr_db
    return single_look_snr_db, required_snr_db, margin_db
