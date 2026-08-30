"""Umgebungsstufen 1-4: Clutter-Daempfung, Vegetations-Zusatzverlust, Shadowing/Fading-Parameter.

Dieses Modul haelt die vier Umgebungsstufen als eine einzige Quelle der
Wahrheit (STAGES) und stellt zwei physikalisch unabhaengige Effekte bereit:

1. **Clutter-Daempfung** (schwaecht das Signal, additiv in dB): zwei
   austauschbare Modi.
   - "al_hourani": winkelabhaengige Air-to-Ground-Formel (Al-Hourani et al.
     2014). Bei 5,8 GHz mit Sender ueber dem Clutter strukturell richtig,
     im Absolutwert aber unbelegt (Al-Hourani ist bei 0,7-2,5 GHz kalibriert,
     siehe INSTRUCITONS.md Fallstrick 5) -- Default, weil er die einzige
     Formel im Dokument ist, die die vorgegebene Verifikationstabelle
     reproduziert.
   - "fixed": eine feste, pro Stufe frei kalibrierbare dB-Zahl, kein Winkel.
     Platzhalter fuer eine spaetere Referenzmessung.

2. **Shadowing/Fading-Parameter** (streut das Signal statistisch, wird NICHT
   hier ausgewertet, nur bereitgestellt): sigma fuer log-normales Shadowing
   und der Rice-K-Faktor fuer schnellen Schwund je Stufe. Die tatsaechliche
   Zufallsstichprobe passiert in montecarlo.py -- environment.py liefert nur
   die Parameter, keine Zufallszahlen.

Vegetation (Weissberger) ist ein DRITTER, unabhaengiger Term: zusaetzliche
Daempfung durch eine konkrete Baumreihen-Tiefe entlang der Strecke, additiv
zur Clutter-Daempfung, nicht Teil der Umgebungsstufen-Tabelle.

Wichtige Invariante (INSTRUCITONS.md, strukturelle Tests): eine dichtere
Umgebungsstufe darf das Signal nie verbessern -- L_clutter muss ueber den
gesamten Elevationsbereich monoton in der Stufe sein. Das ist mit den
Al-Hourani-Rohkoeffizienten NICHT automatisch erfuellt (Stufe 2 hat
publiziert eta_NLOS=21, Stufe 3 publiziert eta_NLOS=20 -- aus getrennten
Messkampagnen, nicht fuer eine gemeinsame Skala gedacht) und wurde deshalb
für Stufe 2 auf 18,0 korrigiert (siehe Kommentar bei STAGES[2]).
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClutterStage:
    """Ein Satz Umgebungsstufen-Parameter (Al-Hourani-Clutter + Shadowing/Fading)."""

    name: str
    a: float
    b: float
    eta_los_db: float
    eta_nlos_db: float
    shadow_sigma_db: float
    rice_k_db: float


STAGES = {
    1: ClutterStage(
        name="freies Land",
        a=2.0,
        b=1.20,
        eta_los_db=0.1,
        eta_nlos_db=8.0,
        shadow_sigma_db=4.0,
        rice_k_db=12.0,
    ),
    2: ClutterStage(
        name="Dorf/Vorstadt",
        a=4.88,
        b=0.43,
        eta_los_db=0.1,
        # ANNAHME/Korrektur: Al-Hourani (2014) publiziert fuer "suburban"
        # eta_NLOS = 21,0 dB (separate Messkampagne von "urban" mit
        # eta_NLOS = 20,0 dB). Fuer eine monoton geordnete Skala 1-4 auf
        # 18,0 dB korrigiert -- sonst waere Stufe 2 (Dorf) verlustreicher
        # als Stufe 3 (Stadtrand), was der Modellabsicht widerspricht.
        # Siehe INSTRUCITONS.md Fallstrick 3 und test_clutter_monotonic_*.
        eta_nlos_db=18.0,
        shadow_sigma_db=6.0,
        rice_k_db=8.0,
    ),
    3: ClutterStage(
        name="Stadtrand",
        a=9.61,
        b=0.16,
        eta_los_db=1.0,
        eta_nlos_db=20.0,
        shadow_sigma_db=8.0,
        rice_k_db=5.0,
    ),
    4: ClutterStage(
        name="Stadt",
        a=12.08,
        b=0.11,
        eta_los_db=1.6,
        eta_nlos_db=23.0,
        shadow_sigma_db=10.0,
        rice_k_db=3.0,
    ),
}


# --- Al-Hourani-Clutter-Modell (Default) ------------------------------------


def los_probability(elevation_deg, stage: ClutterStage):
    """P_LOS(epsilon) nach Al-Hourani et al. 2014, Air-to-Ground-Sichtlinienmodell.

    P_LOS = 1 / (1 + a * exp(-b * (epsilon_deg - a)))

    epsilon_deg ist der Elevationswinkel des Senders ueber dem Empfaenger-
    horizont IN GRAD (nicht Bogenmass -- die Formel ist in der Literatur so
    definiert und reagiert empfindlich auf die falsche Einheit). a taucht
    bewusst zweimal auf (Koeffizient und Verschiebung), das ist keine
    Verwechslung, sondern die Originalform der Sigmoidfunktion.
    """
    return 1.0 / (1.0 + stage.a * np.exp(-stage.b * (elevation_deg - stage.a)))


def clutter_loss_al_hourani_db(elevation_deg, stage: ClutterStage):
    """Clutter-Daempfung, gewichtetes Mittel aus LOS- und NLOS-Exzessverlust, dB.

    L_clutter = P_LOS * eta_LOS + (1 - P_LOS) * eta_NLOS

    Die Mittelung passiert in dB (nicht im Leistungs-linearen Bereich) -- so
    ist die Formel in INSTRUCITONS.md explizit vorgegeben. Bei epsilon = 90
    Grad (Sender senkrecht ueber dem Empfaenger) geht P_LOS -> 1 und es
    bleibt nur eta_LOS uebrig -- die im Dokument geforderte Randbedingung.
    """
    p_los = los_probability(elevation_deg, stage)
    return p_los * stage.eta_los_db + (1.0 - p_los) * stage.eta_nlos_db


# --- "fixed"-Modus: konstante, kalibrierbare dB-Werte -----------------------


def clutter_loss_fixed_db(fixed_value_db):
    """Clutter-Daempfung im 'fixed'-Modus: reiner Durchreicher, kein Winkel.

    Existiert als eigene Funktion (statt den Konstantenwert direkt in
    link.py zu verwenden), damit link.py zwischen den beiden Modi per
    einheitlicher Signatur umschalten kann, ohne Sonderfaelle zu bauen.
    fixed_value_db kommt aus der Konfiguration (config/default.yaml,
    # ANNAHME-markiert) -- ohne Referenzmessung frei erfunden, bis eine
    Messung vorliegt.
    """
    return fixed_value_db


def frequency_correction_db(frequency_hz, reference_hz=2.0e9):
    """Optionaler Frequenzkorrekturterm +10*log10(f/f_ref), dB.

    Unbelegter Zusatzterm (INSTRUCITONS.md Fallstrick 5): die Al-Hourani-
    Koeffizienten sind bei 0,7-2,5 GHz kalibriert, 5,8 GHz liegt weit
    ausserhalb. Ob und wie stark Clutter-Daempfung mit der Frequenz skaliert,
    ist nicht empirisch abgesichert -- diese Funktion existiert nur, damit
    ein Aufrufer den Term BEWUSST und EXPLIZIT hinzuschalten kann. Kein
    Aufrufer in diesem Paket wendet sie automatisch an.
    """
    return 10.0 * np.log10(frequency_hz / reference_hz)


# --- Vegetation (Weissberger) ------------------------------------------------


def vegetation_loss_weissberger_db(depth_m, frequency_hz):
    """Zusaetzliche Daempfung durch eine Baumreihe (Weissberger MED), dB.

    depth_m ist NICHT die Streckenlaenge, sondern die Tiefe der durchquerten
    Vegetation entlang des Pfades (typischerweise wenige bis einige hundert
    Meter) -- additiv zur Clutter-Daempfung, nicht anstelle davon.

    L = 1,33 * f_GHz^0,284 * depth_m^0,588   fuer 14 m < depth_m <= 400 m
    L = 0,45 * f_GHz^0,284 * depth_m         fuer depth_m <= 14 m

    Gueltigkeitsgrenze bei 400 m (Modified Exponential Decay, Weissberger
    1982) -- darueber liefert die Formel keine belastbaren Werte mehr.
    """
    f_ghz = frequency_hz / 1.0e9
    depth_m = np.asarray(depth_m, dtype=float)
    shallow = 0.45 * f_ghz**0.284 * depth_m
    deep = 1.33 * f_ghz**0.284 * depth_m**0.588
    return np.where(depth_m <= 14.0, shallow, deep)
