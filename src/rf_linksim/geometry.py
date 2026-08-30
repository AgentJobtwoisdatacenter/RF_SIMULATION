"""Reine Streckengeometrie: Schraegentfernung, Elevation, Fresnelzone, Radiohorizont.

Dieses Modul kennt keine Leistungen, keine Antennenmuster und keine
Umgebungsstufen -- nur Positionen (Hoehen, Grundentfernung) und die Winkel
und Laengen, die sich rein geometrisch daraus ergeben. Alles hier ist exakte
euklidische/sphaerische Geometrie, keine Naeherung mit Modellunsicherheit.

Konvention: Winkel werden intern im Bogenmass gefuehrt (numpy-Konvention).
Grad-Varianten (mit `_deg`-Suffix) existieren nur dort, wo eine Formel aus
der Literatur explizit in Grad definiert ist (z. B. Al-Hourani in
environment.py), damit dort kein stiller Umrechnungsfehler entsteht.
"""

import numpy as np


def slant_range(d_ground, h_tx, h_rx):
    """Schraegentfernung Sender-Empfaenger, m.

    d_slant = sqrt(d_ground^2 + (h_tx - h_rx)^2). Das ist die Groesse, die in
    die Freiraumdaempfung gehoert -- nicht die Grundentfernung. Bei flachen
    Geometrien (grosse d_ground, kleine Hoehendifferenz) ist der Unterschied
    klein, bei kurzen/steilen Strecken aber signifikant (z. B. 100 m Flughoehe
    auf 200 m Grundentfernung: 12 % laenger als d_ground, das sind 1,0 dB
    zusaetzliche Freiraumdaempfung).
    """
    return np.sqrt(d_ground**2 + (h_tx - h_rx) ** 2)


def elevation_angle(d_ground, h_tx, h_rx):
    """Elevationswinkel des Senders ueber dem Horizont des Empfaengers, rad.

    epsilon = atan2(h_tx - h_rx, d_ground), positiv wenn der Sender hoeher
    steht als der Empfaenger. atan2 statt atan, damit auch h_tx < h_rx und
    d_ground = 0 sauber behandelt werden.
    """
    return np.arctan2(h_tx - h_rx, d_ground)


def elevation_angle_deg(d_ground, h_tx, h_rx):
    """Wie elevation_angle, aber in Grad -- fuer Formeln, die epsilon_deg
    direkt in Grad erwarten (z. B. das Al-Hourani-Clutter-Modell)."""
    return np.degrees(elevation_angle(d_ground, h_tx, h_rx))


def fresnel_zone_radius(d1, d2, wavelength_m, n=1):
    """Radius der n-ten Fresnelzone an einem Punkt zwischen den Enden, m.

    r_n = sqrt(n * lambda * d1 * d2 / (d1 + d2)), mit d1, d2 den Abstaenden
    des betrachteten Punkts von den beiden Streckenenden (d1 + d2 = Gesamt-
    strecke). Die erste Fresnelzone (n=1) begrenzt den Bereich, der bei freier
    Sicht ("Freiraumfall") tatsaechlich frei von Hindernissen sein muss --
    nicht nur die direkte Sichtlinie selbst.
    """
    return np.sqrt(n * wavelength_m * d1 * d2 / (d1 + d2))


def fresnel_zone_radius_midpoint(d_total, wavelength_m, n=1):
    """Fresnelzonenradius am Streckenmittelpunkt (d1 = d2 = d_total / 2), m.

    Der Mittelpunkt ist meist der kritische Punkt: dort ist die Fresnelzone
    am breitesten und liegt bei einer geneigten Sichtlinie (Sender hoch,
    Empfaenger niedrig) am naechsten am Boden.
    """
    return fresnel_zone_radius(d_total / 2.0, d_total / 2.0, wavelength_m, n)


def radio_horizon_km(height_m, k_factor=4.0 / 3.0, earth_radius_m=6_371_000.0):
    """Geometrischer Radiohorizont einer einzelnen Antennenhoehe, km.

    d = sqrt(2 * k * R_e * h) -- Tangente an eine Kugel mit effektivem Radius
    k*R_e (Standardnaeherung fuer troposphaerische Strahlkruemmung bei
    Standardatmosphaere, k = 4/3). Mit R_e = 6371 km ergibt das den bekannten
    Vorfaktor d[km] = 4,12 * sqrt(h[m]).

    Der Horizont zweier Stationen addiert sich: die Gesamt-Sichtweite zwischen
    Sender und Empfaenger ist radio_horizon_km(h_tx) + radio_horizon_km(h_rx),
    nicht der Wert einer einzelnen Hoehe allein.
    """
    return np.sqrt(2.0 * k_factor * earth_radius_m * height_m) / 1000.0
