"""rf_linksim -- hardwareunabhaengige RF-Ausbreitungssimulation.

Rechnet, wie stark das Signal eines analogen 5,8-GHz-FPV-Videodownlinks an
der Empfangsantennenklemme ankommt (P_rx) und wie viel Traegerleistung pro
Rauschleistungsdichte das ergibt (C/N0). Das ist Schritt 1 eines
zweistufigen Vorhabens und bleibt bewusst hardwareunabhaengig: keine
Empfaengerbandbreite, keine Rauschzahl, kein ADC, kein SDR-Modell. Siehe
docs/physik.md fuer Herleitung und Quellen aller Formeln.
"""

__version__ = "0.1.0"
