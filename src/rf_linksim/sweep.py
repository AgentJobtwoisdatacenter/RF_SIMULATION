"""Distanz-, Hoehen- und 2D-Sweeps -- duenne Wrapper um link.compute_link_budget.

link.compute_link_budget() ist bereits voll numpy-vektorisiert: jede Formel
in geometry/antenna/pathloss/environment akzeptiert Arrays genauso wie
Skalare. Ein "Sweep" ist deshalb nichts anderes als link.compute_link_budget
mit einem Array statt eines Skalars an der Stelle, die durchlaufen werden
soll -- dieses Modul baut dafuer keine neue Rechnung, sondern nur bequeme
Konstruktoren (dataclasses.replace() auf Scenario/Transmitter), damit man
nicht jedes Mal von Hand ein neues Scenario/Transmitter-Objekt zusammenbauen
muss.

Alle Rueckgaben sind link.LinkResult- bzw. link.DualDiversityResult-Objekte,
deren Felder dann Arrays (1D fuer distance_sweep/height_sweep, 2D fuer
grid_sweep) statt Skalare sind -- exakt dieselbe Struktur wie bei einem
einzelnen Szenario, nur mit mehr Werten pro Feld.
"""

import dataclasses

import numpy as np

from rf_linksim import link


def distance_sweep(d_ground_m, scenario_template, tx, rx):
    """P_rx/C-N0 ueber ein Array von Grundentfernungen, ein Szenario/Sender/Empfaenger."""
    scenario = dataclasses.replace(scenario_template, d_ground_m=np.asarray(d_ground_m, dtype=float))
    return link.compute_link_budget(scenario, tx, rx)


def distance_sweep_all_stages(d_ground_m, scenario_template, tx, rx, stages=(1, 2, 3, 4)):
    """distance_sweep() fuer mehrere Umgebungsstufen auf einmal, {stage: LinkResult}."""
    results = {}
    for stage in stages:
        scenario = dataclasses.replace(
            scenario_template, d_ground_m=np.asarray(d_ground_m, dtype=float), environment_stage=stage
        )
        results[stage] = link.compute_link_budget(scenario, tx, rx)
    return results


def height_sweep(h_tx_m, scenario_template, tx_template, rx):
    """P_rx/C-N0 ueber ein Array von Sender-Flughoehen, feste Grundentfernung."""
    tx = dataclasses.replace(tx_template, height_m=np.asarray(h_tx_m, dtype=float))
    return link.compute_link_budget(scenario_template, tx, rx)


def grid_sweep(d_ground_m, h_tx_m, scenario_template, tx_template, rx):
    """2D-Gitter Distanz x Flughoehe, indexing='ij' (Zeile=Distanz, Spalte=Hoehe).

    Rueckgabe ist ein einzelnes link.LinkResult, dessen Felder 2D-Arrays der
    Form (len(d_ground_m), len(h_tx_m)) sind -- eine Rechnung, kein
    Doppel-Loop.
    """
    d_grid, h_grid = np.meshgrid(
        np.asarray(d_ground_m, dtype=float), np.asarray(h_tx_m, dtype=float), indexing="ij"
    )
    scenario = dataclasses.replace(scenario_template, d_ground_m=d_grid)
    tx = dataclasses.replace(tx_template, height_m=h_grid)
    return link.compute_link_budget(scenario, tx, rx)
