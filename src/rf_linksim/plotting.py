"""Standardplots: Distanz-Sweeps je Umgebungsstufe, 2D-Gitter als Heatmap.

Reine Darstellung -- keine neue Physik. Nimmt die Ergebnisse aus sweep.py/
link.py entgegen und zeichnet sie. Referenzschwellen aus der Literatur
duerfen hier als Linien auftauchen (INSTRUCITONS.md), das Modul selbst
wertet nichts aus.

Farbwahl folgt einer festen, kolorblind-sicheren Kategorial-Palette (eine
Farbe pro Umgebungsstufe, IMMER in derselben Reihenfolge Stufe 1-4 -- Farbe
folgt der Identitaet der Stufe, nicht ihrem Rang in einem bestimmten Plot).
"""

import matplotlib

matplotlib.use("Agg")  # headless: dieses Paket erzeugt Dateien, kein interaktives Fenster
import matplotlib.pyplot as plt

from rf_linksim import environment

# Kategoriale Palette, Stufe 1-4 in fester Reihenfolge (kolorblind-validiert,
# siehe Konversation/dataviz-Skill: worst-case CVD-Delta 9,1 im hellen Modus).
STAGE_COLORS = {
    1: "#2a78d6",  # blau
    2: "#eb6834",  # orange
    3: "#1baf7a",  # aqua
    4: "#eda100",  # gelb
}

_QUANTITY_LABELS = {
    "p_rx_dbm": "P_rx [dBm]",
    "cn0_db_hz": "C/N0 [dB-Hz]",
}


def _style_axes(ax):
    """Zuruckhaltende Gitterlinien/Achsen, wie im dataviz-Skill spezifiziert."""
    ax.grid(True, linewidth=1.0, color="#d8d7d2", zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#8a8980")


def plot_distance_sweep_by_stage(
    d_ground_m,
    results_by_stage,
    quantity="cn0_db_hz",
    title=None,
    threshold_lines=None,
    ax=None,
):
    """Sweep-Ergebnis (dict {stage: LinkResult}) ueber der Distanz, eine Linie je Stufe.

    threshold_lines: optionale Liste von (label, wert)-Paaren, die als
    horizontale Referenzlinien eingezeichnet werden (z. B. Literatur-
    Schwellen) -- rein informativ, keine Bewertung.
    """
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(8, 5.5))
    ylabel = _QUANTITY_LABELS.get(quantity, quantity)

    for stage in sorted(results_by_stage.keys()):
        result = results_by_stage[stage]
        y = getattr(result, quantity)
        stage_def = environment.STAGES[stage]
        ax.plot(
            d_ground_m,
            y,
            color=STAGE_COLORS[stage],
            linewidth=2.0,
            marker="o",
            markersize=6.0,
            markeredgewidth=1.5,
            markeredgecolor="white",
            label=f"Stufe {stage}: {stage_def.name}",
            zorder=3,
        )

    if threshold_lines:
        for label, value in threshold_lines:
            ax.axhline(value, color="#8a8980", linewidth=1.0, linestyle="--", zorder=2)
            ax.text(
                d_ground_m[-1], value, f"  {label}", color="#52514e", fontsize=8,
                va="center", ha="left",
            )

    ax.set_xlabel("Grundentfernung [m]")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def plot_grid_heatmap(d_ground_m, h_tx_m, grid_result, quantity="p_rx_dbm", title=None, ax=None):
    """2D-Gitter (Distanz x Flughoehe) als Heatmap, sequentielle Ein-Farb-Rampe.

    grid_result ist das Ergebnis von sweep.grid_sweep() -- Felder haben Form
    (len(d_ground_m), len(h_tx_m)).
    """
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(8, 5.5))
    z = getattr(grid_result, quantity)
    label = _QUANTITY_LABELS.get(quantity, quantity)

    im = ax.pcolormesh(
        h_tx_m, d_ground_m, z, shading="nearest", cmap="Blues_r" if quantity == "p_rx_dbm" else "Blues"
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(label)
    cbar.outline.set_visible(False)

    ax.set_xlabel("Flughoehe TX [m]")
    ax.set_ylabel("Grundentfernung [m]")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def save_figure(fig, path, dpi=150):
    """Bequemlichkeitsfunktion: Figure speichern und schliessen."""
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path
