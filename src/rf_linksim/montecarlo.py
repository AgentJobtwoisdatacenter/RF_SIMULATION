"""Shadowing + schneller Schwund als Zufallsterme auf einer deterministischen P_rx.

Zentrale Vorgabe aus INSTRUCITONS.md: **die deterministische Kette (link.py)
wird genau einmal gerechnet**, nicht n_runs-mal. Shadowing und Fading sind
beides additive dB-Terme, die auf das EINE deterministische P_rx aufaddiert
werden -- Monte Carlo passiert komplett hier, nicht durch wiederholtes
Aufrufen von link.compute_link_budget().

Zwei physikalisch getrennte Zufallseffekte:

1. **Shadowing** -- log-normal, sigma je Umgebungsstufe (environment.STAGES).
   Langsam veraenderliche Abschattung durch Gelaende/Bebauung abseits der
   direkten Sichtlinie. Symmetrisch um 0 dB (kann die Verbindung ebenso
   verbessern wie verschlechtern).

2. **Schneller Schwund (Fading)** -- Rice-verteilt mit K-Faktor je Stufe.
   K hoch = dominanter direkter Pfad (wenig Schwund, Stufe 1), K niedrig =
   viele vergleichbar starke Mehrwegpfade (starker Schwund, Stufe 4). Auf
   E[|h|^2] = 1 normiert, sonst verschiebt der Schwund selbst den Mittelwert
   der Kette -- das waere ein Modellfehler, kein Fading-Effekt mehr.

**Median statt Mittelwert**: log-normale und Rice-Verteilungen sind schief.
Der Erwartungswert von P_rx in dBm ist NICHT dasselbe wie 10*log10(E[P_rx in
mW]) -- insbesondere zieht Fading den linearen Mittelwert von |h|^2 auf
exakt 1, aber der Mittelwert von 10*log10(|h|^2) ist negativ (Jensensche
Ungleichung). Der Median ist robust gegen diese Verzerrung und die richtige
Zentralgroesse fuer Perzentil-Aussagen.
"""

import numpy as np


def sample_shadowing_db(sigma_db, size, rng):
    """Log-normales Shadowing, dB. N(0, sigma_db^2) -- symmetrisch additiv."""
    return rng.normal(0.0, sigma_db, size=size)


def sample_rice_fading_db(k_db, size, rng):
    """Schneller Schwund (Rice-verteilt), als additiver dB-Term.

    K linear = 10^(K_dB/10). Mit
      s       = sqrt(K / (K+1))         (Amplitude des dominanten Pfads)
      sigma_f = sqrt(1 / (2*(K+1)))     (Streuung der diffusen Pfade)
      h = (s + sigma_f * N(0,1)) + j * (sigma_f * N(0,1))
    ist E[|h|^2] = s^2 + 2*sigma_f^2 = K/(K+1) + 1/(K+1) = 1 -- die
    Normierung, die INSTRUCITONS.md fordert. Rueckgabe ist 10*log10(|h|^2),
    additiv zur deterministischen Kette.
    """
    k_lin = 10.0 ** (k_db / 10.0)
    s = np.sqrt(k_lin / (k_lin + 1.0))
    sigma_f = np.sqrt(1.0 / (2.0 * (k_lin + 1.0)))
    h_real = s + sigma_f * rng.standard_normal(size)
    h_imag = sigma_f * rng.standard_normal(size)
    power_lin = h_real**2 + h_imag**2
    with np.errstate(divide="ignore"):
        return 10.0 * np.log10(power_lin)


def run_monte_carlo(
    p_rx_deterministic_dbm,
    shadow_sigma_db,
    rice_k_db,
    n_runs,
    rng=None,
    include_fading=True,
):
    """P_rx-Stichprobe: deterministisches P_rx + Shadowing (+ Fading), dBm.

    Rechnet die deterministische Kette NICHT neu -- p_rx_deterministic_dbm
    ist ein einzelner Wert (aus link.compute_link_budget), nur die beiden
    Zufallsterme werden n_runs-mal gezogen und addiert.
    """
    rng = np.random.default_rng() if rng is None else rng
    samples = p_rx_deterministic_dbm + sample_shadowing_db(shadow_sigma_db, n_runs, rng)
    if include_fading:
        samples = samples + sample_rice_fading_db(rice_k_db, n_runs, rng)
    return samples


def percentile_summary(samples, percentiles=(5.0, 50.0, 95.0)):
    """Perzentile einer Stichprobe als {perzentil: wert}-Dict. Median (50) ist
    die empfohlene Zentralgroesse, siehe Modul-Docstring."""
    values = np.percentile(samples, percentiles)
    return dict(zip(percentiles, values))
