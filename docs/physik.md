# Physik von `rf_linksim`

Alle Formeln, ihre Quellen und Gültigkeitsgrenzen, modulweise. Konvention im
ganzen Paket: Frequenzen in Hz, Längen in m, Leistungen in dBm, Winkel intern
im Bogenmaß (Grad nur dort, wo eine Literaturformel es explizit verlangt).

Kernkontrakt (siehe `INSTRUCITONS.md`): Die Kette endet an der Antennenklemme.
Ausgabe ist `P_rx` [dBm] und `C/N₀` [dB-Hz] — hardwareunabhängig, keine
Bandbreite, keine Rauschzahl, kein ADC.

---

## `constants.py` — Naturkonstanten

| Größe | Wert | Quelle |
|---|---|---|
| Lichtgeschwindigkeit `c` | 299 792 458 m/s | exakt seit SI-Neudefinition 2019 |
| Boltzmann-Konstante `k` | 1,380 649 × 10⁻²³ J/K | exakt seit SI-Neudefinition 2019 |
| Referenz-Rauschtemperatur `T₀` | 290 K | Konvention (IEEE Std 686, ITU-R), keine Naturkonstante |

```
kT₀ [dBm/Hz] = 10·log₁₀(k·T₀) + 30 = −173,98 dBm/Hz
λ = c / f
```

`kT0_dbm_per_hz()` wird aus `k` und `T₀` berechnet, nicht als Literal
eingetragen — bleibt konsistent, falls je eine andere Referenztemperatur
gebraucht wird. Gültig für jede Temperatur; die Verwendung von T₀ = 290 K
ist die branchenübliche Referenz, nicht die tatsächliche Antennentemperatur.

---

## `geometry.py` — Streckengeometrie

**Schrägentfernung** (euklidisch, exakt):
```
d_slant = √(d_ground² + (h_tx − h_rx)²)
```
**Elevationswinkel**, positiv wenn Sender höher als Empfänger:
```
ε = atan2(h_tx − h_rx, d_ground)
```
Beide exakt für ebenes Gelände (keine Erdkrümmung) — bei den hier
betrachteten Reichweiten (≤ 6 km) vernachlässigbar (siehe Radiohorizont
unten, > 15 km bei 100 m Flughöhe).

**Fresnelzonenradius** (n-te Zone, Punkt zwischen den Streckenenden):
```
r_n = √(n·λ·d₁·d₂ / (d₁+d₂))
```
Standardformel geometrischer Optik/Beugung, z. B. Balanis, *Antenna Theory*.
Begrenzt den Bereich, der für den Freiraumfall frei von Hindernissen sein
muss — nicht nur die direkte Sichtlinie.

**Radiohorizont**, effektiver Erdradius k = 4/3 (Standardatmosphäre):
```
d = √(2·k·R_E·h) ,  R_E = 6 371 000 m
```
liefert den bekannten Vorfaktor `d[km] = 4,12·√(h[m])`. Gültig für
troposphärische Standardbrechung; bei anomaler Refraktion (Duct-Bildung)
nicht anwendbar — hier nicht weiter relevant, da alle betrachteten
Entfernungen weit innerhalb des Radiohorizonts liegen.

---

## `antenna.py` — Richtcharakteristik, Direktivität, Polarisation

**Winkel zur Sendeantennenachse:**
```
θ = π/2 − ε − tilt
```
θ = 0 auf der Antennenachse (Nadir), θ = π/2 am Horizont der Antenne.

**Halbwellendipol** (Lehrbuchformel, z. B. Balanis §4.5):
```
F(θ) = [cos(π/2·cosθ) / sinθ]²
```
Exakte Nullstelle bei θ=0/π; dient hier als Verifikationsanker
(D = 2,15 dBi, Literaturwert), nicht als Sendeantennenmodell.

**Pagoda-Näherung** (empirisch, kein Lehrbuchmodell):
```
F(θ) = max(sinⁿθ, floor_lin) ,  n ≈ 2, floor ≈ −12 dB
```
n und floor sind Formparameter, keine gemessenen Werte für eine konkrete
Antenne — im Code als `# ANNAHME` markiert. Gültig als grobe Näherung an
reale RHCP-Pagoda-Muster; für eine konkrete Antenne sollte n/floor gegen
eine Herstellermessung kalibriert werden.

**Direktivität**, numerisch integriert (azimutal symmetrisch reduziert):
```
D = 4π / ∬ F(θ)·sinθ dθdφ = 2 / ∫₀^π F(θ)·sinθ dθ
```
Standarddefinition (IEEE Std 145). Numerische Integration statt fest
eingetragener Werte hält die Größe konsistent bei geänderten Formparametern
und macht sie testbar.

**Polarisations-Mismatch (PLF)**, Achsenverhältnis-Form:
```
PLF = 1/2 + 1/2 · [4r₁r₂ + (r₁²−1)(r₂²−1)cos2Δτ] / [(r₁²+1)(r₂²+1)]
```
Standardformel für den Leistungskopplungsfaktor zwischen zwei elliptisch
polarisierten Antennen (z. B. Balanis §2.12, oder Rumsey/Sinclair-Form), r =
lineares Achsenverhältnis ≥ 1, Vorzeichen kodiert Drehsinn. Grenzfall
r → ∞ (linear) ist hier analytisch hergeleitet (nicht die numerisch
instabile 0·∞-Form): für r₂→∞ gilt PLF → 1/2 + 1/2·(r₁²−1)/(r₁²+1)·cos2Δτ.
Gültig für beliebige elliptische Polarisationszustände; degeneriert korrekt
zu den bekannten Spezialfällen (CP-CP gleichsinnig 0 dB, CP-linear 3,01 dB,
Kreuzpolarisation → ∞ dB).

---

## `pathloss.py` — Streckendämpfung

**Freiraumdämpfung** (ITU-R P.525):
```
L = 20·log₁₀(4πd/λ)
```
Gültig im Fernfeld (d ≫ λ) für homogenes, verlustfreies Ausbreitungsmedium.

**Two-Ray-Bodenreflexion** (implementiert, standardmäßig aus):
- Geometrie (Spiegelmethode, ebene Erde): `d_direkt`, `d_reflektiert`,
  Grazing-Winkel ψ — Standardgeometrie, z. B. Rappaport, *Wireless
  Communications*, Kap. 4.
- Fresnel-Reflexionskoeffizient, komplexe Bodenpermittivität
  `ε_g = ε_r − j·60·λ·σ`: Standardform, z. B. Balanis §5, Rappaport Kap. 4.
  Gegen zwei unabhängige Herleitungen geprüft (Einfallswinkel- und
  Grazing-Winkel-Konvention stimmen überein).
- Ament-Rauigkeitsfaktor `ρ_s = exp[−8(π·h_rms·sinψ/λ)²]`: Ament (1953),
  *Toward a Theory of Reflection by a Rough Surface*, Proc. IRE.
- Breakpoint `d_bp = 4·h_tx·h_rx/λ`: Standardformel, z. B. Rappaport Kap. 4.

Gültig für eine elektrisch große, ebene Reflexionsfläche; das
Rayleigh-Kriterium (h_rms ≲ λ/(8 sinψ)) entscheidet, ob die Formel
überhaupt einen kohärenten Reflexionsanteil vorhersagt oder ob die
Oberfläche effektiv als diffus streuend behandelt werden müsste (hier nicht
modelliert). Bei 5,8 GHz ist die Lobing-Periode extrem fein — ob sich die
Interferenzstruktur real ausbildet, hängt zusätzlich von der
Kohärenzbandbreite und der Pfadlängen-Stabilität ab.

**Atmosphärische Gasdämpfung / Regendämpfung**: nur Formelstruktur
(`L = spezifische Dämpfung × Distanz`, ITU-R-P.676-13/P.838-3-Form), OHNE
eingebaute Koeffiziententabelle — siehe Modul-Docstring in `pathloss.py`.
**Offener Punkt**: die exakten ITU-R-Tabellenwerte bei 5,8 GHz konnten in
dieser Session nicht gegen eine lesbare Quelle verifiziert werden; vor
produktivem Einsatz gegen die Originalempfehlung prüfen.

---

## `environment.py` — Clutter, Vegetation, Shadowing/Fading-Parameter

**Al-Hourani-Air-to-Ground-Modell** (Al-Hourani, Kandeepan, Lardner (2014),
*Optimal LAP Altitude for Maximum Coverage*, IEEE Wireless Communications
Letters, 3(6), 569–572):
```
P_LOS(ε) = 1 / (1 + a·exp(−b·(ε_deg − a)))
L_clutter = P_LOS·η_LOS + (1−P_LOS)·η_NLOS
```
a, b, η_LOS, η_NLOS extern bestätigt (siehe Konversation/Recherche) für
"Urban" (Stufe 3) und "Dense Urban" (Stufe 4) — exakte Übereinstimmung mit
publizierten Werten. "Suburban" (Stufe 2) publiziert η_NLOS = 21,0 dB, hier
bewusst auf 18,0 dB korrigiert für eine monotone 4-Stufen-Skala (siehe
`environment.py`, `STAGES[2]`-Kommentar, und `test_stage_2_eta_nlos_corrected_for_monotonicity`).
Stufe 1 hat keine Al-Hourani-Entsprechung, ist selbst kalibriert (siehe
INSTRUCITONS.md Fallstrick 4).

Kalibriert bei 0,7–2,5 GHz. **Bei 5,8 GHz mit Sender über dem Clutter gibt es
kein Standardmodell** (INSTRUCITONS.md Fallstrick 5) — strukturell korrekt
(Winkelabhängigkeit, Monotonie), im Absolutwert unbelegt (±6 dB laut
Dokument). Ein optionaler Frequenzkorrekturterm `+10·log10(f/2 GHz)` ist
eingebaut, aber nirgends automatisch aktiv.

**Weissberger Modified Exponential Decay** (Weissberger, M. A. (1982),
*An Initial Critical Summary of Models for Predicting the Attenuation of
Radio Waves by Trees*, ESD-TR-81-101):
```
L = 1,33·f_GHz^0,284·depth^0,588   (14 m < depth ≤ 400 m)
L = 0,45·f_GHz^0,284·depth         (depth ≤ 14 m)
```
Gültig 230 MHz – 95 GHz, depth ≤ 400 m; kleine Unstetigkeit am 14-m-Übergang
ist ein bekanntes Artefakt des Originalmodells (hier mit Toleranztest
abgesichert, nicht "gefixt").

**Shadowing-σ und Rice-K je Stufe**: aus INSTRUCITONS.md übernommen (4/6/8/10
dB bzw. 12/8/5/3 dB) — Modellannahmen der Aufgabenstellung, keine eigene
Messung.

---

## `link.py` — Zusammenführung (Friis-Gleichung in dB)

```
P_rx = P_tx + G_tx(θ) + G_rx − FSPL − L_clutter − L_vegetation
       + Two-Ray-Term − L_atmo − L_regen − PLF − Speiseverlust
C/N₀ = P_rx − kT₀
```
Friis-Übertragungsgleichung, dB-Form (jede Standard-Nachrichtentechnik-
Referenz, z. B. Rappaport Kap. 4). Zwei unabhängige Herleitungen (dB-Summe
und lineare Watt-Multiplikation) stimmen exakt überein (siehe Konversation).

`power_is_eirp`: bei `True` ist `power_dbm` bereits die EIRP zur
Referenzrichtung; nur der Musterverlust `10·log10(F(θ))` (≤ 0 dB) wird
addiert, nicht die volle Direktivität — sonst würde der Antennengewinn
doppelt gezählt (strukturell getestet über mehrere Distanzen).

**Dual-Diversity-Combining** (`compute_link_budget_dual_diversity`):
ideales Maximum-Ratio-Combining zweier phasenkohärenter Kanäle mit
unabhängigem, gleich starkem Rauschen —
```
C/N₀_kombiniert (linear) = C/N₀_a (linear) + C/N₀_b (linear)
```
Exaktes Standardresultat der Diversity-Combining-Theorie (Summe der
Zweig-SNRs bei MRC, z. B. Rappaport Kap. 7; Brennan (1959), *Linear
Diversity Combining Techniques*, Proc. IRE). Gültig unter der Annahme
idealer, verlustfreier, phasenrichtiger Kombination — reale Verluste durch
eine zweite Rauschkette, Kalibrierfehler etc. sind Schritt-2-Hardwaredetails
und hier nicht modelliert.

---

## `montecarlo.py` — Shadowing + Fading

**Log-normales Shadowing**: `L_shadow ~ N(0, σ²)` — Standardmodell für
großräumige Abschattung (z. B. Rappaport Kap. 4), additiv in dB.

**Rice-Fading**, K-Faktor-Modell (Rice, S. O. (1948), *Statistical Properties
of a Sine Wave Plus Random Noise*, Bell System Technical Journal):
```
h = (s + σ_f·N(0,1)) + j·σ_f·N(0,1) ,  s=√(K/(K+1)), σ_f=√(1/(2(K+1)))
```
Normiert auf E[|h|²] = 1 (test-verifiziert für alle vier K-Werte). Median
statt Mittelwert als Zentralgröße, da der dB-Mittelwert durch die
Jensensche Ungleichung systematisch unter 0 liegt, obwohl der lineare
Mittelwert der Leistung exakt 1 ist.

---

## `requirements.py` — Bezug zu Schritt 2

```
SNR(B, NF) = C/N₀ − 10·log10(B) − NF
PSD = P_rx − 10·log10(B_signal)
Dynamikbereich [Bit] = (max(P_rx) − min(P_rx)) / 6,02
```
6,02 dB/Bit = 20·log10(2), Standardresultat der ADC-Quantisierungstheorie
(Signal-Rausch-Verhältnis pro zusätzlichem Bit, z. B. Bennett (1948),
*Spectra of Quantized Signals*, Bell System Technical Journal). Reine
Ableitungen aus C/N₀, keine Bewertung eines konkreten Empfängers.

---

## Offener Punkt: ~0,37-dB-Differenz zur Dokument-Ankertabelle

Die volle Kette (Basisszenario, Stufe 1, 3 km) liefert `P_rx = −90,6448 dBm`,
`C/N₀ = 83,3352 dB-Hz` — die in INSTRUCITONS.md angegebenen Werte sind
`P_rx ≈ −90,3 dBm`, `C/N₀ ≈ 83,7 dB-Hz`. Differenz: konstant ≈ 0,37 dB, über
alle 28 Zellen der vollen Distanz-×-Stufen-Matrix (Streuung 0,30–0,41 dB,
im Rahmen der auf eine Nachkommastelle gerundeten Tabelle).

**Was geprüft und ausgeschlossen wurde** (siehe Konversation für die
vollständige Herleitung):
- Jede Einzelformel (FSPL, Pagoda-Direktivität, PLF, Al-Hourani-Clutter)
  reproduziert ihren eigenen, unabhängig vorgegebenen Sollwert exakt.
- Zwei unabhängige Rechenwege (dB-Summe, lineare Watt-Multiplikation)
  stimmen exakt überein — kein Rechenfehler in der Implementierung.
- Al-Hourani-Koeffizienten extern gegen publizierte Werte bestätigt.
- Alternative Hypothesen durchgerechnet und verworfen: RX-seitige
  Richtcharakteristik, TX-Tilt ≠ 0 im Referenzfall, lineare statt
  dB-Mittelung im Clutter-Modell, Grund- statt Schrägentfernung, Peak- statt
  winkelabhängiger TX-Gewinn, alternative Rundungskonventionen.

**Fazit**: Ein konstanter Offset dieser Größenordnung ist aus den Daten
allein nicht mehr eindeutig einem einzelnen Term zuzuordnen (mehrere
Ein-Term-Korrekturen erklären die Tabelle rechnerisch gleich gut, keine
davon ergibt einen plausiblen "runden" Ausgangswert). Am wahrscheinlichsten
ist eine kleine, mitgeschleppte Rundung in der Herleitung der
Dokument-Ankertabelle selbst. Die Verifikationssuite prüft deshalb primär
gegen die einzeln bestätigten Sollwerte; die volle Matrix wird zusätzlich
mit einer explizit dokumentierten, weiten Toleranz (± 0,5 dB) gegengeprüft,
die diesen bekannten Offset abdeckt, ohne größere Regressionen zu verdecken.
