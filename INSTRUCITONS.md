# Claude-Code-Prompt: RF-Ausbreitungssimulation 5,8-GHz-FPV-Downlink

**Schritt 1: Wie stark ist das Signal am Boden — hardwareunabhängig.**

Zwei Teile. **Teil 0** sind Entscheidungen, die du treffen musst. **Teil 1** ist
der Prompt: ab der Linie alles markieren und in Claude Code einfügen.

---

## Der Schnitt zwischen Schritt 1 und Schritt 2

Schritt 1 rechnet bis zur **Antennenklemme** und hört dort auf. Ergebnis ist die
Empfangsleistung `P_rx` und daraus die Kenngröße, die kein SDR kennt:

```
C/N₀ [dB-Hz] = P_rx [dBm] − kT₀ [dBm/Hz] = P_rx + 173,98
```

**C/N₀ ist Trägerleistung geteilt durch Rauschleistungsdichte.** Es hängt nur von
Sender, Strecke und Antenne ab — nicht von Bandbreite, Rauschzahl, ADC-Bits oder
FFT-Länge. Genau deshalb ist es die richtige Ausgabe für Schritt 1.

In Schritt 2 fällt daraus jedes SDR heraus, mit einer Zeile:

```
SNR(B, NF) = C/N₀ − 10·log10(B) − NF
```

Beispiel aus dem Basisszenario (C/N₀ = 83,7 dB-Hz):

| Empfänger | SNR |
|---|---|
| 20 MHz, NF 6 dB | +4,7 dB |
| 27 MHz, NF 6 dB | +3,4 dB |
| 27 MHz, NF 3 dB (Mast-LNA) | +6,4 dB |
| 56 MHz, NF 6 dB | +0,2 dB |

Alles, was eine Bandbreite, eine Rauschzahl oder eine FFT braucht, gehört
deshalb **nicht** in Schritt 1. Der Prompt unten hält das konsequent durch.

---

# TEIL 0 — Was du vorher entscheiden musst

Nur noch die Punkte, die `P_rx` verschieben. Rauschzahl, Bandbreite und
ADC-Auflösung sind bewusst nicht dabei — das ist Schritt 2.

| # | Frage | Spanne | Default im Prompt |
|---|---|---|---|
| 1 | **Sind die 2 W EIRP oder HF-Ausgangsleistung?** VTX-Datenblätter meinen fast immer die PA-Ausgangsleistung, Zulassungsangaben (CE/FCC) meinen EIRP. Welcher VTX konkret? Viele „2-W"-Module liefern real 1,2–1,6 W. | 1,8 dB | Ausgangsleistung |
| 2 | **Empfangsantenne: Gewinn.** Rundstrahler am Dach = 2–3 dBi. 4-Element-Array kohärent = +6 dB, 8 Elemente = +9 dB. Der größte einzelne Hebel. | bis 9 dB | 2 dBi |
| 3 | **Empfangsantenne: Polarisation.** RHCP-Sender auf lineare Antenne kostet konstant 3 dB. Mit RHCP-Empfang entfällt das — aber gegen einen LHCP-Sender werden daraus >20 dB. Für einen Detektor, der *jeden* Sender finden soll, ist linear die robustere Wahl: 3 dB sicher statt Glücksspiel. | 3 bis >20 dB | linear |
| 4 | **Fluglage.** Senkrecht hängende Antenne oder Vorwärtsflug mit 20–30° Nickwinkel? Bei flachen Elevationswinkeln (Basisszenario: 1,87°) egal, bei kurzen Entfernungen mit steilem Winkel entscheidend. | 0–10 dB | senkrecht |
| 5 | **Gelände.** Ebene Annahme oder echtes Höhenprofil (SRTM)? Nachgerechnet: bei 100 m / 3 km liegt die Sichtlinie mittig auf 51 m, erste Fresnelzone 6,2 m Radius → Freiheitsgrad 8,2, klar Freiraumfall. Gilt aber nur für ebenes Gelände; ein Hügel dazwischen oder das Fahrzeug in einer Senke kippt das komplett. | bis 20 dB | eben |

## Der wichtigste konzeptionelle Punkt

Deine „Rauschstufe 1–4" sind physikalisch **drei getrennte Dinge**:

1. **Clutter-Dämpfung** — schwächt das Signal. Bäume und Gebäude auf den letzten
   Metern vor dem Empfänger.
2. **Shadowing-σ** — streut es statistisch, 4 dB (Land) bis 10 dB (Stadt).
3. **Störflur** — hebt das Rauschen.

Nur 1 und 2 gehören in Schritt 1, weil nur sie `P_rx` verändern. Punkt 3 wirkt
erst gegen einen konkreten Empfänger — Schritt 2.

Und dabei die Überraschung: **Bei 5,8 GHz gibt es praktisch kein man-made
noise.** Die ITU-R-P.372-Kurven enden bei ~1 GHz, weil Zündfunken und
Schaltnetzteile darüber nichts mehr beitragen. Was in der Stadt bei 5,8 GHz
stört, ist WLAN — und das ist kein gaußartiges Rauschen, sondern bursty und
kanalisiert. Für einen Detektor, dessen stärkster Erstfilter „100 % Duty Cycle"
ist, fällt WLAN gerade heraus. Der Stadt-Land-Unterschied kommt deshalb fast
vollständig aus der Dämpfung.

---
---

# TEIL 1 — Ab hier alles kopieren

---

Baue ein Python-Paket `rf_linksim`, das berechnet, **wie stark das Signal eines
analogen 5,8-GHz-FPV-Videodownlinks an einer Empfangsantenne am Boden ankommt**.

Das ist Schritt 1 eines zweistufigen Vorhabens. Ziel ist herauszufinden, welche
Signalpegel überhaupt auftreten — daraus werden anschließend die Anforderungen an
ein SDR abgeleitet. **Schritt 1 muss deshalb streng hardwareunabhängig bleiben.**

## Die zentrale Randbedingung

Die Rechnung endet an der **Antennenklemme**. Kernausgabe ist:

```
P_rx  [dBm]    Empfangsleistung an der Antennenklemme
C/N₀  [dB-Hz]  = P_rx − kT₀ = P_rx + 173,98
```

C/N₀ ist Trägerleistung durch Rauschleistungsdichte und hängt nur von Sender,
Strecke und Antenne ab. In Schritt 2 folgt daraus jeder Empfänger mit einer
Zeile: `SNR(B, NF) = C/N₀ − 10·log10(B) − NF`.

**Deshalb gilt für Schritt 1:** keine Empfängerbandbreite, keine Rauschzahl,
keine FFT-Länge, kein Integrationsgewinn, keine ADC-Auflösung, kein SDR-Modell.
Wenn eine Größe eine Bandbreite oder eine Rauschzahl braucht, gehört sie nicht in
dieses Paket. Einzige Ausnahme ist die Konstante kT₀ = −173,98 dBm/Hz, die keine
Hardwareeigenschaft ist, sondern Physik bei 290 K.

Ebenso: **keine Pass/Fail-Bewertung, keine Reichweitenaussage.** Ausgabe ist eine
reine Pegelkarte. Referenzschwellen aus der Literatur dürfen als Konstanten
bereitstehen und in Plots als Linien auftauchen, aber das Modell wertet sie
nirgends aus.

## Basisszenario

| Größe | Wert |
|---|---|
| Frequenz | 5,8 GHz (Suchraum 5325–5945 MHz) |
| Signalbandbreite | 27 MHz (RSGB-Messung; nur für die Leistungsdichte, s. u.) |
| Sendeleistung | 2 W = 33,0 dBm **HF-Ausgangsleistung**, nicht EIRP |
| Sendeantenne | RHCP-Pagoda, senkrecht hängend |
| Flughöhe | 100 m über Grund |
| Grundentfernung | 3 km |
| Empfangsantenne | 2 dBi, linear polarisiert, 2 m über Grund (Fahrzeugdach) |
| Speiseverlust RX | 1,5 dB (Kabel Dach → Gerät) |

## Struktur

```
src/rf_linksim/
  constants.py     Naturkonstanten, kT₀, dB-Umrechnungen
  geometry.py      Schrägentfernung, Elevation, Fresnel, Radiohorizont
  antenna.py       Richtcharakteristiken, Polarisations-Mismatch
  pathloss.py      FSPL, Atmosphäre, Regen, optional Two-Ray
  environment.py   Umgebungsstufen 1–4: Clutter, Shadowing
  link.py          die Kette → P_rx und C/N₀, alle Zwischenwerte
  montecarlo.py    Shadowing + Fading, Perzentile
  sweep.py         Distanz-, Höhen-Sweeps, 2D-Gitter
  requirements.py  Ableitungen für die SDR-Spezifikation
  plotting.py      Standardplots
  config.py        YAML → Objekte
config/default.yaml
scripts/run_baseline.py, run_sweeps.py, run_sdr_requirements.py
tests/test_physics.py
docs/physik.md
```

Konvention: Frequenzen in Hz, Längen in m, Leistungen in dBm, Winkel intern im
Bogenmaß. Numpy-Arrays überall, damit Sweeps ohne Schleifen laufen. Dataclasses
für Szenario, Sender, Empfangsantenne, Ergebnis. Jeder Modul-Docstring erklärt
die Physik, nicht die Signatur.

Halte den Schnitt zu Schritt 2 sauber: `link.py` gibt C/N₀ zurück und weiß nichts
von Empfängern. Schritt 2 wird später ein Modul `receiver.py` ergänzen, das C/N₀
plus SDR-Kenndaten entgegennimmt. Baue nichts davon vorweg.

## Die Formeln

**Geometrie.** In die Freiraumdämpfung gehört die *Schräg*entfernung, nicht die
Grundentfernung: `d_slant = sqrt(d_ground² + (h_tx − h_rx)²)`. Bei 100 m / 3 km
sind das nur 1,6 m Unterschied, bei 100 m / 200 m aber 12 % ≙ 1,0 dB.
Elevationswinkel `ε = atan2(h_tx − h_rx, d_ground)`.

**Winkel zur Sendeantenne.** Die Antenne hängt senkrecht unter der Drohne, ihre
Achse zeigt nach unten. Der Strahl zum Empfänger liegt von dieser Achse aus bei
`θ = π/2 − ε − tilt`. Ein Rundstrahler hat sein Maximum bei θ = π/2 (Horizont)
und seine Null bei θ = 0 (senkrecht unter der Drohne).

**Freiraumdämpfung** (ITU-R P.525): `L = 20·log10(4πd/λ)`.

**Antennenmuster.** Halbwellendipol: `F(θ) = [cos(π/2·cos θ)/sin θ]²`, echte Null
auf der Achse. RHCP-Pagoda: `F(θ) = max(sin^n(θ), floor)` mit n ≈ 2 und `floor`
= −12 dB — die Null ist bei realen Pagodas durch Streustrahlung aufgefüllt, nicht
unendlich tief.

**Direktivität numerisch integrieren**, nicht hart eintragen:
`D = 4π / ∫∫ F(θ)·sin θ dθ dφ` mit F auf Maximum 1 normiert. Das macht sie
testbar und bleibt konsistent, wenn man die Formparameter ändert.

**Polarisations-Mismatch:**
```
PLF = ½ + ½ · (4·r₁·r₂ + (r₁²−1)(r₂²−1)·cos 2Δτ) / ((r₁²+1)(r₂²+1))
```
r = lineares Achsenverhältnis ≥ 1, Δτ = Winkel zwischen den Hauptachsen.
Gegenläufiger Drehsinn über das Vorzeichen von r₂. Die Grenzfälle r → ∞ (perfekt
linear) brauchen einen eigenen Zweig, sonst wird es numerisch instabil.

**Clutter-Dämpfung**, Air-to-Ground nach Al-Hourani et al. 2014:
```
P_LOS(ε) = 1 / (1 + a·exp(−b·(ε_deg − a)))
L_clutter = P_LOS·η_LOS + (1 − P_LOS)·η_NLOS
```

| Stufe | Umgebung | a | b | η_LOS | η_NLOS | σ | Rice-K |
|---|---|---|---|---|---|---|---|
| 1 | freies Land | 2,0 | 1,20 | 0,1 | **8,0** | 4 dB | 12 dB |
| 2 | Dorf/Vorstadt | 4,88 | 0,43 | 0,1 | **18,0** | 6 dB | 8 dB |
| 3 | Stadtrand | 9,61 | 0,16 | 1,0 | 20,0 | 8 dB | 5 dB |
| 4 | Stadt | 12,08 | 0,11 | 1,6 | 23,0 | 10 dB | 3 dB |

**Vegetation** (Weissberger MED, zusätzlich zum Clutter):
`L = 1,33·f_GHz^0,284·d^0,588` für 14 < d ≤ 400 m, darunter `0,45·f_GHz^0,284·d`.

**Shadowing:** log-normal N(0, σ²). **Schneller Schwund:** Rice mit K-Faktor, auf
`E[|h|²] = 1` normiert, sonst verschiebt der Schwund den Mittelwert:
```
s = sqrt(K/(K+1));  σ_f = sqrt(1/(2(K+1)));  h = (s + σ_f·randn) + j·σ_f·randn
```
Beide Terme sind in dB additiv — die deterministische Kette also **einmal**
rechnen und nur die Zufallsterme daraufaddieren, nicht n_runs-mal die ganze
Kette. Der Median ist die richtige Zentralgröße, nicht der Mittelwert.

**Two-Ray-Bodenreflexion** implementieren, aber per Default **aus**. Direkter und
reflektierter Strahl mit Fresnel-Reflexionskoeffizient und Ament-Rauigkeits-
dämpfung. Breakpoint `4·h_tx·h_rx/λ` = 15,5 km bei 100 m / 2 m / 5,8 GHz — der
ganze interessante Bereich liegt also in der Interferenzzone. Bei 5,8 GHz ist die
Lobing-Struktur aber extrem fein (eine Periode = wenige Meter Höhe), und ob sie
sich real ausbildet, hängt an der Bodenrauigkeit: das Rayleigh-Kriterium fordert
glatter als ~20 cm RMS. Über Wasser und Asphalt voll da, über Acker grenzwertig,
über Wald weg.

## Was `requirements.py` liefern soll

Das ist die Brücke zu Schritt 2 — aber noch ohne konkrete Hardware. Drei reine
Ableitungen aus C/N₀:

**1. Maximal zulässige Rauschzahl** für ein gegebenes Ziel-SNR und eine gegebene
Bandbreite: `NF_max = C/N₀ − 10·log10(B) − SNR_ziel`. Als Tabelle über mehrere
Bandbreiten und Ziel-SNRs, ohne eine davon auszuwählen — das ist genau die
Spezifikation, gegen die später SDRs geprüft werden.

**2. Erforderlicher Dynamikbereich.** Die Spanne zwischen dem stärksten und dem
schwächsten Szenario im Sweep-Raum, in dB und in ADC-Bits (`dB / 6,02`). Das ist
die Anforderung an die ADC-Auflösung — und der Grund, warum die Wahl zwischen
8 bit und 12 bit keine Geschmacksfrage ist.

**3. Spektrale Leistungsdichte** `PSD = P_rx − 10·log10(B_signal)` in dBm/Hz.
Damit lässt sich in Schritt 2 für jede beliebige Auflösungsbandbreite ausrechnen,
wieviel Signal in einem Bin landet. **Achtung:** Das ist die einzige Stelle, an
der die Signalbandbreite auftaucht, und sie beschreibt den *Sender*, nicht den
Empfänger. Verwechsle sie nicht mit einer Empfängerbandbreite.

## Fallstricke — hier bitte besonders sorgfältig

**1. Halte Schritt 1 sauber.** Die Versuchung ist groß, „schnell noch" ein SNR
auszurechnen. Sobald irgendwo eine Empfängerbandbreite oder eine Rauschzahl
auftaucht, ist der hardwareunabhängige Charakter der Ausgabe verloren und der
ganze Zweck von Schritt 1 dahin. C/N₀ ist die Grenze.

**2. Trenne Gesamtleistung von Leistungsdichte.** `P_rx` ist die gesamte
einfallende Leistung, verteilt über 27 MHz. `PSD` ist die Dichte. Wer die beiden
verwechselt, liegt um 74 dB daneben.

**3. Die Al-Hourani-Koeffizienten sind untereinander nicht monoton.** Publiziert
ist suburban η_NLOS = 21 und urban η_NLOS = 20 — die Werte stammen aus getrennten
Messkampagnen. Für eine Skala 1–4 muss die Reihe monoton sein, deshalb steht in
der Tabelle oben für Stufe 2 der Wert 18,0 statt 21,0. Kommentiere die Abweichung
im Code und sichere die Monotonie mit einem Test über den gesamten
Elevationsbereich ab.

**4. Stufe 1 hat bei Al-Hourani keine Entsprechung** und ist selbst gesetzt. Das
Modell kennt die Streckenlänge nicht, nur den Winkel — bei 100 m / 3 km liegt die
Sichtlinie mittig auf 51 m und ist über jedem Baum frei, während das Modell aus
dem flachen Winkel auf NLOS schließt. Die Werte für Stufe 1 sind darauf
kalibriert, dass Clutter nur auf den letzten ~300 m vor dem Fahrzeug entsteht.

**5. Für 5,8 GHz mit Sender über dem Clutter gibt es kein Standardmodell.**
Al-Hourani ist bei 0,7–2,5 GHz kalibriert, ITU-R P.2108 §3.3 (genau unser
Geometriefall) gilt erst ab 10 GHz, und P.2108 §3.2 setzt beide Enden im Clutter
voraus. Die Werte sind strukturell richtig, im Absolutwert aber ungesichert
(±6 dB). Baue deshalb einen zweiten Modus `clutter_model: "fixed"` mit einem
konstanten kalibrierbaren dB-Wert je Stufe ein — sobald eine Referenzmessung
vorliegt, wird darauf umgestellt. Einen optionalen Frequenzkorrekturterm
`+10·log10(f/2 GHz)` vorsehen, aber **per Default aus**, weil unbelegt.

**6. Die Antennen-Null dominiert den Nahbereich.** Bei 3 km steht die Drohne
1,87° über dem Horizont — voller Antennengewinn. Bei 200 m sind es 26°, bei 100 m
schon 44°, und die Pagoda verliert dort mehrere dB. Im Distanz-Sweep kann die
Empfangsleistung dadurch bei *kürzerer* Distanz *sinken*. Das ist kein Fehler,
sondern der Effekt, den der Sweep zeigen soll — im Docstring erklären, damit es
später niemand „wegfixt".

## Verifikation

`tests/test_physics.py` mit pytest. Jeder Test prüft eine Größe, deren Sollwert
**unabhängig von diesem Code** bekannt ist. Diese Werte sind nachgerechnet und
müssen exakt herauskommen:

| Prüfung | Sollwert |
|---|---|
| k·T₀ bei 290 K | −173,98 dBm/Hz |
| λ bei 5,8 GHz | 51,69 mm |
| Schrägentfernung 3 km / 100 m / 2 m | 3001,6 m |
| Elevationswinkel dazu | 1,871° |
| FSPL 5 km @ 5,8 GHz | 121,70 dB |
| FSPL 6 km @ 5,8 GHz | 123,28 dB |
| FSPL: Entfernung verdoppeln | exakt 6,02 dB |
| Radiohorizont k = 4/3 | d[km] = 4,12·√h[m] |
| Erste Fresnelzone mittig, 3 km | 6,23 m |
| Two-Ray-Breakpoint 100 m / 2 m | 15,5 km |
| Halbwellendipol-Direktivität | 2,15 dBi |
| Isotrop | 0,00 dBi |
| Pagoda (sin², −12 dB Fill) | 1,75 dBi |
| Rundstrahler-Maximum | bei θ = 90° |
| PLF: CP → CP gleichsinnig | 0,00 dB |
| PLF: CP → linear | 3,01 dB |
| PLF: RHCP → LHCP | → ∞ |
| PLF: linear → linear, 90° verdreht | → ∞ |
| Weissberger 10 m @ 5,8 GHz | 7,4 dB |
| Monte-Carlo: P95 − P5 | 2·1,645·σ |
| Rician-Schwund | E\[\|h\|²\] = 1 |
| C/N₀ − P_rx | exakt 173,98 dB |

Dazu strukturelle Tests: Die Summe aller Budget-Terme muss `P_rx` exakt ergeben.
Eine dichtere Umgebungsstufe darf das Signal nie verbessern. Die Clutter-Reihe
muss über den gesamten Elevationsbereich monoton sein. Senkrecht über dem
Empfänger bleibt nur η_LOS übrig. `power_is_eirp = true` darf den Antennengewinn
nicht doppelt zählen. Ohne Clutter und mit isotroper Antenne kostet jede
Distanzverdopplung exakt 6,02 dB.

**Plausibilitätsanker fürs Ganze.** Das Basisszenario in Stufe 1 muss auf
`P_rx ≈ −90,3 dBm` und `C/N₀ ≈ 83,7 dB-Hz` kommen. Die vollständige C/N₀-Matrix
in dB-Hz als Sollwert:

| d [m] | Stufe 1 | Stufe 2 | Stufe 3 | Stufe 4 |
|---|---|---|---|---|
| 200 | 110,9 | 110,9 | 102,3 | 94,0 |
| 500 | 104,5 | 100,0 | 86,8 | 83,1 |
| 1000 | 98,5 | 84,7 | 79,8 | 76,6 |
| 2000 | 89,3 | 76,2 | 73,5 | 70,5 |
| 3000 | **83,7** | 72,3 | 69,9 | 66,9 |
| 5000 | 78,1 | 67,6 | 65,4 | 62,4 |
| 6000 | 76,3 | 65,9 | 63,8 | 60,8 |

Kommt etwas deutlich anderes heraus, stimmt ein Vorzeichen oder ein Winkel nicht.

## Vorgehen

Arbeite modulweise: erst `constants`/`geometry`, dann `antenna`, `pathloss`,
`environment`, dann `link` als Zusammenführung, zuletzt `montecarlo`, `sweep`,
`requirements`, `plotting`. Nach jedem Modul die zugehörigen Tests schreiben und
laufen lassen, bevor du weitergehst — die Verifikationswerte oben sind so
sortiert, dass das geht.

Schreibe `docs/physik.md` mit allen Formeln, ihren Quellen und den
Gültigkeitsgrenzen. Markiere in `config/default.yaml` jeden Wert, der eine
Annahme ist und keine Messung, mit `# ANNAHME` — davon gibt es mehr als man
denkt, und in drei Monaten weiß das sonst niemand mehr.
