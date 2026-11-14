# CiS-Projekt: Anderson-Lokalisierung

Numerische Untersuchung der Anderson-Lokalisierung in einer eindimensionalen
Tight-Binding-Kette mit offenen Randbedingungen.

## Inhalte

- Hamiltonoperator und Diagonalisierung fuer endliche Ketten
- Test des Spektrums bei verschwindender Unordnung
- Aufenthaltswahrscheinlichkeiten `|psi_j|^2`
- inverse Partizipationsrate (IPR)
- Skalierung mit Systemgroesse `L` und Unordnungsstaerke `W/t`

## Installation

```bash
python -m pip install -e ".[dev]"
```

## Beispiel

```bash
python scripts/run_study.py --lengths 50 100 200 --disorder 0 1 4 --samples 20
```

Dabei werden `ipr_near_zero.csv`, `ipr_near_zero.png`,
`ipr_vs_length.png` und `probabilities.png` im Ordner `results/` gespeichert.
`--no-plots` deaktiviert die Abbildungen, und `--output PFAD` wählt einen
anderen Ausgabeordner.

Die zentrale Funktion `diagonalize(...)` liefert die sortierten Eigenwerte und
die zugehörigen normierten Eigenvektoren als Spaltenmatrix. `ipr_all(...)`
berechnet die IPR aller dieser Zustände.

## Physikalischer Benchmark

Fuer `W=0` muessen die Eigenwerte einer Kette der Laenge `L` mit

```text
E_n = -2 t cos(n pi / (L + 1)),  n = 1, ..., L
```

uebereinstimmen.

## KI-Nutzung

Bei der Programmierung koennen KI-Werkzeuge zur Unterstuetzung eingesetzt
werden. Die konkrete Nutzung wird im Bericht transparent angegeben.
