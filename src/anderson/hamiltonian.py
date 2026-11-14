"""Hamiltonoperator und Grundgroessen der 1D-Anderson-Kette."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def hamiltonian(
    length: int,
    hopping: float = 1.0,
    disorder: float = 0.0,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Erzeugt H fuer eine offene Kette mit i.i.d. gleichverteilter Unordnung."""
    if length < 1:
        raise ValueError("length muss positiv sein")
    if hopping <= 0:
        raise ValueError("hopping muss positiv sein")
    if disorder < 0:
        raise ValueError("disorder darf nicht negativ sein")

    generator = np.random.default_rng() if rng is None else rng
    matrix = np.diag(
        generator.uniform(-disorder / 2, disorder / 2, size=length)
    )
    bond = np.full(length - 1, -hopping)
    matrix += np.diag(bond, 1) + np.diag(bond, -1)
    return matrix


def clean_energies(length: int, hopping: float = 1.0) -> NDArray[np.float64]:
    """Exakte Eigenwerte der homogenen offenen Tight-Binding-Kette."""
    if length < 1:
        raise ValueError("length muss positiv sein")
    if hopping <= 0:
        raise ValueError("hopping muss positiv sein")
    indices = np.arange(1, length + 1)
    return -2 * hopping * np.cos(indices * np.pi / (length + 1))


def diagonalize(
    length: int,
    hopping: float = 1.0,
    disorder: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Diagonalisiert eine Realisierung und gibt Eigenwerte und Eigenvektoren zurück."""
    energies, states = np.linalg.eigh(
        hamiltonian(length, hopping=hopping, disorder=disorder, rng=rng)
    )
    return energies, states


def ipr(state: ArrayLike) -> float:
    """Inverse Partizipationsrate eines normierten Zustands."""
    amplitudes = np.asarray(state)
    norm = np.linalg.norm(amplitudes)
    if norm == 0:
        raise ValueError("Der Zustand darf nicht der Nullvektor sein")
    normalized = amplitudes / norm
    return float(np.sum(np.abs(normalized) ** 4))


def ipr_all(states: ArrayLike) -> NDArray[np.float64]:
    """Berechnet die IPR jeder Spalte einer Eigenvektormatrix."""
    amplitudes = np.asarray(states)
    if amplitudes.ndim != 2:
        raise ValueError("states muss eine zweidimensionale Matrix sein")
    norms = np.linalg.norm(amplitudes, axis=0)
    if np.any(norms == 0):
        raise ValueError("Zustaende dürfen nicht der Nullvektor sein")
    normalized = amplitudes / norms
    return np.sum(np.abs(normalized) ** 4, axis=0)
