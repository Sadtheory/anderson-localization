"""Kleines Startskript fuer IPR-Messungen nahe E=0."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from anderson import hamiltonian, ipr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", nargs="+", type=int, default=[50, 100, 200])
    parser.add_argument("--disorder", nargs="+", type=float, default=[0, 1, 2, 4])
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--hopping", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    output = Path("results")
    output.mkdir(exist_ok=True)
    rows = []
    for length in args.lengths:
        for disorder in args.disorder:
            values = []
            for _ in range(args.samples):
                energies, states = np.linalg.eigh(
                    hamiltonian(length, args.hopping, disorder, rng)
                )
                center = np.argmin(np.abs(energies))
                values.append(ipr(states[:, center]))
            rows.append((length, disorder, np.mean(values), np.std(values)))

    np.savetxt(
        output / "ipr_near_zero.csv",
        rows,
        delimiter=",",
        header="L,W_over_t,ipr_mean,ipr_std",
        comments="",
    )
    print("Gespeichert: results/ipr_near_zero.csv")


if __name__ == "__main__":
    main()
