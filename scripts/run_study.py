"""Numerische Auswertung der 1D-Anderson-Kette."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from anderson import diagonalize, ipr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", nargs="+", type=int, default=[50, 100, 200])
    parser.add_argument("--disorder", nargs="+", type=float, default=[0, 1, 2, 4])
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--hopping", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if any(length < 1 for length in args.lengths):
        parser.error("alle Laengen muessen positiv sein")
    if any(ratio < 0 for ratio in args.disorder):
        parser.error("W/t darf nicht negativ sein")
    if args.samples < 1 or args.hopping <= 0:
        parser.error("samples und hopping muessen positiv sein")

    rng = np.random.default_rng(args.seed)
    output = args.output
    output.mkdir(exist_ok=True)
    rows = []
    for length in args.lengths:
        for ratio in args.disorder:
            disorder = ratio * args.hopping
            values = []
            for _ in range(args.samples):
                energies, states = diagonalize(length, args.hopping, disorder, rng)
                center = np.argmin(np.abs(energies))
                values.append(ipr(states[:, center]))
            rows.append((length, ratio, np.mean(values), np.std(values)))

    np.savetxt(
        output / "ipr_near_zero.csv",
        rows,
        delimiter=",",
        header="L,W_over_t,ipr_mean,ipr_std",
        comments="",
    )
    if not args.no_plots:
        _plot_ipr_scaling(rows, args.disorder, output)
        _plot_ipr_vs_length(rows, args.disorder, output)
        _plot_ipr_vs_length(rows, args.disorder, output)
        _plot_probabilities(args.lengths[0], args.hopping, args.disorder, args.seed, output)
    print(f"Gespeichert: {output / 'ipr_near_zero.csv'}")


def _plot_ipr_scaling(
    rows: list[tuple[int, float, float, float]],
    disorder_ratios: list[float],
    output: Path,
) -> None:
    data = np.asarray(rows)
    figure, axis = plt.subplots()
    for length in np.unique(data[:, 0]).astype(int):
        selected = data[data[:, 0] == length]
        axis.errorbar(selected[:, 1], selected[:, 2], yerr=selected[:, 3],
                      marker="o", capsize=3, label=f"L={length}")
    axis.set(xlabel="W/t", ylabel="IPR nahe E=0", title="IPR-Skalierung")
    axis.set_xticks(disorder_ratios)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "ipr_near_zero.png", dpi=160)
    plt.close(figure)


def _plot_ipr_vs_length(
    rows: list[tuple[int, float, float, float]],
    disorder_ratios: list[float],
    output: Path,
) -> None:
    data = np.asarray(rows)
    figure, axis = plt.subplots()
    for ratio in disorder_ratios:
        selected = data[np.isclose(data[:, 1], ratio)]
        selected = selected[np.argsort(selected[:, 0])]
        axis.errorbar(selected[:, 0], selected[:, 2], yerr=selected[:, 3],
                      marker="o", capsize=3, label=f"W/t={ratio:g}")
    axis.set(xlabel="Systemgröße L", ylabel="IPR nahe E=0",
             title="Skalierung der IPR mit der Systemgröße")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.grid(alpha=0.25, which="both")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "ipr_vs_length.png", dpi=160)
    plt.close(figure)


def _plot_ipr_vs_length(
    rows: list[tuple[int, float, float, float]],
    disorder_ratios: list[float],
    output: Path,
) -> None:
    data = np.asarray(rows)
    figure, axis = plt.subplots()
    for ratio in disorder_ratios:
        selected = data[np.isclose(data[:, 1], ratio)]
        selected = selected[np.argsort(selected[:, 0])]
        axis.errorbar(selected[:, 0], selected[:, 2], yerr=selected[:, 3],
                      marker="o", capsize=3, label=f"W/t={ratio:g}")
    axis.set(xlabel="Systemgröße L", ylabel="IPR nahe E=0",
             title="Skalierung der IPR mit der Systemgröße")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.grid(alpha=0.25, which="both")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "ipr_vs_length.png", dpi=160)
    plt.close(figure)


def _plot_probabilities(
    length: int,
    hopping: float,
    disorder_ratios: list[float],
    seed: int,
    output: Path,
) -> None:
    figure, axes = plt.subplots(len(disorder_ratios), 1, squeeze=False,
                                figsize=(7, 2.2 * len(disorder_ratios)))
    for row, ratio in zip(axes[:, 0], disorder_ratios):
        energies, states = diagonalize(
            length, hopping, ratio * hopping, np.random.default_rng(seed)
        )
        index = np.argmin(np.abs(energies))
        row.plot(np.arange(1, length + 1), np.abs(states[:, index]) ** 2)
        row.set_ylabel(r"$|\psi_j|^2$")
        row.set_title(f"W/t={ratio:g}, E={energies[index]:.3g}")
        row.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("Platz j")
    figure.tight_layout()
    figure.savefig(output / "probabilities.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
