import numpy as np

from anderson import clean_energies, hamiltonian, ipr


def test_clean_spectrum_matches_exact_solution() -> None:
    length = 12
    actual = np.linalg.eigvalsh(hamiltonian(length, disorder=0.0))
    expected = clean_energies(length)
    np.testing.assert_allclose(actual, expected)


def test_ipr_limits() -> None:
    assert ipr(np.ones(100)) == np.float64(0.01)
    localized = np.zeros(100)
    localized[37] = 1
    assert ipr(localized) == 1.0
