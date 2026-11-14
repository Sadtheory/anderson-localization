import numpy as np

from anderson import clean_energies, diagonalize, hamiltonian, ipr, ipr_all


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


def test_diagonalization_returns_normalized_states() -> None:
    energies, states = diagonalize(15, disorder=2.0, rng=np.random.default_rng(4))
    assert energies.shape == (15,)
    np.testing.assert_allclose(states.T @ states, np.eye(15), atol=1e-14)
    np.testing.assert_allclose(ipr_all(states), [ipr(states[:, i]) for i in range(15)])


def test_clean_energies_reject_nonpositive_hopping() -> None:
    with np.testing.assert_raises(ValueError):
        clean_energies(4, hopping=0)
