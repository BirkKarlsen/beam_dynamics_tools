'''
Functions to compute longitudinal and RF parameters
'''

import numpy as np
from scipy.constants import c, e, proton_mass
from scipy.integrate import quad
from scipy.optimize import fmin, bisect

E_0 = proton_mass * c ** 2 / e
C = 26658.883
h = 35640
gamma_t = 53.606713
gamma_t_b1 = 53.606713
gamma_t_b2 = 53.674152
f_rf = 400.789e6

# Some useful functions for scalings
synchronous_energy = lambda p_s: np.sqrt(p_s ** 2 + E_0 ** 2)
relativistic_gamma = lambda E_s: E_s / E_0
momentum_compaction = lambda gamma_t, gamma_s: 1 / gamma_t ** 2 - 1 / gamma_s ** 2
relativistic_beta = lambda gamma: np.sqrt(1 - 1 / gamma ** 2)


def single_rf_potential(V, harmonic, gamma_t=53.606713, C=26658.883, p_s=450e9):
    r"""
    Computes the potential well from a single RF system and returns it as a function of time.
    """

    E_s = synchronous_energy(p_s)
    gamma_s = relativistic_gamma(E_s)
    beta = relativistic_beta(gamma_s)
    T_rev = C / (c * beta)
    eta = momentum_compaction(gamma_t, gamma_s)
    omega_rf = 2 * np.pi * harmonic / T_rev

    constant = 2 * beta ** 2 * E_s / (np.abs(eta) * T_rev * omega_rf)

    potential = lambda dt: constant * V * (1 - np.cos(omega_rf * dt))

    return potential


def double_rf_potential(V1, V2, harmonic1, harmonic2, gamma_t=18, C=2 * np.pi * 1100.009, p_s=450e9):
    r"""
    Computes the potential well from a double RF system and returns it as a function of time.
    """

    E_s = synchronous_energy(p_s)
    gamma_s = relativistic_gamma(E_s)
    beta = relativistic_beta(gamma_s)
    T_rev = C / (c * beta)
    eta = momentum_compaction(gamma_t, gamma_s)
    omega_rf1 = 2 * np.pi * harmonic1 / T_rev
    omega_rf2 = 2 * np.pi * harmonic2 / T_rev

    constant = 2 * beta ** 2 * E_s / (abs(eta) * T_rev * omega_rf1)

    potential = lambda dt: constant * (V1 * (1 - np.cos(omega_rf1 * dt)) + 0.25 * V2 * (1 - np.cos(omega_rf2 * dt)))

    return potential


def compute_momentum_spread(bunch_length, potential, p_s=450e9):
    r"""
    Computes the momentum spread from the bunch length for a given potential well function.
    """
    E_s = synchronous_energy(p_s)
    gamma_s = relativistic_gamma(E_s)
    beta = relativistic_beta(gamma_s)

    dE_b = np.sqrt(potential(bunch_length / 2))

    return dE_b / (beta ** 2 * E_s)


def compute_emittance(bunch_length, potential):
    r"""
    Computes the emittance for a given bunch length and potential well function.
    """

    dE_b = np.sqrt(potential(bunch_length / 2))

    integrad = lambda x: np.sqrt(dE_b ** 2 - potential(x))

    return 4 * quad(integrad, 0, bunch_length / 2)[0]


def compute_bucket_height(potential, guess):
    r"""
    Computes the height of the RF bucket in units of energy offset.
    """
    min_pot = lambda x: -potential(x)

    minval = fmin(min_pot, guess)

    return np.sqrt(potential(minval)[0])

def compute_bucket_area(potential, guess, harmonic=35640, C=26658.883, p_s=450e9):
    r"""
    Compute the bucket area of the RF bucket in units of eVs
    """
    omega_rf = 2 * np.pi * compute_rf_frequency(harmonic=harmonic, C=C, p_s=p_s)

    # Find synchronous phase
    min_pot = lambda x: -potential(x)
    minval = fmin(min_pot, guess)

    integrad = lambda x: np.sqrt(potential(omega_rf * np.pi - minval) - potential(x))

    # find turning point
    minval = fmin(lambda x: potential(omega_rf * np.pi - minval) - potential(x), omega_rf * 0.80)

    return 4 * quad(integrad, 0, minval)[0]


def compute_bunch_length(emittance, potential, guess):
    r"""
    Computes the bunch length based on the longitudinal emittance and the potential well.
    """
    f = lambda x: compute_emittance(x, potential) - emittance

    return bisect(f, guess[0], guess[1])


def compute_rf_frequency(harmonic, C=26658.883, p_s=450e9):
    E_s = synchronous_energy(p_s)
    gamma_s = relativistic_gamma(E_s)
    beta = relativistic_beta(gamma_s)
    T_rev = C / (c * beta)

    return harmonic / T_rev


def binomial_distribution(intensity, tau_b, mu, pos: float = 0):
    tau_full = tau_b / bunch_length_ratio_full_to_fwhm(mu) / bunch_length_ratio_fwhm_to_4sigma()

    form_fact_integrad = lambda dt: np.abs(1 - (2 * np.abs(dt - pos) / tau_full) ** 2) ** (mu + 0.5) \
                                    * (np.heaviside(dt - (pos - tau_full / 2), 0) - np.heaviside(
        dt - (pos + tau_full / 2), 0))

    form_fact = quad(form_fact_integrad, pos - tau_full / 2, pos + tau_full / 2)[0]

    return lambda dt: intensity / form_fact * form_fact_integrad(dt)


def rf_beam_current_analytic(line_density: callable, omega_rf, phi_rf, bunch_spacing):
    integrand_real = lambda dt: 2 * e * line_density(dt) * np.cos(omega_rf * dt + phi_rf)
    integrand_imag = lambda dt: -2 * e * line_density(dt) * np.sin(omega_rf * dt + phi_rf)

    lower_lim = 0
    upper_lim = 2 * np.pi / omega_rf

    scal_fact = omega_rf / (2 * np.pi * bunch_spacing)

    real_beam_current = quad(integrand_real, lower_lim, upper_lim)[0] * scal_fact
    imag_beam_current = quad(integrand_imag, lower_lim, upper_lim)[0] * scal_fact

    return real_beam_current + 1j * imag_beam_current


def bunch_length_ratio_full_to_fwhm(mu):
    return np.sqrt(1 - np.exp(np.log(0.5) / (mu + 0.5)))


def bunch_length_ratio_fwhm_to_4sigma():
    return 2 / np.sqrt(2 * np.log(2))