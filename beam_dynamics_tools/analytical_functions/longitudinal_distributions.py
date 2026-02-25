import numpy as np
from scipy.integrate import quad


def binomial_line_density_lambda(intensity, tau_b, mu, pos: float = 0):
    tau_full = tau_b / bunch_length_ratio_full_to_fwhm(mu) / bunch_length_ratio_fwhm_to_4sigma()

    form_fact_integrad = lambda dt: np.abs(1 - (2 * np.abs(dt - pos) / tau_full) ** 2) ** (mu + 0.5) \
                                    * (np.heaviside(dt - (pos - tau_full / 2), 0) - np.heaviside(
        dt - (pos + tau_full / 2), 0))

    form_fact = quad(form_fact_integrad, pos - tau_full / 2, pos + tau_full / 2)[0]

    return lambda dt: intensity / form_fact * form_fact_integrad(dt)


def binomial_line_density_short_bunch(dt, tau_b, mu, pos, norm):
    tau_full = tau_b / bunch_length_ratio_full_to_fwhm(mu) / bunch_length_ratio_fwhm_to_4sigma()

    form_fact_integrad = lambda dt: np.abs(1 - (2 * np.abs(dt - pos) / tau_full) ** 2) ** (mu + 0.5) \
                                    * (np.heaviside(dt - (pos - tau_full / 2), 0) - np.heaviside(
        dt - (pos + tau_full / 2), 0))

    return norm * form_fact_integrad(dt)


def binomial_line_density_exact_full(dt, tau_full, mu, pos, omega_rf, norm=1):
    _phi = omega_rf * (dt - pos)
    phi_max = (tau_full / 2) * omega_rf

    epsilon_tilde = np.sin(phi_max / 2) ** 2

    infunc = 1 - np.sin(_phi / 2) ** 2 / epsilon_tilde
    infunc[infunc < 0] = 0

    heavi_factor = (np.heaviside(dt - (pos - np.pi / omega_rf), 0)
                    - np.heaviside(dt - (pos + np.pi / omega_rf), 0))

    return norm * np.abs(infunc) ** (mu + 0.5) * heavi_factor


def binomial_line_density_exact_4sigma(dt, tau_b, mu, pos, omega_rf, norm=1):
    tau_fwhm = tau_b / bunch_length_ratio_fwhm_to_4sigma()
    tau_full = bunch_length_exact_fwhm_to_full(tau_fwhm, mu, omega_rf)

    return binomial_line_density_exact_full(dt, tau_full, mu, pos, omega_rf, norm)


def bunch_length_ratio_full_to_fwhm(mu):
    return np.sqrt(1 - np.exp(np.log(0.5) / (mu + 0.5)))


def bunch_length_ratio_fwhm_to_4sigma():
    return 2 / np.sqrt(2 * np.log(2))


def bunch_length_exact_fwhm_to_full(tau_fwhm, mu, omega_rf):
    mu_factor = bunch_length_ratio_full_to_fwhm(mu)
    phi_max = np.arcsin(np.sin(omega_rf * tau_fwhm / 4) / mu_factor)

    return phi_max / omega_rf * 4
