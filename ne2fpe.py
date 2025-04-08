from scipy import constants
import numpy as np

def ne2fpe(inp, inverse=False):
    """Convert number density [cm^-3] to plasma frequency [Hz] or vice versa."""
    # Constants for plasma frequency calculation
    ep0 = constants.epsilon_0
    m_e = constants.electron_mass
    q_e = constants.elementary_charge
    if not inverse:  # ne (cc) -> ne (SI) -> f_pe
        return 1 / (2 * np.pi) * np.sqrt(inp * 1e6 * q_e**2 / (m_e * ep0))
    else:  # f_pe -> ne (SI -> ne (cc))
        w_pe_peaks = 2 * np.pi * inp
        n_e_peaks = w_pe_peaks**2 * ep0 * m_e / q_e**2
        return n_e_peaks * 1e-6  # Convert to cubic centimeters
    