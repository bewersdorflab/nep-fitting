import numpy as np
#------------------------------------------------ PSF Model Functions ------------------------------------------------#

def lorentzian_psf(x, fwhm):
    """
    Generates a normalized Lorentzian profile of specified fwhm, centered in an array the same length as input x
    Parameters
    ----------
    x : 1darray
        position vector
    fwhm : float
        Full-width at half-maximum of a Lorentzian function

    Returns
    -------
    model : 1darray

    """
    x0 = 0.5 * (np.max(x) - np.min(x))
    return (0.5 * fwhm / np.pi) * (1 / ((x - x0)**2 + (0.5 * fwhm)**2))

def gaussian_psf(x, fwhm):
    """
    Generates a normalized Gaussian profile of specified fwhm, centered in an array the same length as input x
    Parameters
    ----------
    x : 1darray
        position vector
    fwhm : float
        Full-width at half-maximum of a Lorentzian function

    Returns
    -------
    model : 1darray

    """
    x0 = 0.5 * (np.max(x) - np.min(x))
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - x0)**2 / (2 * sigma**2))

#---------------------------------------------- Simple Model Functions -----------------------------------------------#

def naive_lorentzian(p, x):
    amp, fwhm, x0, bkgnd = p
    peak = 2 / (np.pi * fwhm)
    # normed such that the peak will be amp
    return ((amp * 0.5 * fwhm / np.pi) / peak) * (1 / ((x - x0) ** 2 + (0.5 * fwhm) ** 2)) + bkgnd

def naive_gaussian(p, x):
    amp, fwhm, x0, bkgnd = p
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    # normed such that the peak will be amp
    # return amp * (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - x0)**2 / (2 * sigma**2)) + bkgnd
    return amp * np.exp(-(x - x0)**2 / (2 * sigma**2)) + bkgnd

#------------------------------------------- PSF-convolved Model Functions -------------------------------------------#

def lorentz_convolved_annulus(p, distance, psf_fwhm):
    amp, r_inner, center, bkgnd, r_outer = p

    t = distance - center

    neg_rt_inner = np.sqrt(1 + (4 * r_inner ** 2) / (-2j * t + psf_fwhm) ** 2)
    pos_rt_inner = np.sqrt(1 + (4 * r_inner ** 2) / (2j * t + psf_fwhm) ** 2)
    neg_rt_outer = np.sqrt(1 + (4 * r_outer ** 2) / (-2j * t + psf_fwhm) ** 2)
    pos_rt_outer = np.sqrt(1 + (4 * r_outer ** 2) / (2j * t + psf_fwhm) ** 2)

    pre = (0.5 * np.pi / psf_fwhm) * (psf_fwhm / ((np.pi ** 2) * (r_inner - r_outer) * (r_inner + r_outer)))

    peak = (4 / np.pi / psf_fwhm) / (np.sqrt(1 + 4*r_inner**2/psf_fwhm**2) + np.sqrt(1 + 4*r_outer**2/psf_fwhm**2))

    inner = (2j * t) * (-neg_rt_inner + pos_rt_inner) + psf_fwhm * (-2 + neg_rt_inner + pos_rt_inner)
    outer = (2j * t) * (-neg_rt_outer + pos_rt_outer) + psf_fwhm * (-2 + neg_rt_outer + pos_rt_outer)
    model = np.real(pre * (inner - outer))  # clip the (zero) imaginary part, so fitpack doesnt get upset
    return amp * model / peak + bkgnd

def lorentz_convolved_tubule_membrane_antibody(parameters, distance, psf_fwhm):
    amp, d_inner, center, bkgnd = parameters

    r_inner = 0.5 * d_inner
    # 25 nm without antibodies, 60 nm with primary and secondary, see 10.1073/pnas.75.4.1820
    r_outer = r_inner + 17.5  # [nm]

    return lorentz_convolved_annulus([amp, r_inner, center, bkgnd, r_outer], distance, psf_fwhm)

def lorentz_convolved_tubule_membrane_antibody_ne(parameters, distance):
    amp, d_inner, center, bkgnd, psf_fwhm = parameters

    r_inner = 0.5 * d_inner
    # 25 nm without antibodies, 60 nm with primary and secondary, see 10.1073/pnas.75.4.1820
    r_outer = r_inner + 17.5  # [nm]

    return lorentz_convolved_annulus([amp, r_inner, center, bkgnd, r_outer], distance, psf_fwhm)


def lorentz_convolved_microtubule_antibody(parameters, distance, psf_fwhm):
    amp, center, bkgnd = parameters
    # 25 nm without antibodies, 60 nm with primary and secondary, see 10.1073/pnas.75.4.1820
    r_inner = 12.5  # [nm]
    r_outer = r_inner + 17.5  # [nm]

    return lorentz_convolved_annulus([amp, r_inner, center, bkgnd, r_outer], distance, psf_fwhm)

def lorentz_convolved_coated_tubule_selflabeling(parameters, distance, psf_fwhm):
    amp, d_inner, center, bkgnd = parameters

    r_inner = 0.5 * d_inner
    # SNAP diameter ~ 3.6 nm, Halo diameter ~ 4.4 nm, size of Dyes themselves ~ 1 nm, so 4.5 nm offset
    r_outer = r_inner + 4.5  # [nm]

    return lorentz_convolved_annulus([amp, r_inner, center, bkgnd, r_outer], distance, psf_fwhm)

def lorentz_convolved_coated_tubule_selflabeling_ne(parameters, distance):
    amp, d_inner, center, bkgnd, psf_fwhm = parameters

    return lorentz_convolved_coated_tubule_selflabeling(parameters[:-1], distance, psf_fwhm)

def lorentz_convolved_tubule_membrane(p, x, gamma):
    """
    Model function for membrane-labeled tubule imaged with STED. The membrane is assumed to be infinitely thin.

    Parameters
    ----------
    p : array-like
        list of fit parameters [amplitude, tubule diameter, center position, background]
    x : array
        position vector
    gamma : float
        FWHM of Lorentzian PSF

    Returns
    -------
    model : array
        profile of membrane-labeled tubule, projected and convolved with a Lorentzian

    """
    a, d, x0, c = p
    r = d / 2

    topleft = gamma * (np.sqrt(1 + (4 * r ** 2) / (gamma + 2j * (x - x0)) ** 2) + np.sqrt(
        1 + (4 * r ** 2) / (gamma - 2j * x + 2j * x0) ** 2))

    topright = 2j * (np.sqrt(1 + (4 * r ** 2) / (gamma + 2j * (x - x0)) ** 2) - np.sqrt(
        1 + (4 * r ** 2) / (gamma - 2j * x + 2j * x0) ** 2)) * (x - x0)

    denom = (gamma ** 2 + 4 * (x - x0) ** 2) * (
        np.sqrt(1 + (4 * r ** 2) / (gamma - 2j * (x - x0)) ** 2) * np.sqrt(
            1 + (4 * r ** 2) / (gamma + 2j * (x - x0)) ** 2))

    mid = (topleft + topright) / denom

    # scale so maximum value is ~a
    amp = r * a

    return np.real((amp * mid) + c)


def lorentz_convolved_tubule_membrane_misfit(p, x, data, gamma):
    """
    Error function for membrane-labeled tubule imaged with STED

    Parameters
    ----------
    p : array-like
        list of fit parameters [amplitude, tubule diameter, center position, background]
    x : array
        position vector
    gamma : float
        FWHM of Lorentzian PSF

    Returns
    -------
    misfit : array
        difference array of data and model function

    """
    modelFunc = lorentz_convolved_tubule_membrane(p, x, gamma)

    return (data - modelFunc)


def lorentz_convolved_tubule_lumen(p, x, gamma):
    """
    Model function for lumen-labeled tubule imaged with STED. The PSF along z is assumed to be larger than the diameter
    of the tubule.

    Parameters
    ----------
    p : array-like
        list of fit parameters [amplitude, tubule diameter, center position, background]
    x : array
        position vector
    gamma : float
        FWHM of Lorentzian PSF

    Returns
    -------
    model : array
        profile of lumen-labeled tubule, projected and convolved with a Lorentzian

    """
    a, d, x0, c = p
    r = d / 2

    left = gamma * (-2 + np.sqrt(1 + (4 * r ** 2) / (gamma + 2j * (x - x0)) ** 2) + np.sqrt(
        1 + (4 * r ** 2) / (gamma - 2j * x + 2j * x0) ** 2))
    right = 2j * (np.sqrt(1 + (4 * r ** 2) / (gamma + 2j * (x - x0)) ** 2) - np.sqrt(
        1 + (4 * r ** 2) / (gamma - 2j * x + 2j * x0) ** 2)) * (x - x0)

    # scale so maximum value is a
    amp = a / (0.25 * gamma * (-2 + 2 * np.sqrt(1 + (4 * r ** 2) / gamma ** 2)))

    return np.real((amp / 4) * (left + right) + c)


def lorentz_convolved_tubule_lumen_misfit(p, x, data, gamma):
    """
        Error function for lumen-labeled tubule imaged with STED

        Parameters
        ----------
        p : array-like
            list of fit parameters [amplitude, tubule diameter, center position, background]
        x : array
            position vector
        gamma : float
            FWHM of Lorentzian PSF

        Returns
        -------
        misfit : array
            difference array of data and model function

        """
    modelFunc = lorentz_convolved_tubule_lumen(p, x, gamma)

    return (data - modelFunc)

def lorentz_convolved_tubule_lumen_ne(p, x):
    """
    Model function for lumen-labeled tubule imaged with STED. The PSF along z is assumed to be larger than the diameter
    of the tubule.

    Parameters
    ----------
    p : array-like
        list of fit parameters [amplitude, tubule diameter, center position, background]
    x : array
        position vector

    Returns
    -------
    model : array
        profile of lumen-labeled tubule, projected and convolved with a Lorentzian

    """
    a, d, x0, c, psf_fwhm = p

    return lorentz_convolved_tubule_lumen(p[:-1], x, psf_fwhm)