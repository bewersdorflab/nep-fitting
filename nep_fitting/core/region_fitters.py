import numpy as np
from scipy import optimize
from scipy.signal import convolve2d

class RegionFitter(object):
    def __init__(self, region_handler):
        self._handler = region_handler
        self.results = None

        self._fit_result_dtype = None  # to be overridden in derived class
        self._ensemble_parameter = None  # to be overridden in derived class

    def _model_function(self, parameters, coordinates, ensemble_parameter=None):
        """
        prototype function to be overridden in derived classes

        Parameters
        ----------
        parameters
        coordinates
        ensemble_parameter

        Returns
        -------

        """
        raise NotImplementedError

    def _error_function(self, parameters, coordinates, data, ensemble_parameter=None):
        """
        prototype function to be overridden in derived classes

        Parameters
        ----------
        parameters
        coordinates
        data
        ensemble_parameter

        Returns
        -------

        """
        raise NotImplementedError

    def _calc_guess(self, roi):
        """
        prototype function to be overridden in derived classes
        Parameters
        ----------
        line_profile

        Returns
        -------

        """
        raise NotImplementedError

    def fit_regions(self, ensemble_parameter=None):
        rois = self._handler.get_rois()
        roi_count = len(rois)
        self.results = np.zeros(roi_count, dtype=self._fit_result_dtype)
        mse = np.zeros(roi_count)
        for ri in range(roi_count):
            r = rois[ri]
            guess = self._calc_guess(r)

            (res, cov_x, infodict, mesg, resCode) = optimize.leastsq(self._error_function, guess,
                                                                 args=(r.get_coordinates(), r.get_data(), ensemble_parameter), full_output=1)

            # estimate uncertainties
            residuals = infodict['fvec'] # note that fvec is error function evaluation, or (data - model_function)
            try:
                # calculate residual variance, a.k.a. reduced chi-squared
                residual_variance = np.sum(residuals**2) / (len(r.get_data()[:]) - len(guess))
                # multiply cov by residual variance for estimating parameter variance
                errors = np.sqrt(np.diag(residual_variance * cov_x))
            except TypeError: # cov_x is None for singular matrices -> ~no curvature along at least one dimension
                errors = -1 * np.ones_like(res)
            self.results[ri]['index'] = ri
            self.results[ri]['ensemble_parameter'] = np.atleast_1d(ensemble_parameter).astype('f')
            self.results[ri]['fitResults'] = res.astype('f')
            self.results[ri]['fitError'] = errors.astype('f')

            mse[ri] = np.mean(residuals**2)

        ensemble_error = mse.mean()

        return ensemble_error

    def ensemble_test(self, test_vals):
        ensemble_error = np.zeros(len(test_vals))
        for fi in range(test_vals):
            ensemble_error[fi] = self.fit_regions(ensemble_parameter=test_vals[fi])

        return ensemble_error

    def ensemble_fit(self, guess):
        fitpars = optimize.minimize(self.fit_regions, guess, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
        self.results
        return fitpars.x

def bead(R, C, beadDiam):
    """
    returns a sum-normalized bead projection
    """
    r0, c0 = np.mean(R), np.mean(C)
    beadRad = 0.5*beadDiam
    beadProj = np.nan_to_num(np.sqrt(beadRad**2 - (R - r0)**2 - (C - c0)**2))
    # return normalized bead projection
    return (1/np.sum(beadProj))*beadProj

def lorentzian(R, C, r0, c0, psf_fwhm):
    #  rough normalization, can't do it analytically for 2D Lorentzian
    rmax = np.sqrt((0.5*(R.max() - R.min()))**2 + ((0.5*(C.max() - C.min()))**2))
    norm = 1. / (np.pi * np.log(1 + rmax**2))
    return (norm) * (0.5 * psf_fwhm) / ((R - r0)**2 + (C - c0)**2 + (0.5 * psf_fwhm)**2)

def lorentz_convolved_bead_ensemblePSF(p, x, psf_fwhm):
    """
    astigmatic Gaussian model function with constant offset (parameter vector [A, x0, y0, sx, sy, b])
    convolved with a bead shape
    """
    A, r0, c0, beadDiam, bgd = p
    R, C = x

    step = R[1, 0] - R[0, 0]
    # note - cannot normalize a 2D lorentzian
    lor = lorentzian(R, C, r0, c0, psf_fwhm)
    # how many steps to include beadDiam
    ns = np.ceil(beadDiam / step) + 1
    beadProj = A * bead(R[:2*ns], C[:2*ns], beadDiam)
    model = convolve2d(lor, beadProj, mode='same') + bgd
    #return model[interpFactor:-interpFactor:interpFactor, interpFactor:-interpFactor:interpFactor]
    return model

def lorentz_convolved_bead_ensemblePSF_ensembleBeadDiam(p, x, ensemble_parameters):
    A, r0, c0, bgd = p
    R, C = x
    psf_fwhm, beadDiam = ensemble_parameters

    step = R[1, 0] - R[0, 0]
    # note - cannot normalize a 2D lorentzian
    lor = lorentzian(R, C, r0, c0, psf_fwhm)
    # how many steps to include beadDiam
    ns = np.ceil(beadDiam / step) + 1
    beadProj = A * bead(R[:2 * ns], C[:2 * ns], beadDiam)
    model = convolve2d(lor, beadProj, mode='same') + bgd
    #return model[interpFactor:-interpFactor:interpFactor, interpFactor:-interpFactor:interpFactor]
    return model


class LorentzianConvolvedSolidSphere_ensemblePSF(RegionFitter):
    def __init__(self, region_handler):
        super(self.__class__, self).__init__(region_handler)

        # [amplitude, bead diameter, center position, background]
        self._fit_result_dtype = [('index', '<i4'),
                                  ('ensemble_parameter', [('psf_fwhm', '<f4')]),
                                  # A, r0, c0, beadDiam, bgd = p
                                  ('fitResults', [('amplitude', '<f4'), ('row_center', '<f4'), ('column_center', '<f4'),
                                                  ('diameter', '<f4'), ('background', '<f4')]),
                                  ('fitError', [('amplitude', '<f4'), ('row_center', '<f4'), ('column_center', '<f4'),
                                                ('diameter', '<f4'), ('background', '<f4')])]

        self._ensemble_parameter = 'PSF FWHM [nm]'

    def _model_function(self, parameters, distance, ensemble_parameter=None):
        return lorentz_convolved_bead_ensemblePSF(parameters, distance, ensemble_parameter)

    def _error_function(self, parameters, x, data, psf_fwhm=None):
        modelFunc = lorentz_convolved_bead_ensemblePSF(parameters, x, psf_fwhm)
        return (data - modelFunc).ravel()

    def _calc_guess(self, roi):
        # [amplitude, row_center, column_center, bead_diameter, background]
        data = roi.get_data()
        R, C = roi.get_coordinates()
        background = data.min()
        amplitude = data.max() - background
        step_size_squared = np.max([R[1, 0] - R[0, 0], C[1, 0] - C[0, 0]]) * np.max([R[0, 1] - R[0, 0], C[0, 1] - C[0, 0]])
        # calculate area above half of the amplitude, then assuming a circular geometry, convert to a radius
        bead_diameter = np.sqrt((step_size_squared / np.pi)*np.sum(data >= background + 0.5 * amplitude))
        max_loc = np.where(data == background + amplitude)

        return amplitude, R[max_loc][0], C[max_loc][0], bead_diameter, background

fitters = {
    'LorentzianConvolvedSolidSphere_ensemblePSF': LorentzianConvolvedSolidSphere_ensemblePSF
}
