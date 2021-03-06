from scipy import optimize
import numpy as np
from . import models
from .profile_fitters import ProfileFitter

class StackFitter(ProfileFitter):

    def _model_function(self, parameters, positions, ensemble_parameters):
        """
        prototype function to be overridden in derived classes

        Parameters
        ----------
        parameters : tuple
            fit model parameters, e.g. amplitude, diameter, etc.
        positions : tuple
            tuple of 1D position array in units of nanometers
        ensemble_parameters : None, float, or tuple of float
            value(s) which are to be held consistent during fitting across all line profiles

        Returns
        -------
        model : ndarray

        """
        raise NotImplementedError

    def _error_function(self, parameters, positions, profiles, ensemble_parameters):
        """
        prototype function to be overridden in derived classes

        parameters : tuple
            fit model parameters, e.g. amplitude, diameter, etc.
        positions : tuple
            tuple of 1D position array in units of nanometers
        data : tuple
            tuple of 1D line profiles to be fit
        ensemble_parameters : None, float, or tuple of float
            value(s) which are to be held consistent during fitting across all line profiles

        Returns
        -------
        model : ndarray

        """
        raise NotImplementedError

    def _calc_guess(self, profiles):
        """
        prototype function to be overridden in derived classes

        Parameters
        ----------
        profiles : nep_fitting.core.rois.MultiaxisProfile

        Returns
        -------
        guess : tuple
            initial parameters for the fit based on simple metrics of the line profile input

        """
        raise NotImplementedError

    def fit_profiles(self, ensemble_parameters=None):
        """
        Fit all line profiles. If the fit factory is marked "ne" standing for non-ensemble, then all fits will be
        independent of each other. Otherwise, the ensemble_parameter tuple will be held the same for all tubules.

        Parameters
        ----------
        ensemble_parameters : tuple
            Fit parameter(s) which are to be held constant (and the same) in all profile fits.

        Returns
        -------
        residuals : ndarray
            1D array of concatenated residual arrays for each profile

        """
        profiles = self._handler.get_line_profiles()
        profile_count = len(profiles)
        self.results = np.zeros(profile_count, dtype=self._fit_result_dtype)
        # mse = np.zeros(profile_count)
        all_residuals = []
        for pi in range(profile_count):
            p = profiles[pi]
            guess = self._calc_guess(p)
            pos, profs = p.data
            (res, cov_x, infodict, mesg, resCode) = optimize.leastsq(self._error_function, guess, args=(pos, profs, ensemble_parameters), full_output=1)

            # estimate uncertainties
            residuals = infodict['fvec'] # note that fvec is error function evaluation, or (data - model_function)
            try:
                # calculate residual variance, a.k.a. reduced chi-squared
                residual_variance = np.sum(residuals**2) / (np.sum([len(posi) for posi in pos]) - len(guess))
                # multiply cov by residual variance for estimating parameter variance
                errors = np.sqrt(np.diag(residual_variance * cov_x))
            except TypeError: # cov_x is None for singular matrices -> ~no curvature along at least one dimension
                errors = -1 * np.ones_like(res)
            self.results[pi]['index'] = pi
            if ensemble_parameters is not None:
                self.results[pi]['ensemble_parameter'] = tuple(ensemble_parameters)

            self.results[pi]['fitResults'] = tuple(res.astype('f'))
            self.results[pi]['fitError'] = tuple(errors.astype('f'))

            # for models where the radius is squared, convergence to negative values is valid, but negative sign is not
            # meaningful and can confuse users unfamiliar with the models
            if self._squared_radius:
                self.results[pi]['fitResults']['diameter'] = np.abs(self.results[pi]['fitResults']['diameter'])


            # mse[pi] = np.mean(residuals**2)
            all_residuals.append(residuals)

        return np.hstack(all_residuals)

    def fit_profiles_mean(self, ensemble_parameters=None):
        """
        Like fit_profiles, but returns the average mean-squared error of all line profile fits.

        Fit all line profiles. If the fit factory is marked "ne" standing for non-ensemble, then all fits will be
        independent of each other. Otherwise, the ensemble_parameter tuple will be held the same for all tubules.

        Parameters
        ----------
        ensemble_parameter : tuple
            Fit parameter(s) which are to be held constant (and the same) in all profile fits.

        Returns
        -------
        ensemble_error : float
            average of mean-squared error where the average is taken over all line profile fits.

        """
        residuals = self.fit_profiles(ensemble_parameters)
        return np.mean(np.stack([(resi**2).mean() for resi in residuals]))


    def ensemble_test(self, psf_xy, psf_z):
        """
        Test ensemble parameters over an array of input values, returning the average mean-squared error for each test

        Parameters
        ----------
        test_parameters : dict
            keys denote the ensemble parameter to be tested, while the corresponding item contains the array of values
            to test.

        Returns
        -------
        ensemble_error : structured ndarray
            array of average mean-squared error corresponding to the tests

        Notes
        -----
        PYME tabular types don't currently support saving multidimensional structured arrays like ensemble_error. To
        save to hdf, unravel (call ensemble_error.ravel()) before converting to PYME tabular and saving/.

        """
        tDict = dict(self._fit_result_dtype)
        dt = [('ensemble_parameter', tDict['ensemble_parameter']),
              ('fit_mean', tDict['fitResults']),
              ('fit_stddev', tDict['fitResults']),
              ('ensemble_meanMSE', '<f')]

        # num_tests = len(list(test_parameters.items())[0][1])
        ensemble_error = np.zeros((psf_xy.shape[0], psf_z.shape[0]), dtype=dt)

        for xi in range(psf_xy.shape[0]):
            for zi in range(psf_z.shape[0]):
                test_params = (psf_xy[xi], psf_z[zi])

                ensemble_error[xi, zi]['ensemble_meanMSE'] = np.mean(self.fit_profiles(ensemble_parameters=test_params)**2)
                res = self.results
                for ti in tDict['fitResults']:
                    field = ti[0]
                    ensemble_error[xi, zi]['fit_mean'][field] = res['fitResults'][field].mean()
                    ensemble_error[xi, zi]['fit_stddev'][field] = res['fitResults'][field].std()
                for ti in tDict['ensemble_parameter']:
                    field = ti[0]
                    ensemble_error[xi, zi]['ensemble_parameter'][field] = res['ensemble_parameter'][field][0]


        return ensemble_error

    def ensemble_fit(self, guess, return_meanmse=False):
        """

        Parameters
        ----------
        guess : float or list-like of floats
            initial guess for ensemble parameter(s) value

        Returns
        -------
        res : structured ndarray
            fit results for each profile, including the ensemble parameters and ensemble parameters' uncertainty

        """
        try:
            n_params = len(guess)
        except TypeError:
            n_params = 1
        fitpars = optimize.minimize(self.fit_profiles_mean, guess, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
        res, cov_x, infodict, mesg, resCode = optimize.leastsq(self.fit_profiles, fitpars.x, full_output=1, maxfev=600)#maxfev=1000, xtol=1e-09)#, ftol=1e-9)
        # estimate uncertainties
        residuals = infodict['fvec']  # note that fvec is error function evaluation, or (data - model_function)
        try:
            # calculate residual variance, a.k.a. reduced chi-squared
            residual_variance = np.sum(residuals ** 2) / (len(residuals) - n_params)
            # multiply cov by residual variance for estimating parameter variance
            errors = np.sqrt(np.diag(residual_variance * cov_x))
        except TypeError:  # cov_x is None for singular matrices -> ~no curvature along at least one dimension
            errors = -1 * np.ones_like(res)

        self.results[:]['ensemble_uncertainty'] = tuple(errors)

        if return_meanmse:
            return res, fitpars.fun
        return res

    def plot_results(self, plot_dir, x_bounds=None, y_bounds=None):
        """

        Parameters
        ----------
        plot_dir : str
            path to directory where pdfs of plots should be saved
        x_bounds : dict
            item-value pairs for x bounds in order to overwrite default pyplot auto-settings. Should be of the form
            {'xmin': float, 'xmax': float}
        y_bounds : dict
            item-value pairs to y bounds in order to overwrite default pyplot auto-settings. Should be of the form
            {'ymin': foat, 'ymax': float}

        Returns
        -------

        """
        import matplotlib.pyplot as plt

        res = self.results

        fig = plt.figure()
        for ind in range(self._handler.n):
            positions, profiles = self._handler.profile_by_index(ind).data
            interpolated_coords = [np.linspace(positions[ii].min(), positions[ii].max(), len(positions[ii])*10) for ii in range(len(positions))]
            for pi, prof in enumerate(profiles):
                plt.scatter(positions[pi], prof, label='Cross-section')
                # try:
                plt.plot(interpolated_coords[pi], self._model_function(res[ind]['fitResults'].tolist(),
                                                                       interpolated_coords,
                                                                       res[ind]['ensemble_parameter'].tolist())[pi],
                         label='Fit')
                # except ValueError:  # if this is a non-ensemble fit
                #     plt.plot(interpolated_coords[pi], self._model_function(res[ind]['fitResults'], interpolated_coords)[pi],
                #              color=colors[1], label='Fit')
            if x_bounds:
                plt.xlim(**x_bounds)
            if y_bounds:
                plt.ylim(**y_bounds)
            plt.xlabel('Position [nm]', fontsize=26)
            plt.ylabel('Amplitude [ADU]', fontsize=26)
            plt.title('Fitted diameter: %.2f +/- %.2f nm'  % (res[ind]['fitResults']['diameter'], res[ind]['fitError']['diameter']))
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_dir + '%s.pdf' % str(ind))
            fig.clf()

class TwoAxisNonEnsembleBase(StackFitter):
    _fit_result_dtype = [('index', '<i4'),
                         ('fitResults', [('amplitude_xy', '<f4'), ('amplitude_z', '<f4'),
                                         ('fwhm_xy', '<f4'), ('fwhm_z', '<f4'),
                                         ('center_xy', '<f4'), ('center_z', '<f4'),
                                         ('background_xy', '<f4'), ('background_z', '<f4')]),
                         ('fitError', [('amplitude_xy', '<f4'), ('amplitude_z', '<f4'),
                                       ('fwhm_xy', '<f4'), ('fwhm_z', '<f4'),
                                       ('center_xy', '<f4'), ('center_z', '<f4'),
                                       ('background_xy', '<f4'), ('background_z', '<f4')])]

    _ensemble_parameters = None

    def __init__(self, profile_handler):
        StackFitter.__init__(self, profile_handler)

    def _error_function(self, parameters, positions, profiles, ensemble_parameters=None):
        model = self._model_function(parameters, positions, ensemble_parameters)
        return np.concatenate([profiles[0] - model[0], profiles[1] - model[1]])

    def _calc_guess(self, multiaxis_profile):
        # [amplitude_xy, amplitude_z, fwhm_xy, fwhm_z, center_xy, center_z, background_xy, background_z]
        positions, profiles = multiaxis_profile.data
        background = profiles[0].min(), profiles[1].min()
        peak = profiles[0].max(), profiles[1].max()
        amp_xy = peak[0] - background[0]
        amp_z = peak[1] - background[1]
        center_xy = positions[0][np.where(peak[0] == profiles[0])[0][0]]
        center_z = positions[1][np.where(peak[1] == profiles[1])[0][0]]
        fwhm_xy = np.sum(profiles[0] >= background[0] + 0.5 * amp_xy) * abs(positions[0][1] - positions[0][0])
        fwhm_z = np.sum(profiles[1] >= background[1] + 0.5 * amp_z) * abs(positions[1][1] - positions[1][0])
        return amp_xy, amp_z, fwhm_xy, fwhm_z, center_xy, center_z, background[0], background[1]

    def plot_results(self, plot_dir, x_bounds=None, y_bounds=None):
        """

        Parameters
        ----------
        plot_dir : str
            path to directory where pdfs of plots should be saved
        x_bounds : dict
            item-value pairs for x bounds in order to overwrite default pyplot auto-settings. Should be of the form
            {'xmin': float, 'xmax': float}
        y_bounds : dict
            item-value pairs to y bounds in order to overwrite default pyplot auto-settings. Should be of the form
            {'ymin': foat, 'ymax': float}

        Returns
        -------

        """
        import matplotlib.pyplot as plt

        res = self.results

        fig = plt.figure()
        for ind in range(self._handler.n):
            positions, profiles = self._handler.profile_by_index(ind).data
            interpolated_coords = [np.linspace(positions[ii].min(), positions[ii].max(), len(positions[ii])*10) for ii in range(len(positions))]
            for pi, prof in enumerate(profiles):
                plt.scatter(positions[pi], prof, label='Cross-section')
                # try:
                plt.plot(interpolated_coords[pi],
                         self._model_function(res[ind]['fitResults'].tolist(), interpolated_coords, None)[pi],
                         label='Fit')
            if x_bounds:
                plt.xlim(**x_bounds)
            if y_bounds:
                plt.ylim(**y_bounds)
            plt.xlabel('Position [nm]', fontsize=26)
            plt.ylabel('Amplitude [ADU]', fontsize=26)
            plt.title('XY FWHM: %.2f nm, Z FWHM: %.2f nm'  % (res[ind]['fitResults']['fwhm_xy'],
                                                              res[ind]['fitResults']['fwhm_z']))
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_dir + '%s.pdf' % str(ind))
            fig.clf()

class TwoAxisBasicLorentzian(TwoAxisNonEnsembleBase):
    def _model_function(self, parameters, positions, ensemble_parameters=None):
        # [amplitude_xy, amplitude_z, fwhm_xy, fwhm_z, center_xy, center_z, background_xy, background_z] -> [amp, fwhm, x0, bkgnd]
        xy_pars = np.delete(parameters, [1, 3, 5, 7])  # remove z-specifics
        z_pars = np.delete(parameters, [0, 2, 4, 6])  # remove x-specifics

        xy = models.naive_lorentzian(xy_pars, positions[0])
        z = models.naive_lorentzian(z_pars, positions[1])

        return xy, z

class TwoAxisBasicGaussian(TwoAxisNonEnsembleBase):
    def _model_function(self, parameters, positions, ensemble_parameters=None):
        # [amplitude_xy, amplitude_z, fwhm_xy, fwhm_z, center_xy, center_z, background_xy, background_z] -> [amp, fwhm, x0, bkgnd]
        xy_pars = np.delete(parameters, [1, 3, 5, 7])  # remove z-specifics
        z_pars = np.delete(parameters, [0, 2, 4, 6])  # remove x-specifics

        xy = models.naive_gaussian(xy_pars, positions[0])
        z = models.naive_gaussian(z_pars, positions[1])

        return xy, z

class TwoAxisEnsembleBase(StackFitter):
    _fit_result_dtype = [('index', '<i4'),
                         ('ensemble_parameter', [('psf_fwhm_xy', '<f4'), ('psf_fwhm_z', '<f4')]),
                         ('ensemble_uncertainty', [('psf_fwhm_xy', '<f4'), ('psf_fwhm_z', '<f4')]),
                         ('fitResults', [('amplitude_xy', '<f4'),
                                         ('amplitude_z', '<f4'),  # fixme - should be unnecessary
                                         ('diameter', '<f4'),
                                         ('center_xy', '<f4'), ('center_z', '<f4'),
                                         ('background_xy', '<f4'),
                                         ('background_z', '<f4')]),  # fixme - should be unnecessary
                         ('fitError', [('amplitude_xy', '<f4'),
                                       ('amplitude_z', '<f4'),  # fixme - should be unnecessary
                                       ('diameter', '<f4'),
                                       ('center_xy', '<f4'), ('center_z', '<f4'),
                                       ('background_xy', '<f4'),
                                       ('background_z', '<f4')])]  # fixme - should be unnecessary

    _ensemble_parameters = ['XY PSF FWHM [nm]', 'Z PSF FWHM [nm]']

    def __init__(self, profile_handler):
        StackFitter.__init__(self, profile_handler)

    def _error_function(self, parameters, positions, profiles, ensemble_parameters):
        model = self._model_function(parameters, positions, ensemble_parameters)
        return np.concatenate([profiles[0] - model[0], profiles[1] - model[1]])

    def _calc_guess(self, multiaxis_profile):
        # [amplitude_xy, amplitude_z, tubule diameter, center_xy, center_z, background_xy, background_z]
        positions, profiles = multiaxis_profile.data
        background = profiles[0].min(), profiles[1].min()
        peak = profiles[0].max(), profiles[1].max()
        amp_xy = peak[0] - background[0]
        amp_z = peak[1] - background[1]
        center_xy = positions[0][np.where(peak[0] == profiles[0])[0][0]]
        center_z = positions[1][np.where(peak[1] == profiles[1])[0][0]]
        d_xy = np.sum(profiles[0] >= background[0] + 0.5 * amp_xy) * abs(positions[0][1] - positions[0][0])
        d_z = np.sum(profiles[1] >= background[1] + 0.5 * amp_z) * abs(positions[1][1] - positions[1][0])
        tubule_diameter = 0.5 * (d_xy + d_z)
        return amp_xy, amp_z, tubule_diameter, center_xy, center_z, background[0], background[1]

class TwoAxisEnsembleBaseZTilt(TwoAxisEnsembleBase):
    """

    """
    _fit_result_dtype = [('index', '<i4'),
                         ('ensemble_parameter', [('psf_fwhm_xy', '<f4'), ('psf_fwhm_z', '<f4')]),
                         ('ensemble_uncertainty', [('psf_fwhm_xy', '<f4'), ('psf_fwhm_z', '<f4')]),
                         ('fitResults', [('amplitude_xy', '<f4'),
                                         ('amplitude_z', '<f4'),  # fixme - should be unnecessary
                                         ('diameter', '<f4'),
                                         ('center_xy', '<f4'), ('center_z', '<f4'),
                                         ('background_xy', '<f4'),
                                         ('background_z', '<f4'),  # fixme - should be unnecessary
                                         ('background_tilt_z', '<f4')]),
                         ('fitError', [('amplitude_xy', '<f4'),
                                       ('amplitude_z', '<f4'),  # fixme - should be unnecessary
                                       ('diameter', '<f4'),
                                       ('center_xy', '<f4'), ('center_z', '<f4'),
                                       ('background_xy', '<f4'),
                                       ('background_z', '<f4'),  # fixme - should be unnecessary
                                       ('background_tilt_z', '<f4')])]

    def _calc_guess(self, multiaxis_profile):
        # [amplitude_xy, amplitude_z, tubule diameter, center_xy, center_z, background_xy, background_z]
        return (TwoAxisEnsembleBase._calc_guess(self, multiaxis_profile)) + tuple([0])

class TwoAxisEnsembleBaseTilt(TwoAxisEnsembleBase):
    """

    """
    _fit_result_dtype = [('index', '<i4'),
                         ('ensemble_parameter', [('psf_fwhm_xy', '<f4'), ('psf_fwhm_z', '<f4')]),
                         ('ensemble_uncertainty', [('psf_fwhm_xy', '<f4'), ('psf_fwhm_z', '<f4')]),
                         ('fitResults', [('amplitude_xy', '<f4'),
                                         ('amplitude_z', '<f4'),  # fixme - should be unnecessary
                                         ('diameter', '<f4'),
                                         ('center_xy', '<f4'), ('center_z', '<f4'),
                                         ('background_xy', '<f4'),
                                         ('background_z', '<f4'),  # fixme - should be unnecessary
                                         ('background_tilt_xy', '<f4'),
                                         ('background_tilt_z', '<f4')]),
                         ('fitError', [('amplitude_xy', '<f4'),
                                       ('amplitude_z', '<f4'),  # fixme - should be unnecessary
                                       ('diameter', '<f4'),
                                       ('center_xy', '<f4'), ('center_z', '<f4'),
                                       ('background_xy', '<f4'),
                                       ('background_z', '<f4'),  # fixme - should be unnecessary
                                       ('background_tilt_xy', '<f4'),
                                       ('background_tilt_z', '<f4')])]

    def _calc_guess(self, multiaxis_profile):
        # [amplitude_xy, amplitude_z, tubule diameter, center_xy, center_z, background_xy, background_z,
        # background_tilt_xy, background_tilt_z]
        return (TwoAxisEnsembleBase._calc_guess(self, multiaxis_profile)) + tuple([0, 0])


class TwoAxisLorentzSelfLabeling(TwoAxisEnsembleBase):
    """
    This is for use with SNAP-tag and Halo tag labels, which result in an annulus of roughly 5 nm thickness
    """
    _squared_radius = True

    def _model_function(self, parameters, positions, ensemble_parameters):
        # [amplitude_xy, amplitude_z, tubule diameter, center_xy, center_z, bgnd_xy, bgnd_z] -> [amp, d_inner, center, bkgnd]
        xy_pars = np.delete(parameters, [1, 4, 6])  # remove z-specifics
        z_pars = np.delete(parameters, [0, 3, 5])  # remove x-specifics

        xy_psf_fwhm, z_psf_fwhm = ensemble_parameters
        # parameters, distance, psf_fwhm
        xy = models.lorentz_convolved_coated_tubule_selflabeling(xy_pars, positions[0], xy_psf_fwhm)
        z = models.lorentz_convolved_coated_tubule_selflabeling(z_pars, positions[1], z_psf_fwhm)

        return xy, z

class TwoAxisLorentzFilled(TwoAxisEnsembleBase):
    """
    """
    _squared_radius = True

    def _model_function(self, parameters, positions, ensemble_parameters):
        # [amplitude_xy, amplitude_z, tubule diameter, center_xy, center_z, bgnd_xy, bgnd_z] -> [amp, d_inner, center, bkgnd]
        xy_pars = np.delete(parameters, [1, 4, 6])  # remove z-specifics
        z_pars = np.delete(parameters, [0, 3, 5])  # remove x-specifics

        xy_psf_fwhm, z_psf_fwhm = ensemble_parameters
        # parameters, distance, psf_fwhm
        xy = models.lorentz_convolved_tubule_lumen(xy_pars, positions[0], xy_psf_fwhm)
        z = models.lorentz_convolved_tubule_lumen(z_pars, positions[1], z_psf_fwhm)

        return xy, z


class TwoAxisLorentzSelfLabelingZTilt(TwoAxisEnsembleBaseZTilt):
    """
    This is for use with SNAP-tag and Halo tag labels, which result in an annulus of roughly 5 nm thickness
    """
    _squared_radius = True

    def _model_function(self, parameters, positions, ensemble_parameters):
        # [amplitude_xy, amplitude_z, tubule diameter, center_xy, center_z, bgnd_xy, bgnd_z, tiltz] -> [amp, d_inner, center, bkgnd]
        xy_pars = np.delete(parameters, [1, 4, 6, 7])  # remove z-specifics
        # [amplitude_xy, amplitude_z, tubule diameter, center_xy, center_z, bgnd_xy, bgnd_z, tiltz] -> [amp, d_inner, center, bkgnd, bx]
        z_pars = np.delete(parameters, [0, 3, 5])  # remove x-specifics
        #

        xy_psf_fwhm, z_psf_fwhm = ensemble_parameters
        # parameters, distance, psf_fwhm
        xy = models.lorentz_convolved_coated_tubule_selflabeling(xy_pars, positions[0], xy_psf_fwhm)
        z = models.lorentz_convolved_coated_tubule_selflabeling_tilt(z_pars, positions[1], z_psf_fwhm)

        return xy, z


class TwoAxisLorentzFilledZTilt(TwoAxisEnsembleBaseZTilt):
    """
    """
    _squared_radius = True

    def _model_function(self, parameters, positions, ensemble_parameters):
        # [amplitude_xy, amplitude_z, tubule diameter, center_xy, center_z, bgnd_xy, bgnd_z, tiltz] -> [amp, d_inner, center, bkgnd]
        xy_pars = np.delete(parameters, [1, 4, 6, 7])  # remove z-specifics
        # [amplitude_xy, amplitude_z, tubule diameter, center_xy, center_z, bgnd_xy, bgnd_z, tiltz] -> [amp, d_inner, center, bkgnd, bx]
        z_pars = np.delete(parameters, [0, 3, 5])  # remove x-specifics
        #

        xy_psf_fwhm, z_psf_fwhm = ensemble_parameters
        # parameters, distance, psf_fwhm
        xy = models.lorentz_convolved_tubule_lumen(xy_pars, positions[0], xy_psf_fwhm)
        z = models.lorentz_convolved_tubule_lumen_tilt(z_pars, positions[1], z_psf_fwhm)

        return xy, z

class TwoAxisLorentzFilledTilt(TwoAxisEnsembleBaseTilt):
    """
    """
    _squared_radius = True

    def _model_function(self, parameters, positions, ensemble_parameters):
        # [amplitude_xy, amplitude_z, tubule diameter, center_xy, center_z, bgnd_xy, bgnd_z, tiltxy, tiltz] -> [amp, d_inner, center, bkgnd]
        xy_pars = np.delete(parameters, [1, 4, 6, 8])  # remove z-specifics
        # [amplitude_xy, amplitude_z, tubule diameter, center_xy, center_z, bgnd_xy, bgnd_z, tiltxy, tiltz] -> [amp, d_inner, center, bkgnd, bx]
        z_pars = np.delete(parameters, [0, 3, 5, 7])  # remove x-specifics
        #

        xy_psf_fwhm, z_psf_fwhm = ensemble_parameters
        # parameters, distance, psf_fwhm
        xy = models.lorentz_convolved_tubule_lumen_tilt(xy_pars, positions[0], xy_psf_fwhm)
        z = models.lorentz_convolved_tubule_lumen_tilt(z_pars, positions[1], z_psf_fwhm)

        return xy, z


class TwoAxisGaussSelfLabeling(TwoAxisEnsembleBase):
    """
    This is for use with SNAP-tag and Halo tag labels, which result in an annulus of roughly 5 nm thickness
    """
    _squared_radius = False

    def _model_function(self, parameters, positions, ensemble_parameters):
        # [amplitude_xy, amplitude_z, tubule diameter, center_xy, center_z, bgnd_xy, bgnd_z] -> [amp, d_inner, center, bkgnd]
        xy_pars = np.delete(parameters, [1, 4, 6])  # remove z-specifics
        z_pars = np.delete(parameters, [0, 3, 5])  # remove x-specifics

        xy_psf_fwhm, z_psf_fwhm = ensemble_parameters
        # parameters, distance, psf_fwhm
        xy = models.gauss_convolved_coated_tubule_selflabeling(xy_pars, positions[0], xy_psf_fwhm)
        z = models.gauss_convolved_coated_tubule_selflabeling(z_pars, positions[1], z_psf_fwhm)

        return xy, z

class TwoAxisGaussFilled(TwoAxisEnsembleBase):
    """
    """
    _squared_radius = False

    def _model_function(self, parameters, positions, ensemble_parameters):
        # [amplitude_xy, amplitude_z, tubule diameter, center_xy, center_z, bgnd_xy, bgnd_z] -> [amp, d_inner, center, bkgnd]
        xy_pars = np.delete(parameters, [1, 4, 6])  # remove z-specifics
        z_pars = np.delete(parameters, [0, 3, 5])  # remove x-specifics

        xy_psf_fwhm, z_psf_fwhm = ensemble_parameters
        # parameters, distance, psf_fwhm
        xy = models.gauss_convolved_tubule_lumen_approx(xy_pars, positions[0], xy_psf_fwhm)
        z = models.gauss_convolved_tubule_lumen_approx(z_pars, positions[1], z_psf_fwhm)

        return xy, z


class TwoAxisGaussSelfLabelingZTilt(TwoAxisEnsembleBaseZTilt):
    """
    This is for use with SNAP-tag and Halo tag labels, which result in an annulus of roughly 5 nm thickness
    """
    _squared_radius = False

    def _model_function(self, parameters, positions, ensemble_parameters):
        # [amplitude_xy, amplitude_z, tubule diameter, center_xy, center_z, bgnd_xy, bgnd_z, tiltz] -> [amp, d_inner, center, bkgnd]
        xy_pars = np.delete(parameters, [1, 4, 6, 7])  # remove z-specifics
        # [amplitude_xy, amplitude_z, tubule diameter, center_xy, center_z, bgnd_xy, bgnd_z, tiltz] -> [amp, d_inner, center, bkgnd, bx]
        z_pars = np.delete(parameters, [0, 3, 5])  # remove x-specifics
        #

        xy_psf_fwhm, z_psf_fwhm = ensemble_parameters
        # parameters, distance, psf_fwhm
        xy = models.gauss_convolved_coated_tubule_selflabeling(xy_pars, positions[0], xy_psf_fwhm)
        z = models.gauss_convolved_coated_tubule_selflabeling_tilt(z_pars, positions[1], z_psf_fwhm)

        return xy, z


class TwoAxisGaussFilledZTilt(TwoAxisEnsembleBaseZTilt):
    """
    """
    _squared_radius = True

    def _model_function(self, parameters, positions, ensemble_parameters):
        # [amplitude_xy, amplitude_z, tubule diameter, center_xy, center_z, bgnd_xy, bgnd_z, tiltz] -> [amp, d_inner, center, bkgnd]
        xy_pars = np.delete(parameters, [1, 4, 6, 7])  # remove z-specifics
        # [amplitude_xy, amplitude_z, tubule diameter, center_xy, center_z, bgnd_xy, bgnd_z, tiltz] -> [amp, d_inner, center, bkgnd, bx]
        z_pars = np.delete(parameters, [0, 3, 5])  # remove x-specifics
        #

        xy_psf_fwhm, z_psf_fwhm = ensemble_parameters
        # parameters, distance, psf_fwhm
        xy = models.gauss_convolved_tubule_lumen_approx(xy_pars, positions[0], xy_psf_fwhm)
        z = models.gauss_convolved_tubule_lumen_approx_tilt(z_pars, positions[1], z_psf_fwhm)

        return xy, z