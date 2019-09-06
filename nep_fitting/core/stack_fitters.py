
import numpy as np
from scipy import optimize
from . import models

class MultiaxisFitter(object):
    def __init__(self, multi_axis_handler):
        """

        Parameters
        ----------
        multi_axis_handler : nep_fitting.core.handler.LineProfileHandler
            LineProfile handler.
        """
        self._handler = multi_axis_handler

        self.results = None

        self._base_parameter_indices = {}
        self._fit_result_base_dtype = None  # to be overridden in derived class
        self._ensemble_parameters = None  # to be overridden in derived class

        # flag fit factories where radius is squared in model and negative diameters can/should be returned as positive
        self._squared_radius = False

    def _model_function(self, parameters, distance, ensemble_parameters=None):
        raise NotImplementedError

    def _error_function(self, parameters, distance, data, ensemble_parameters=None):
        raise NotImplementedError

    def _calc_guess(self, line_profile):
        raise NotImplementedError

    def fit_profiles(self, ensemble_parameters):
        """
        Fit all line profiles, enforcing ensemble constraints. When used in conjunction with a multi_ensemble_handler
        ensemble values can be applied to specific profiles.

        Parameters
        ----------
        ensemble_parameter : tuple
            Fit parameter(s) which are to be held constant (and the same) in all profile fits.

        Returns
        -------
        residuals : ndarray
            1D array of concatenated residual arrays for each profile

        """
        n = self._handler.n

        self.results = np.zeros(n, dtype=self._fit_result_dtype)

        all_residuals = []
        for pi in range(n):
            p = self._handler
            guess = self._calc_guess(p)

            (res, cov_x, infodict, mesg, resCode) = optimize.leastsq(self._error_function, guess, args=(p.get_coordinates(), p.get_data(), ensemble_parameter), full_output=1)

            # estimate uncertainties
            residuals = infodict['fvec'] # note that fvec is error function evaluation, or (data - model_function)
            try:
                # calculate residual variance, a.k.a. reduced chi-squared
                residual_variance = np.sum(residuals**2) / (len(p.get_coordinates()) - len(guess))
                # multiply cov by residual variance for estimating parameter variance
                errors = np.sqrt(np.diag(residual_variance * cov_x))
            except TypeError: # cov_x is None for singular matrices -> ~no curvature along at least one dimension
                errors = -1 * np.ones_like(res)
            self.results[pi]['index'] = pi
            if ensemble_parameters:
                self.results[pi]['ensemble_parameter'] = np.atleast_1d(ensemble_parameters).astype('f')

            self.results[pi]['fitResults'] = tuple(res.astype('f'))
            self.results[pi]['fitError'] = tuple(errors.astype('f'))

            # for models where the radius is squared, convergence to negative values is valid, but negative sign is not
            # meaningful and can confuse users unfamiliar with the models
            if self._squared_radius:
                self.results[pi]['fitResults']['diameter'] = np.abs(self.results[pi]['fitResults']['diameter'])


            # mse[pi] = np.mean(residuals**2)
            all_residuals.append(residuals)

        return np.hstack(all_residuals)

    def fit_profiles_mean(self, ensemble_parameters):
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

    def get_pep_guesses(self):
        self.update_peps()
        peps = sorted(self._handler.partial_ensemble_parameters.keys())
        pep_guesses = np.zeros(len(peps))
        for pep_ind, pep in enumerate(peps):
            parameter = pep.split('~')[0]
            indices = self._handler.partial_ensemble_parameters[pep]
            param_guesses = np.zeros(len(indices))
            for pind in indices:
                profile = self._handler.profile_by_index(pind)
                param_guesses[pind] = (self._calc_guess(profile)[self._base_parameter_indices[parameter]])
            pep_guesses[pep_ind] = np.median(param_guesses)
        return pep_guesses

    def update_peps(self):
        """

        Check what partial ensemble parameters the handler is aware of and make sure our dtype matches

        Returns
        -------
        None
        """
        partial_eps = sorted(self._handler.partial_ensemble_parameters.keys())
        # extend the ensemble parameters / errors in the fitresult dtype
        peps = [(p, '<f4') for p in partial_eps]

        self._fit_result_dtype = self._fit_result_base_dtype + [('partial_ensemble_parameters', peps),
                                                                ('partial_ensemble_uncertainty', peps)]
        return len(partial_eps)

    def fit_partial_ensembles(self, ensemble_parameters):
        pep_guesses = self.get_pep_guesses()
        n_params = len(pep_guesses)
        fitpars = optimize.minimize(self.fit_profiles_mean, pep_guesses, args=ensemble_parameters, method='nelder-mead',
                                    options={'xtol': 1e-8, 'disp': True})
        res, cov_x, infodict, mesg, resCode = optimize.leastsq(self.fit_profiles, fitpars.x, args=ensemble_parameters,
                                                               full_output=1, maxfev=600)  # maxfev=1000, xtol=1e-09)#, ftol=1e-9)
        # estimate uncertainties
        residuals = infodict['fvec']  # note that fvec is error function evaluation, or (data - model_function)
        try:
            # calculate residual variance, a.k.a. reduced chi-squared
            residual_variance = np.sum(residuals ** 2) / (len(residuals) - n_params)
            # multiply cov by residual variance for estimating parameter variance
            errors = np.sqrt(np.diag(residual_variance * cov_x))
        except TypeError:  # cov_x is None for singular matrices -> ~no curvature along at least one dimension
            errors = -1 * np.ones_like(res)

        self.results[:]['partial_ensemble_uncertainty'] = np.atleast_1d(errors).astype('f')

        return res


    def ensemble_fit(self, guesses):
        """

        Parameters
        ----------
        guesses : float or list-like of floats
            initial guess for ensemble parameter(s) value

        Returns
        -------
        res : structured ndarray
            fit results for each profile, including the ensemble parameters and ensemble parameters' uncertainty

        """
        n_params = len(guesses)
        if self.update_peps() > 0:
            fitpars = optimize.minimize(self.fit_partial_ensembles, guesses, method='nelder-mead',
                                        options={'xtol': 1e-8, 'disp': True})
            res, cov_x, infodict, mesg, resCode = optimize.leastsq(self.fit_profiles, fitpars.x, full_output=1,
                                                                   maxfev=600)  # maxfev=1000, xtol=1e-09)#, ftol=1e-9)
        else:
            fitpars = optimize.minimize(self.fit_partial_ensembles, guesses, method='nelder-mead',
                                        options={'xtol': 1e-8, 'disp': True})
            res, cov_x, infodict, mesg, resCode = optimize.leastsq(self.fit_profiles, fitpars.x, full_output=1,
                                                                   maxfev=600)  # maxfev=1000, xtol=1e-09)#, ftol=1e-9)
        # estimate uncertainties
        residuals = infodict['fvec']  # note that fvec is error function evaluation, or (data - model_function)
        try:
            # calculate residual variance, a.k.a. reduced chi-squared
            residual_variance = np.sum(residuals ** 2) / (len(residuals) - n_params)
            # multiply cov by residual variance for estimating parameter variance
            errors = np.sqrt(np.diag(residual_variance * cov_x))
        except TypeError:  # cov_x is None for singular matrices -> ~no curvature along at least one dimension
            errors = -1 * np.ones_like(res)

        self.results[:]['ensemble_uncertainty'] = np.atleast_1d(errors).astype('f')

        return res

