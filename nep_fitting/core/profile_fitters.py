import numpy as np
from scipy import optimize

from .models import *

class ProfileFitter(object):
    def __init__(self, line_profile_handler):
        self._handler = line_profile_handler
        # if any([lp.get_data() is None for lp in self._handler.get_line_profiles()]):
        # self._handler.calculate_profiles()

        self.results = None

        self._fit_result_dtype = None  # to be overridden in derived class
        self._ensemble_parameter = None  # to be overridden in derived class

        # flag fit factories where radius is squared in model and negative diameters can/should be returned as positive
        self._squared_radius = False

    def _model_function(self, parameters, distance, ensemble_parameter=None):
        """
        prototype function to be overridden in derived classes

        Parameters
        ----------
        parameters : tuple
            fit model parameters, e.g. amplitude, diameter, etc.
        distance : ndarray
            1D position array in units of nanometers
        ensemble_parameter : float, or tuple of float
            value(s) which are to be held consistent during fitting across all line profiles

        Returns
        -------
        model : ndarray

        """
        raise NotImplementedError

    def _error_function(self, parameters, distance, data, ensemble_parameter=None):
        """
        prototype function to be overridden in derived classes

        parameters : tuple
            fit model parameters, e.g. amplitude, diameter, etc.
        distance : ndarray
            1D position array in units of nanometers
        data : ndarray
            1D line profile to be fit
        ensemble_parameter : float, or tuple of float
            value(s) which are to be held consistent during fitting across all line profiles

        Returns
        -------
        model : ndarray

        """
        raise NotImplementedError

    def _calc_guess(self, line_profile):
        """
        prototype function to be overridden in derived classes

        Parameters
        ----------
        line_profile : nep-fitting.core.rois.LineProfile

        Returns
        -------
        guess : tuple
            initial parameters for the fit based on simple metrics of the line profile input

        """
        raise NotImplementedError

    def fit_profiles(self, ensemble_parameter=None):
        """
        Fit all line profiles. If the fit factory is marked "ne" standing for non-ensemble, then all fits will be
        independent of each other. Otherwise, the ensemble_parameter tuple will be held the same for all tubules.

        Parameters
        ----------
        ensemble_parameter : tuple
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
            if ensemble_parameter:
                self.results[pi]['ensemble_parameter'] = np.atleast_1d(ensemble_parameter).astype('f')

            self.results[pi]['fitResults'] = res.astype('f')
            self.results[pi]['fitError'] = errors.astype('f')

            # for models where the radius is squared, convergence to negative values is valid, but negative sign is not
            # meaningful and can confuse users unfamiliar with the models
            if self._squared_radius:
                self.results[pi]['fitResults']['diameter'] = np.abs(self.results[pi]['fitResults']['diameter'])


            # mse[pi] = np.mean(residuals**2)
            all_residuals.append(residuals)

        return np.hstack(all_residuals)

    def fit_profiles_mean(self, ensemble_parameter=None):
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
        profiles = self._handler.get_line_profiles()
        profile_count = len(profiles)
        self.results = np.zeros(profile_count, dtype=self._fit_result_dtype)
        mse = np.zeros(profile_count)

        for pi in range(profile_count):
            p = profiles[pi]
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
            if ensemble_parameter:
                self.results[pi]['ensemble_parameter'] = np.atleast_1d(ensemble_parameter).astype('f')

            self.results[pi]['fitResults'] = res.astype('f')
            self.results[pi]['fitError'] = errors.astype('f')

            # for models where the radius is squared, convergence to negative values is valid, but negative sign is not
            # meaningful and can confuse users unfamiliar with the models
            if self._squared_radius:
                self.results[pi]['fitResults']['diameter'] = np.abs(self.results[pi]['fitResults']['diameter'])


            mse[pi] = np.mean(residuals**2)


        ensemble_error = mse.mean()

        return ensemble_error


    def ensemble_test(self, test_parameters):
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

        """
        tDict = dict(self._fit_result_dtype)
        dt = [('ensemble_parameter', tDict['ensemble_parameter']),
              ('fit_mean', tDict['fitResults']),
              ('fit_stddev', tDict['fitResults']),
              ('ensemble_meanMSE', '<f')]

        num_tests = len(test_parameters.items()[0][1])
        ensemble_error = np.zeros(num_tests, dtype=dt)
        try:
            for fi in range(num_tests):
                ensemble_error[fi]['ensemble_meanMSE'] = np.mean(self.fit_profiles(ensemble_parameter=test_parameters[fi])**2)
                res = self.results
                for ti in tDict['fitResults']:
                    field = ti[0]
                    ensemble_error[fi]['fit_mean'][field] = res['fitResults'][field].mean()
                    ensemble_error[fi]['fit_stddev'][field] = res['fitResults'][field].std()
                for ti in tDict['ensemble_parameter']:
                    field = ti[0]
                    ensemble_error[fi]['ensemble_parameter'][field] = res['ensemble_parameter'][field][0]
        except KeyError:  # TODO - remove flexible input, only accept test_parameters as dict
            keys = test_parameters.keys()
            test_params = zip(*[test_parameters[key] for key in keys])
            for fi in range(num_tests):
                ensemble_error[fi]['ensemble_meanMSE'] = np.mean(self.fit_profiles(ensemble_parameter=test_params[fi])**2)
                res = self.results
                for ti in tDict['fitResults']:
                    field = ti[0]
                    ensemble_error[fi]['fit_mean'][field] = res['fitResults'][field].mean()
                    ensemble_error[fi]['fit_stddev'][field] = res['fitResults'][field].std()
                for ti in tDict['ensemble_parameter']:
                    field = ti[0]
                    ensemble_error[fi]['ensemble_parameter'][field] = res['ensemble_parameter'][field][0]


        return ensemble_error

    def ensemble_fit(self, guess):
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

        self.results[:]['ensemble_uncertainty'] = np.atleast_1d(errors).astype('f')

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
        colors = [u'#4878cf', u'#6acc65', u'#d65f5f', u'#b47cc7', u'#c4ad66', u'#77bedb']

        res = self.results
        # if not np.any(res):
        #     return

        profs = self._handler.get_line_profiles()

        num_profs = len(profs)
        fig = plt.figure()
        for ind in range(num_profs):
            position = profs[ind].get_coordinates()
            plt.scatter(position, profs[ind].get_data(), color=colors[0], label='Cross-section')
            interpolated_coords = np.linspace(position.min(), position.max(), len(position)*10)
            try:
                plt.plot(interpolated_coords, self._model_function(res[ind]['fitResults'], interpolated_coords, res[ind]['ensemble_parameter']),
                         color=colors[1], label='Fit')
            except ValueError:  # if this is a non-ensemble fit
                plt.plot(interpolated_coords, self._model_function(res[ind]['fitResults'], interpolated_coords),
                         color=colors[1], label='Fit')
            if x_bounds:
                plt.xlim(**x_bounds)
            if y_bounds:
                plt.ylim(**y_bounds)
            plt.xlabel('Position [nm]', fontsize=26)
            plt.ylabel('Amplitude [ADU]', fontsize=26)
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_dir + '%s.pdf' % str(ind))
            fig.clf()

    def generate_heatmap(self, res_dir, ensemble_parameter={'psf_fwhm': np.arange(35, 105, 5)}):
        # fixme - should be more general or overridden in derived class
        import matplotlib.pyplot as plt
        psfs = ensemble_parameter['psf_fwhm']
        diameter_bins = np.arange(0, 240, 10)
        num_tests = len(psfs)
        #ens_err = np.zeros(num_tests)
        heat_map = np.zeros((num_tests, len(diameter_bins) - 1))
        for ii in range(num_tests):
            #ens_err[ii] =
            self.fit_profiles(psfs[ii])
            heat_map[ii, :], bins, patches = plt.hist(np.abs(self.results['fitResults']['diameter']), bins=diameter_bins)

        contained = np.sum(heat_map.sum(axis=1))
        expected_counts = len(self.results)*num_tests
        if contained != expected_counts:
            print('missing %i profiles from the heatmap because they fell outside of the diameter bins' % (expected_counts - contained))

        fig, ax = plt.subplots()
        cax = ax.imshow(heat_map, interpolation='nearest', origin='lower',
                        extent=[diameter_bins[0], diameter_bins[-1], psfs[0], psfs[-1]])
        plt.xlabel('Tubule Diameter [nm]', size=26)
        plt.ylabel('PSF FWHM [nm]', size=26)
        cbar = fig.colorbar(cax, ticks=[heat_map.min(), heat_map.max()])
        plt.tight_layout()
        plt.savefig(res_dir + 'heat_map.pdf')


#----------------------------------------------- Naive Fitters --------------------------------------------------------#


class Gaussian(ProfileFitter):
    def __init__(self, line_profile_handler):
        super(self.__class__, self).__init__(line_profile_handler)

        # [amplitude, psf_fwhm, center position, background]]
        self._fit_result_dtype = [('index', '<i4'),
                  ('fitResults', [('amplitude', '<f4'), ('psf_fwhm', '<f4'), ('center', '<f4'), ('background', '<f4')]),
                  ('fitError', [('amplitude', '<f4'), ('psf_fwhm', '<f4'), ('center', '<f4'), ('background', '<f4')])]

        self._ensemble_parameter = None

    def _model_function(self, parameters, distance, ensemble_parameter=None):
        if ensemble_parameter:
            raise UserWarning('This is not an ensemble fit class')

        return naive_gaussian(parameters, distance)

    def _error_function(self, parameters, distance, data, ensemble_parameter=None):
        if ensemble_parameter:
            raise UserWarning('This is not an ensemble fit class')

        return data - self._model_function(parameters, distance)

    def _calc_guess(self, line_profile):
        # [amplitude, psf_fwhm, center position, background]
        profile = line_profile.get_data()
        distance = line_profile.get_coordinates()
        background = profile.min()
        peak = profile.max()
        amplitude = peak - background
        center_position = distance[np.where(peak == profile)[0][0]]
        tubule_diameter = np.sum(profile >= background + 0.5 * amplitude) * (distance[1] - distance[0])
        return amplitude, tubule_diameter, center_position, background

class Lorentzian(ProfileFitter):
    def __init__(self, line_profile_handler):
        super(self.__class__, self).__init__(line_profile_handler)

        # [amplitude, psf_fwhm, center position, background]]
        self._fit_result_dtype = [('index', '<i4'),
                  ('fitResults', [('amplitude', '<f4'), ('psf_fwhm', '<f4'), ('center', '<f4'), ('background', '<f4')]),
                  ('fitError', [('amplitude', '<f4'), ('psf_fwhm', '<f4'), ('center', '<f4'), ('background', '<f4')])]

        self._ensemble_parameter = None

    def _model_function(self, parameters, distance, ensemble_parameter=None):
        if ensemble_parameter:
            raise UserWarning('This is not an ensemble fit class')

        return naive_lorentzian(parameters, distance)

    def _error_function(self, parameters, distance, data, ensemble_parameter=None):
        if ensemble_parameter:
            raise UserWarning('This is not an ensemble fit class')

        return data - self._model_function(parameters, distance)

    def _calc_guess(self, line_profile):
        # [amplitude, psf_fwhm, center position, background]
        profile = line_profile.get_data()
        distance = line_profile.get_coordinates()
        background = profile.min()
        peak = profile.max()
        amplitude = peak - background
        center_position = distance[np.where(peak == profile)[0][0]]
        tubule_diameter = np.sum(profile >= background + 0.5 * amplitude) * (distance[1] - distance[0])
        return amplitude, tubule_diameter, center_position, background

class SimpleFWHM(ProfileFitter):
    def _calc_guess(self, line_profile):
        profile = line_profile.get_data()
        distance = line_profile.get_coordinates()
        background = profile.min()
        peak = profile.max()
        amplitude = peak - background
        fwhm = np.sum(profile >= background + 0.5 * amplitude) * (distance[1] - distance[0])
        return fwhm

#----------------------------------------- Lorentzian-Convolved Fitters -----------------------------------------------#


class STEDTubuleMembraneAntibody(ProfileFitter):
    def __init__(self, line_profile_handler):
        super(self.__class__, self).__init__(line_profile_handler)

        # [amplitude, tubule diameter, center position, background]
        self._fit_result_dtype = [('index', '<i4'),
                                  ('ensemble_parameter', [('psf_fwhm', '<f4')]),
                                  ('ensemble_uncertainty', [('psf_fwhm', '<f4')]),
                  ('fitResults', [('amplitude', '<f4'), ('diameter', '<f4'), ('center', '<f4'), ('background', '<f4')]),
                  ('fitError', [('amplitude', '<f4'), ('diameter', '<f4'), ('center', '<f4'), ('background', '<f4')])]

        self._ensemble_parameter = 'PSF FWHM [nm]'

        # Lorentzian-convolved model functions have squared radii in model functions
        self._squared_radius = True

    def _model_function(self, parameters, distance, ensemble_parameter=None):
        try:
            psf_fwhm = ensemble_parameter[0]
        except (TypeError, IndexError):
            psf_fwhm = ensemble_parameter

        return lorentz_convolved_tubule_surface_antibody(parameters, distance, psf_fwhm)

    def _error_function(self, parameters, distance, data, ensemble_parameter=None):
        return data - self._model_function(parameters, distance, ensemble_parameter)

    def _calc_guess(self, line_profile):
        # [amplitude, tubule diameter, center position, background]
        profile = line_profile.get_data()
        distance = line_profile.get_coordinates()
        background = profile.min()
        peak = profile.max()
        amplitude = peak - background
        center_position = distance[np.where(peak == profile)[0][0]]
        tubule_diameter = np.sum(profile >= background + 0.5 * amplitude) * (distance[1] - distance[0])
        return amplitude, tubule_diameter, center_position, background

class STEDTubuleMembraneAntibody_ne(ProfileFitter):
    def __init__(self, line_profile_handler):
        super(self.__class__, self).__init__(line_profile_handler)

        # [amplitude, tubule diameter, center position, background]
        self._fit_result_dtype = [('index', '<i4'),
                  ('fitResults', [('amplitude', '<f4'), ('diameter', '<f4'), ('center', '<f4'), ('background', '<f4'), ('psf_fwhm', '<f4')]),
                  ('fitError', [('amplitude', '<f4'), ('diameter', '<f4'), ('center', '<f4'), ('background', '<f4'), ('psf_fwhm', '<f4')])]

        self._ensemble_parameter = None  # 'PSF FWHM [nm]'

        # Lorentzian-convolved model functions have squared radii in model functions
        self._squared_radius = True

    def _model_function(self, parameters, distance, ensemble_parameter=None):
        if ensemble_parameter:
            raise UserWarning('This is not an ensemble fit class')
        return lorentz_convolved_tubule_surface_antibody_ne(parameters, distance)

    def _error_function(self, parameters, distance, data, ensemble_parameter=None):
        if ensemble_parameter:
            raise UserWarning('This is not an ensemble fit class')
        return data - self._model_function(parameters, distance)

    def _calc_guess(self, line_profile):
        # [amplitude, tubule diameter, center position, background]
        profile = line_profile.get_data()
        distance = line_profile.get_coordinates()
        background = profile.min()
        peak = profile.max()
        amplitude = peak - background
        center_position = distance[np.where(peak == profile)[0][0]]
        tubule_diameter = np.sum(profile >= background + 0.5 * amplitude) * (distance[1] - distance[0])
        psf_fwhm = tubule_diameter
        return amplitude, tubule_diameter, center_position, background, psf_fwhm

class STEDMicrotubuleAntibody(ProfileFitter):
    def __init__(self, line_profile_handler):
        super(self.__class__, self).__init__(line_profile_handler)

        # [amplitude, tubule diameter, center position, background]
        self._fit_result_dtype = [('index', '<i4'),
                                  ('ensemble_parameter', [('psf_fwhm', '<f4')]),
                                  ('ensemble_uncertainty', [('psf_fwhm', '<f4')]),
                  ('fitResults', [('amplitude', '<f4'), ('center', '<f4'), ('background', '<f4')]),
                  ('fitError', [('amplitude', '<f4'), ('center', '<f4'), ('background', '<f4')])]

        self._ensemble_parameter = 'PSF FWHM [nm]'

        # Lorentzian-convolved model functions have squared radii in model functions
        self._squared_radius = True

    def _model_function(self, parameters, distance, ensemble_parameter=None):
        try:
            psf_fwhm = ensemble_parameter[0]
        except (TypeError, IndexError):
            psf_fwhm = ensemble_parameter

        return lorentz_convolved_microtubule_antibody(parameters, distance, psf_fwhm)

    def _error_function(self, parameters, distance, data, ensemble_parameter=None):
        return data - self._model_function(parameters, distance, ensemble_parameter)

    def _calc_guess(self, line_profile):
        # [amplitude, center position, background]
        profile = line_profile.get_data()
        distance = line_profile.get_coordinates()
        background = profile.min()
        peak = profile.max()
        amplitude = peak - background
        center_position = distance[np.where(peak == profile)[0][0]]
        return amplitude, center_position, background

class STEDTubuleSelfLabeling(ProfileFitter):
    """
    This is for use with SNAP-tag and Halo tag labels, which result in an annulus of roughly 5 nm thickness
    """
    def __init__(self, line_profile_handler):
        super(self.__class__, self).__init__(line_profile_handler)

        # [amplitude, tubule diameter, center position, background]
        self._fit_result_dtype = [('index', '<i4'),
                                  ('ensemble_parameter', [('psf_fwhm', '<f4')]),
                                  ('ensemble_uncertainty', [('psf_fwhm', '<f4')]),
                  ('fitResults', [('amplitude', '<f4'), ('diameter', '<f4'), ('center', '<f4'), ('background', '<f4')]),
                  ('fitError', [('amplitude', '<f4'), ('diameter', '<f4'), ('center', '<f4'), ('background', '<f4')])]

        self._ensemble_parameter = 'PSF FWHM [nm]'

        # Lorentzian-convolved model functions have squared radii in model functions
        self._squared_radius = True

    def _model_function(self, parameters, distance, ensemble_parameter=None):
        try:
            psf_fwhm = ensemble_parameter[0]
        except (TypeError, IndexError):
            psf_fwhm = ensemble_parameter

        return lorentz_convolved_coated_tubule_selflabeling(parameters, distance, psf_fwhm)

    def _error_function(self, parameters, distance, data, ensemble_parameter=None):
        return data - self._model_function(parameters, distance, ensemble_parameter)

    def _calc_guess(self, line_profile):
        # [amplitude, tubule diameter, center position, background]
        profile = line_profile.get_data()
        distance = line_profile.get_coordinates()
        background = profile.min()
        peak = profile.max()
        amplitude = peak - background
        center_position = distance[np.where(peak == profile)[0][0]]
        tubule_diameter = np.sum(profile >= background + 0.5 * amplitude) * (distance[1] - distance[0])
        return amplitude, tubule_diameter, center_position, background

class STEDTubuleSelfLabeling_ne(ProfileFitter):
    """
    This is for use with SNAP-tag and Halo tag labels, which result in an annulus of roughly 5 nm thickness
    """
    def __init__(self, line_profile_handler):
        super(self.__class__, self).__init__(line_profile_handler)

        # [amplitude, tubule diameter, center position, background]
        self._fit_result_dtype = [('index', '<i4'),
                                  # ('ensemble_parameter', [('psf_fwhm', '<f4')]),
                                  # ('ensemble_uncertainty', [('psf_fwhm', '<f4')]),
                  ('fitResults', [('amplitude', '<f4'), ('diameter', '<f4'), ('center', '<f4'), ('background', '<f4'), ('psf_fwhm', '<f4')]),
                  ('fitError', [('amplitude', '<f4'), ('diameter', '<f4'), ('center', '<f4'), ('background', '<f4'), ('psf_fwhm', '<f4')])]

        self._ensemble_parameter = None  # 'PSF FWHM [nm]'

        # Lorentzian-convolved model functions have squared radii in model functions
        self._squared_radius = True

    def _model_function(self, parameters, distance, ensemble_parameter=None):
        if ensemble_parameter:
            raise UserWarning('This is not an ensemble fit class')
        return lorentz_convolved_coated_tubule_selflabeling_ne(parameters, distance)

    def _error_function(self, parameters, distance, data, ensemble_parameter=None):
        if ensemble_parameter:
            raise UserWarning('This is not an ensemble fit class')
        return data - self._model_function(parameters, distance)

    def _calc_guess(self, line_profile):
        # [amplitude, tubule diameter, center position, background]
        profile = line_profile.get_data()
        distance = line_profile.get_coordinates()
        background = profile.min()
        peak = profile.max()
        amplitude = peak - background
        center_position = distance[np.where(peak == profile)[0][0]]
        tubule_diameter = np.sum(profile >= background + 0.5 * amplitude) * (distance[1] - distance[0])
        psf_fwhm = tubule_diameter
        return amplitude, tubule_diameter, center_position, background, psf_fwhm

class STEDTubuleLumen(ProfileFitter):
    def __init__(self, line_profile_handler):
        super(STEDTubuleLumen, self).__init__(line_profile_handler)

        # [amplitude, tubule diameter, center position, background]
        self._fit_result_dtype = [('index', '<i4'),
                                  ('ensemble_parameter', [('psf_fwhm', '<f4')]),
                                  ('ensemble_uncertainty', [('psf_fwhm', '<f4')]),
                  ('fitResults', [('amplitude', '<f4'), ('diameter', '<f4'), ('center', '<f4'), ('background', '<f4')]),
                  ('fitError', [('amplitude', '<f4'), ('diameter', '<f4'), ('center', '<f4'), ('background', '<f4')])]

        self._ensemble_parameter = 'PSF FWHM [nm]'

        # Lorentzian-convolved model functions have squared radii in model functions
        self._squared_radius = True

    def _model_function(self, parameters, distance, ensemble_parameter=None):
        try:
            psf_fwhm = ensemble_parameter[0]
        except (TypeError, IndexError):
            psf_fwhm = ensemble_parameter
        return lorentz_convolved_tubule_lumen(parameters, distance, psf_fwhm)

    def _error_function(self, parameters, distance, data, ensemble_parameter=None):
        return data - self._model_function(parameters, distance, ensemble_parameter)

    def _calc_guess(self, line_profile):
        # [amplitude, tubule diameter, center position, background]
        profile = line_profile.get_data()
        distance = line_profile.get_coordinates()
        background = profile.min()
        peak = profile.max()
        amplitude = peak - background
        center_position = distance[np.where(peak == profile)[0][0]]
        tubule_diameter = np.sum(profile >= background + 0.5 * amplitude) * (distance[1] - distance[0])
        return amplitude, tubule_diameter, center_position, background

class STEDTubuleLumen_ne(ProfileFitter):
    def __init__(self, line_profile_handler):
        super(self.__class__, self).__init__(line_profile_handler)

        # [amplitude, tubule diameter, center position, background]
        self._fit_result_dtype = [('index', '<i4'),
                                  # ('ensemble_parameter', [('psf_fwhm', '<f4')]),
                                  # ('ensemble_uncertainty', [('psf_fwhm', '<f4')]),
                  ('fitResults', [('amplitude', '<f4'), ('diameter', '<f4'), ('center', '<f4'), ('background', '<f4'), ('psf_fwhm', '<f4')]),
                  ('fitError', [('amplitude', '<f4'), ('diameter', '<f4'), ('center', '<f4'), ('background', '<f4'), ('psf_fwhm', '<f4')])]

        self._ensemble_parameter = None  # 'PSF FWHM [nm]'

        # Lorentzian-convolved model functions have squared radii in model functions
        self._squared_radius = True

    def _model_function(self, parameters, distance, ensemble_parameter=None):
        if ensemble_parameter:
            raise UserWarning('This is not an ensemble fit class')
        return lorentz_convolved_tubule_lumen_ne(parameters, distance)

    def _error_function(self, parameters, distance, data, ensemble_parameter=None):
        if ensemble_parameter:
            raise UserWarning('This is not an ensemble fit class')
        return data - self._model_function(parameters, distance)

    def _calc_guess(self, line_profile):
        # [amplitude, tubule diameter, center position, background]
        profile = line_profile.get_data()
        distance = line_profile.get_coordinates()
        background = profile.min()
        peak = profile.max()
        amplitude = peak - background
        center_position = distance[np.where(peak == profile)[0][0]]
        tubule_diameter = np.sum(profile >= background + 0.5 * amplitude) * (distance[1] - distance[0])
        psf_fwhm = tubule_diameter
        return amplitude, tubule_diameter, center_position, background, psf_fwhm

class STEDTubuleMembrane(ProfileFitter):
    """
    Depreciated thin-membrane model
    """
    def __init__(self, line_profile_handler):
        super(self.__class__, self).__init__(line_profile_handler)

        # [amplitude, tubule diameter, center position, background]
        self._fit_result_dtype = [('index', '<i4'),
                                  ('ensemble_parameter', [('psf_fwhm', '<f4')]),
                                  ('ensemble_uncertainty', [('psf_fwhm', '<f4')]),
                  ('fitResults', [('amplitude', '<f4'), ('diameter', '<f4'), ('center', '<f4'), ('background', '<f4')]),
                  ('fitError', [('amplitude', '<f4'), ('diameter', '<f4'), ('center', '<f4'), ('background', '<f4')])]

        self._ensemble_parameter = 'PSF FWHM [nm]'

        # Lorentzian-convolved model functions have squared radii in model functions
        self._squared_radius = True

    def _model_function(self, parameters, distance, ensemble_parameter=None):
        try:
            psf_fwhm = ensemble_parameter[0]
        except (TypeError, IndexError):
            psf_fwhm = ensemble_parameter
        return lorentz_convolved_tubule_membrane(parameters, distance, psf_fwhm)

    def _error_function(self, parameters, distance, data, ensemble_parameter=None):
        return data - self._model_function(parameters, distance, ensemble_parameter)

    def _calc_guess(self, line_profile):
        # [amplitude, tubule diameter, center position, background]
        profile = line_profile.get_data()
        distance = line_profile.get_coordinates()
        background = profile.min()
        peak = profile.max()
        amplitude = peak - background
        center_position = distance[np.where(peak == profile)[0][0]]
        tubule_diameter = np.sum(profile >= background + 0.5 * amplitude) * (distance[1] - distance[0])
        return amplitude, tubule_diameter, center_position, background


# ----------------------------------------- Gaussian-Convolved Fitters ------------------------------------------------#


class GaussTubuleMembraneAntibody(ProfileFitter):
    def __init__(self, line_profile_handler):
        super(self.__class__, self).__init__(line_profile_handler)

        # [amplitude, tubule diameter, center position, background]
        self._fit_result_dtype = [('index', '<i4'),
                                  ('ensemble_parameter', [('psf_fwhm', '<f4')]),
                                  ('ensemble_uncertainty', [('psf_fwhm', '<f4')]),
                  ('fitResults', [('amplitude', '<f4'), ('diameter', '<f4'), ('center', '<f4'), ('background', '<f4')]),
                  ('fitError', [('amplitude', '<f4'), ('diameter', '<f4'), ('center', '<f4'), ('background', '<f4')])]

        self._ensemble_parameter = 'PSF FWHM [nm]'

    def _model_function(self, parameters, distance, ensemble_parameter=None):
        try:
            psf_fwhm = ensemble_parameter[0]
        except (TypeError, IndexError):
            psf_fwhm = ensemble_parameter

        return gauss_convolved_tubule_surface_antibody(parameters, distance, psf_fwhm)

    def _error_function(self, parameters, distance, data, ensemble_parameter=None):
        return data - self._model_function(parameters, distance, ensemble_parameter)

    def _calc_guess(self, line_profile):
        # [amplitude, tubule diameter, center position, background]
        profile = line_profile.get_data()
        distance = line_profile.get_coordinates()
        background = profile.min()
        peak = profile.max()
        amplitude = peak - background
        center_position = distance[np.where(peak == profile)[0][0]]
        tubule_diameter = np.sum(profile >= background + 0.5 * amplitude) * (distance[1] - distance[0])
        return amplitude, tubule_diameter, center_position, background

class GaussTubuleMembraneAntibody_ne(ProfileFitter):
    def __init__(self, line_profile_handler):
        super(self.__class__, self).__init__(line_profile_handler)

        # [amplitude, tubule diameter, center position, background]
        self._fit_result_dtype = [('index', '<i4'),
                  ('fitResults', [('amplitude', '<f4'), ('diameter', '<f4'), ('center', '<f4'), ('background', '<f4'), ('psf_fwhm', '<f4')]),
                  ('fitError', [('amplitude', '<f4'), ('diameter', '<f4'), ('center', '<f4'), ('background', '<f4'), ('psf_fwhm', '<f4')])]

        self._ensemble_parameter = None  # 'PSF FWHM [nm]'

    def _model_function(self, parameters, distance, ensemble_parameter=None):
        if ensemble_parameter:
            raise UserWarning('This is not an ensemble fit class')
        return gauss_convolved_tubule_surface_antibody_ne(parameters, distance)

    def _error_function(self, parameters, distance, data, ensemble_parameter=None):
        if ensemble_parameter:
            raise UserWarning('This is not an ensemble fit class')
        return data - self._model_function(parameters, distance)

    def _calc_guess(self, line_profile):
        # [amplitude, tubule diameter, center position, background]
        profile = line_profile.get_data()
        distance = line_profile.get_coordinates()
        background = profile.min()
        peak = profile.max()
        amplitude = peak - background
        center_position = distance[np.where(peak == profile)[0][0]]
        tubule_diameter = np.sum(profile >= background + 0.5 * amplitude) * (distance[1] - distance[0])
        psf_fwhm = tubule_diameter
        return amplitude, tubule_diameter, center_position, background, psf_fwhm

class GaussMicrotubuleAntibody(ProfileFitter):
    def __init__(self, line_profile_handler):
        super(self.__class__, self).__init__(line_profile_handler)

        # [amplitude, tubule diameter, center position, background]
        self._fit_result_dtype = [('index', '<i4'),
                                  ('ensemble_parameter', [('psf_fwhm', '<f4')]),
                                  ('ensemble_uncertainty', [('psf_fwhm', '<f4')]),
                  ('fitResults', [('amplitude', '<f4'), ('center', '<f4'), ('background', '<f4')]),
                  ('fitError', [('amplitude', '<f4'), ('center', '<f4'), ('background', '<f4')])]

        self._ensemble_parameter = 'PSF FWHM [nm]'

    def _model_function(self, parameters, distance, ensemble_parameter=None):
        try:
            psf_fwhm = ensemble_parameter[0]
        except (TypeError, IndexError):
            psf_fwhm = ensemble_parameter

        return gauss_convolved_microtubule_antibody(parameters, distance, psf_fwhm)

    def _error_function(self, parameters, distance, data, ensemble_parameter=None):
        return data - self._model_function(parameters, distance, ensemble_parameter)

    def _calc_guess(self, line_profile):
        # [amplitude, center position, background]
        profile = line_profile.get_data()
        distance = line_profile.get_coordinates()
        background = profile.min()
        peak = profile.max()
        amplitude = peak - background
        center_position = distance[np.where(peak == profile)[0][0]]
        return amplitude, center_position, background

class GaussTubuleSelfLabeling(ProfileFitter):
    """
    This is for use with SNAP-tag and Halo tag labels, which result in an annulus of roughly 5 nm thickness
    """
    def __init__(self, line_profile_handler):
        super(self.__class__, self).__init__(line_profile_handler)

        # [amplitude, tubule diameter, center position, background]
        self._fit_result_dtype = [('index', '<i4'),
                                  ('ensemble_parameter', [('psf_fwhm', '<f4')]),
                                  ('ensemble_uncertainty', [('psf_fwhm', '<f4')]),
                  ('fitResults', [('amplitude', '<f4'), ('diameter', '<f4'), ('center', '<f4'), ('background', '<f4')]),
                  ('fitError', [('amplitude', '<f4'), ('diameter', '<f4'), ('center', '<f4'), ('background', '<f4')])]

        self._ensemble_parameter = 'PSF FWHM [nm]'

    def _model_function(self, parameters, distance, ensemble_parameter=None):
        try:
            psf_fwhm = ensemble_parameter[0]
        except (TypeError, IndexError):
            psf_fwhm = ensemble_parameter

        return gauss_convolved_coated_tubule_selflabeling(parameters, distance, psf_fwhm)

    def _error_function(self, parameters, distance, data, ensemble_parameter=None):
        return data - self._model_function(parameters, distance, ensemble_parameter)

    def _calc_guess(self, line_profile):
        # [amplitude, tubule diameter, center position, background]
        profile = line_profile.get_data()
        distance = line_profile.get_coordinates()
        background = profile.min()
        peak = profile.max()
        amplitude = peak - background
        center_position = distance[np.where(peak == profile)[0][0]]
        tubule_diameter = np.sum(profile >= background + 0.5 * amplitude) * (distance[1] - distance[0])
        return amplitude, tubule_diameter, center_position, background

class GaussTubuleSelfLabeling_ne(ProfileFitter):
    """
    This is for use with SNAP-tag and Halo tag labels, which result in an annulus of roughly 5 nm thickness
    """
    def __init__(self, line_profile_handler):
        super(self.__class__, self).__init__(line_profile_handler)

        # [amplitude, tubule diameter, center position, background]
        self._fit_result_dtype = [('index', '<i4'),
                                  # ('ensemble_parameter', [('psf_fwhm', '<f4')]),
                                  # ('ensemble_uncertainty', [('psf_fwhm', '<f4')]),
                  ('fitResults', [('amplitude', '<f4'), ('diameter', '<f4'), ('center', '<f4'), ('background', '<f4'), ('psf_fwhm', '<f4')]),
                  ('fitError', [('amplitude', '<f4'), ('diameter', '<f4'), ('center', '<f4'), ('background', '<f4'), ('psf_fwhm', '<f4')])]

        self._ensemble_parameter = None  # 'PSF FWHM [nm]'

    def _model_function(self, parameters, distance, ensemble_parameter=None):
        if ensemble_parameter:
            raise UserWarning('This is not an ensemble fit class')
        return gauss_convolved_coated_tubule_selflabeling_ne(parameters, distance)

    def _error_function(self, parameters, distance, data, ensemble_parameter=None):
        if ensemble_parameter:
            raise UserWarning('This is not an ensemble fit class')
        return data - self._model_function(parameters, distance)

    def _calc_guess(self, line_profile):
        # [amplitude, tubule diameter, center position, background]
        profile = line_profile.get_data()
        distance = line_profile.get_coordinates()
        background = profile.min()
        peak = profile.max()
        amplitude = peak - background
        center_position = distance[np.where(peak == profile)[0][0]]
        tubule_diameter = np.sum(profile >= background + 0.5 * amplitude) * (distance[1] - distance[0])
        psf_fwhm = tubule_diameter
        return amplitude, tubule_diameter, center_position, background, psf_fwhm

class GaussTubuleLumen(ProfileFitter):
    def __init__(self, line_profile_handler):
        super(self.__class__, self).__init__(line_profile_handler)

        # [amplitude, tubule diameter, center position, background]
        self._fit_result_dtype = [('index', '<i4'),
                                  ('ensemble_parameter', [('psf_fwhm', '<f4')]),
                                  ('ensemble_uncertainty', [('psf_fwhm', '<f4')]),
                  ('fitResults', [('amplitude', '<f4'), ('diameter', '<f4'), ('center', '<f4'), ('background', '<f4')]),
                  ('fitError', [('amplitude', '<f4'), ('diameter', '<f4'), ('center', '<f4'), ('background', '<f4')])]

        self._ensemble_parameter = 'PSF FWHM [nm]'

    def _model_function(self, parameters, distance, ensemble_parameter=None):
        try:
            psf_fwhm = ensemble_parameter[0]
        except (TypeError, IndexError):
            psf_fwhm = ensemble_parameter
        return gauss_convolved_tubule_lumen_approx(parameters, distance, psf_fwhm)

    def _error_function(self, parameters, distance, data, ensemble_parameter=None):
        return data - self._model_function(parameters, distance, ensemble_parameter)

    def _calc_guess(self, line_profile):
        # [amplitude, tubule diameter, center position, background]
        profile = line_profile.get_data()
        distance = line_profile.get_coordinates()
        background = profile.min()
        peak = profile.max()
        amplitude = peak - background
        center_position = distance[np.where(peak == profile)[0][0]]
        tubule_diameter = np.sum(profile >= background + 0.5 * amplitude) * (distance[1] - distance[0])
        return amplitude, tubule_diameter, center_position, background

class GaussTubuleLumen_ne(ProfileFitter):
    def __init__(self, line_profile_handler):
        super(self.__class__, self).__init__(line_profile_handler)

        # [amplitude, tubule diameter, center position, background]
        self._fit_result_dtype = [('index', '<i4'),
                                  # ('ensemble_parameter', [('psf_fwhm', '<f4')]),
                                  # ('ensemble_uncertainty', [('psf_fwhm', '<f4')]),
                  ('fitResults', [('amplitude', '<f4'), ('diameter', '<f4'), ('center', '<f4'), ('background', '<f4'), ('psf_fwhm', '<f4')]),
                  ('fitError', [('amplitude', '<f4'), ('diameter', '<f4'), ('center', '<f4'), ('background', '<f4'), ('psf_fwhm', '<f4')])]

        self._ensemble_parameter = None  # 'PSF FWHM [nm]'

    def _model_function(self, parameters, distance, ensemble_parameter=None):
        if ensemble_parameter:
            raise UserWarning('This is not an ensemble fit class')
        return gauss_convolved_tubule_lumen_approx_ne(parameters, distance)

    def _error_function(self, parameters, distance, data, ensemble_parameter=None):
        if ensemble_parameter:
            raise UserWarning('This is not an ensemble fit class')
        return data - self._model_function(parameters, distance)

    def _calc_guess(self, line_profile):
        # [amplitude, tubule diameter, center position, background]
        profile = line_profile.get_data()
        distance = line_profile.get_coordinates()
        background = profile.min()
        peak = profile.max()
        amplitude = peak - background
        center_position = distance[np.where(peak == profile)[0][0]]
        tubule_diameter = np.sum(profile >= background + 0.5 * amplitude) * (distance[1] - distance[0])
        psf_fwhm = tubule_diameter
        return amplitude, tubule_diameter, center_position, background, psf_fwhm

# fitters should be non-ensemble fits only, or ensemble fits with constant ensemble_parameter
fitters = {
    'Gaussian': Gaussian,
    'Lorentzian': Lorentzian,
    'STEDTubule_Filled': STEDTubuleLumen_ne,
    'STEDTubule_SurfaceAntibody': STEDTubuleMembraneAntibody_ne,
    'STEDTubule_SurfaceSNAP': STEDTubuleSelfLabeling_ne
}
ensemble_fitters = {
    'STEDTubule_Filled': STEDTubuleLumen,
    # 'STEDTubuleMembrane': STEDTubuleMembrane,  # thin membrane approximation is depreciated
    'STEDTubule_SurfaceAntibody': STEDTubuleMembraneAntibody,
    'STEDTubule_SurfaceSNAP': STEDTubuleSelfLabeling,
    'STEDMicrotubule_SurfaceAntibody': STEDMicrotubuleAntibody,
    'GaussTubule_Filled': GaussTubuleLumen,
    'GaussTubule_SurfaceAntibody': GaussTubuleMembraneAntibody,
    'GaussTubule_SurfaceSNAP': GaussTubuleSelfLabeling,
    'GaussMicrotubule_SurfaceAntibody': GaussMicrotubuleAntibody
}
