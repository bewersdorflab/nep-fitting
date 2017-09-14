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

    def _model_function(self, parameters, distance, ensemble_parameter=None):
        """
        prototype function to be overridden in derived classes

        Parameters
        ----------
        parameters
        distance
        ensemble_parameter

        Returns
        -------

        """
        raise NotImplementedError

    def _error_function(self, parameters, distance, data, ensemble_parameter=None):
        """
        prototype function to be overridden in derived classes

        Parameters
        ----------
        parameters
        distance
        data
        ensemble_parameter

        Returns
        -------

        """
        raise NotImplementedError

    def _calc_guess(self, line_profile):
        """
        prototype function to be overridden in derived classes
        Parameters
        ----------
        line_profile

        Returns
        -------

        """
        raise NotImplementedError

    def fit_profiles(self, ensemble_parameter=None):
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
                # self.results[pi]['ensemble_uncertainty'] = np.atleast_1d(ensemble_parameter).astype('f')
            self.results[pi]['fitResults'] = res.astype('f')
            self.results[pi]['fitError'] = errors.astype('f')
            #results[pi] = np.array([(pi, res.astype('f'), errors.astype('f'))])#, dtype=self._fit_result_dtype)['fitResults']


            # mse[pi] = np.mean(residuals**2)
            all_residuals.append(residuals)

        # ensemble_error = mse.mean()

        # return ensemble_error
        return np.hstack(all_residuals)

    def fit_profiles_mean(self, ensemble_parameter=None):
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
                # self.results[pi]['ensemble_uncertainty'] = np.atleast_1d(ensemble_parameter).astype('f')
            self.results[pi]['fitResults'] = res.astype('f')
            self.results[pi]['fitError'] = errors.astype('f')
            #results[pi] = np.array([(pi, res.astype('f'), errors.astype('f'))])#, dtype=self._fit_result_dtype)['fitResults']


            mse[pi] = np.mean(residuals**2)


        ensemble_error = mse.mean()

        return ensemble_error


    def ensemble_test(self, test_vals):
        num_tests = len(test_vals.items()[0][1])
        ensemble_error = np.zeros(num_tests)
        try:

            for fi in range(num_tests):
                ensemble_error[fi] = np.mean(self.fit_profiles(ensemble_parameter=test_vals[fi])**2)
        except KeyError:
            keys = test_vals.keys()
            test_params = zip(*[test_vals[key] for key in keys])
            for fi in range(num_tests):
                ensemble_error[fi] = np.mean(self.fit_profiles(ensemble_parameter=test_params[fi])**2)


        return ensemble_error

    def ensemble_fit(self, guess):
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

    def plot_results(self, plot_dir):
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
            interpolated_coords = np.linspace(position.min(), position.max(), 300)
            plt.plot(interpolated_coords, self._model_function(res[ind]['fitResults'], interpolated_coords, res[ind]['ensemble_parameter']),
                     color=colors[1], label='Fit')
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

        return data - naive_gaussian(parameters, distance)

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

        return data - naive_lorentzian(parameters, distance)

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

    def _model_function(self, parameters, distance, ensemble_parameter=None):
        try:
            psf_fwhm = ensemble_parameter[0]
        except IndexError:
            psf_fwhm = ensemble_parameter

        return lorentz_convolved_tubule_membrane_antibody(parameters, distance, psf_fwhm)

    def _error_function(self, parameters, distance, data, ensemble_parameter=None):
        try:
            psf_fwhm = ensemble_parameter[0]
        except IndexError:
            psf_fwhm = ensemble_parameter

        return data - lorentz_convolved_tubule_membrane_antibody(parameters, distance, psf_fwhm)

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

        self._ensemble_parameter = 'PSF FWHM [nm]'

    def _model_function(self, parameters, distance, ensemble_parameter=None):

        return lorentz_convolved_tubule_membrane_antibody_ne(parameters, distance)

    def _error_function(self, parameters, distance, data, ensemble_parameter=None):

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

    def _model_function(self, parameters, distance, ensemble_parameter=None):
        try:
            psf_fwhm = ensemble_parameter[0]
        except IndexError:
            psf_fwhm = ensemble_parameter

        return lorentz_convolved_microtubule_antibody(parameters, distance, psf_fwhm)

    def _error_function(self, parameters, distance, data, ensemble_parameter=None):
        try:
            psf_fwhm = ensemble_parameter[0]
        except IndexError:
            psf_fwhm = ensemble_parameter

        return data - lorentz_convolved_microtubule_antibody(parameters, distance, psf_fwhm)

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

    def _model_function(self, parameters, distance, ensemble_parameter=None):
        try:
            psf_fwhm = ensemble_parameter[0]
        except IndexError:
            psf_fwhm = ensemble_parameter

        return lorentz_convolved_coated_tubule_selflabeling(parameters, distance, psf_fwhm)

    def _error_function(self, parameters, distance, data, ensemble_parameter=None):
        try:
            psf_fwhm = ensemble_parameter[0]
        except IndexError:
            psf_fwhm = ensemble_parameter

        return data - self._model_function(parameters, distance, psf_fwhm)

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

class STEDTubuleLumen(ProfileFitter):
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
        except IndexError:
            psf_fwhm = ensemble_parameter
        return lorentz_convolved_tubule_lumen(parameters, distance, psf_fwhm)

    def _error_function(self, parameters, distance, data, ensemble_parameter=None):
        try:
            psf_fwhm = ensemble_parameter[0]
        except IndexError:
            psf_fwhm = ensemble_parameter
        return lorentz_convolved_tubule_lumen_misfit(parameters, distance, data, psf_fwhm)

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

class STEDTubuleMembrane(ProfileFitter):
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
        except IndexError:
            psf_fwhm = ensemble_parameter
        return lorentz_convolved_tubule_membrane(parameters, distance, psf_fwhm)

    def _error_function(self, parameters, distance, data, ensemble_parameter=None):
        try:
            psf_fwhm = ensemble_parameter[0]
        except IndexError:
            psf_fwhm = ensemble_parameter
        return lorentz_convolved_tubule_membrane_misfit(parameters, distance, data, psf_fwhm)

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

# TODO - fitters should be non-ensemble fits only, or ensemble fits with constant ensemble_parameter
fitters = {
    'Gaussian': Gaussian,
    'Lorentzian': Lorentzian,
    'STEDTubuleLumen': STEDTubuleLumen,
    'STEDTubuleMembrane': STEDTubuleMembrane,
    'STEDTubuleMembraneAntibody_ne': STEDTubuleMembraneAntibody_ne
}
ensemble_fitters = {
    'STEDTubuleLumen': STEDTubuleLumen,
    'STEDTubuleMembrane': STEDTubuleMembrane,
    'STEDTubuleMembraneAntibody': STEDTubuleMembraneAntibody,
    'STEDMicrotubuleAntibody': STEDMicrotubuleAntibody
}
