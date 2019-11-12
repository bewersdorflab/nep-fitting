
import numpy as np
from nep_fitting.core import profile_fitters, handlers, rois

X = np.arange(100.)
GROUND = {
    'amplitude': 100.,
    'diameter': 25.,
    'center': 50.,
    'background': 10.,
    'psf_fwhm': 30.
}

def get_ground(fitter):
    """
    
    Parameters
    ----------
    fitter: profile_fitters.ProfileFitter
        fitter class or instantiated object

    Returns
    -------
    ground: list
        default parameters from GROUND dictionary in the right order for the specified fitter
    ens: list
        optional return for ensemble fit classes, same as ground return but for ensemble parameters

    """
    for d in fitter._fit_result_dtype:
        if d[0] == 'fitResults':
            ground = [GROUND[field[0]] for field in d[1]]
        if d[0] == 'ensemble_parameter':
            ens = [GROUND[field[0]] for field in d[1]]

    if fitter._ensemble_parameter is None:
        return ground
    return ground, ens

def test_non_ensemble_fitters():
    for fitter_class in profile_fitters.non_ensemble_fitters.values():
        ground = get_ground(fitter_class)
        # leave out noise for non-ensemble fitters, otherwise results will be poor when fitting a single profile
        p = fitter_class._model_function(ground, X)
        lp = rois.LineProfile(distance=X, profile=p)
        h = handlers.LineProfileHandler()
        h.add_line_profile(lp, update=False)
        fitter = fitter_class(h)
        fitter.fit_profiles()
        np.testing.assert_allclose(fitter.results['fitResults'][0].tolist(), ground, atol=5, rtol=0.15)


def test_naive_fitter_single_fits():
    for fitter_class in profile_fitters.naive_fitters.values():
        ground = get_ground(fitter_class)
        p = np.random.poisson(fitter_class._model_function(ground, X))
        lp = rois.LineProfile(distance=X, profile=p)
        results = fitter_class.fit_single_profile(lp)
        np.testing.assert_allclose(results['fitResults'][0].tolist(), ground, atol=5, rtol=0.15)

def test_ensemble_fitters():
    for fitter_class in profile_fitters.ensemble_fitters.values():
        ground, ens = get_ground(fitter_class)
        # leave out poisson sampling since we're only making one profile and with noise the lumen fitters mix psf / diam
        p = fitter_class._model_function(ground, X, ens)
        lp = rois.LineProfile(distance=X, profile=p)
        h = handlers.LineProfileHandler()
        h.add_line_profile(lp, update=False)
        fitter = fitter_class(h)
        fitter.ensemble_fit(ens)
        np.testing.assert_allclose(fitter.results['fitResults'][0].tolist(), ground, rtol=0.1)
        np.testing.assert_allclose(fitter.results['ensemble_parameter'][0].tolist(), ens, rtol=0.1)
