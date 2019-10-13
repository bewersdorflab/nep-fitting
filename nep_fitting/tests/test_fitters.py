
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
    ground: 1darray
        default parameters from GROUND dictionary in the right order for the specified fitter

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
    for fitter_class in profile_fitters.fitters.values():
        ground = get_ground(fitter_class)
        p = np.random.poisson(fitter_class._model_function(None, ground, X))
        lp = rois.LineProfile(distance=X, profile=p)
        h = handlers.LineProfileHandler()
        h.add_line_profile(lp, update=False)
        fitter = fitter_class(h)
        fitter.fit_profiles()
        np.testing.assert_allclose(fitter.results['fitResults'][0].tolist(), ground, atol=5, rtol=0.15)

def test_ensemble_fitters():
    for fitter_class in profile_fitters.ensemble_fitters.values():
        ground, ens = get_ground(fitter_class)
        # leave out poisson sampling since we're only making one profile and with noise the lumen fitters mix psf / diam
        p = fitter_class._model_function(None, ground, X, ens)
        lp = rois.LineProfile(distance=X, profile=p)
        h = handlers.LineProfileHandler()
        h.add_line_profile(lp, update=False)
        fitter = fitter_class(h)
        fitter.ensemble_fit(ens)
        np.testing.assert_allclose(fitter.results['fitResults'][0].tolist(), ground, rtol=0.1)
        np.testing.assert_allclose(fitter.results['ensemble_parameter'][0].tolist(), ens, rtol=0.1)

