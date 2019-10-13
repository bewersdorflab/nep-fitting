
import numpy as np
from nep_fitting.core import profile_fitters, handlers, rois, models

x = np.arange(100.)

def test_gaussian():
    # amp, fwhm, x0, bkgnd = p
    ground = 100, 20, 50, 10
    p = np.random.poisson(models.naive_gaussian(ground, x))
    lp = rois.LineProfile(profile=p, distance=x)
    h = handlers.LineProfileHandler()
    h.add_line_profile(lp, update=False)
    fitter = profile_fitters.Gaussian(h)
    fitter.fit_profiles()
    np.testing.assert_allclose(fitter.results['fitResults'][0].tolist(), ground, rtol=10)

def test_lorentzian():
    # amp, fwhm, x0, bkgnd = p
    ground = 100, 20, 50, 10
    p = np.random.poisson(models.naive_lorentzian(ground, x))
    lp = rois.LineProfile(profile=p, distance=x)
    h = handlers.LineProfileHandler()
    h.add_line_profile(lp, update=False)
    fitter = profile_fitters.Lorentzian(h)
    fitter.fit_profiles()
    np.testing.assert_allclose(fitter.results['fitResults'][0].tolist(), ground, rtol=10)