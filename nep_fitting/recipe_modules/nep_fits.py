from PYME.recipes.base import register_module, ModuleBase
from PYME.recipes.traits import Input, Output, Float, CStr, Bool, DictStrList

from PYME.IO import tabular
from PYME.IO import MetaDataHandler

from nep_fitting.core import profile_fitters, region_fitters
from nep_fitting.core.handlers import LineProfileHandler, RegionHandler


@register_module('TestEnsembleParameters')
class TestEnsembleParameters(ModuleBase):
    inputName = Input('profiles')

    fit_type = CStr(list(profile_fitters.ensemble_fitters.keys())[0])
    ensemble_test_values = DictStrList({'psf_fwhm': [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]})

    outputName = Output('fit_results')

    def execute(self, namespace):


        inp = namespace[self.inputName]

        # generate LineProfileHandler from tables
        handler = LineProfileHandler()
        handler._load_profiles_from_list(inp)

        fit_class = profile_fitters.ensemble_fitters[self.fit_type]
        fitter = fit_class(handler)

        results = fitter.ensemble_test(self.ensemble_test_values)
        res = tabular.RecArraySource(results)

        # propagate metadata, if present
        res.mdh = MetaDataHandler.NestedClassMDHandler(getattr(inp, 'mdh', None))
        
        res.mdh['TestEnsembleParameters.FitType'] = self.fit_type
        res.mdh['TestEnsembleParameters.EnsembleTestValues'] = self.ensemble_test_values

        namespace[self.outputName] = res

    @property
    def _fitter_choices(self):
        return list(profile_fitters.ensemble_fitters.keys())


    @property
    def default_view(self):
        from traitsui.api import View, Item
        from PYME.ui.custom_traits_editors import CBEditor

        return View(Item('inputName', editor=CBEditor(choices=self._namespace_keys)),
                    Item('_'),
                    Item('fit_type', editor=CBEditor(choices=self._fitter_choices)),
                    Item('_'),
                    Item(''),
                    Item('ensemble_test_values'),
                    Item(''),
                    Item('outputName'), buttons=['OK'])
    
    @property
    def pipeline_view(self):
        from traitsui.api import View, Item
        from PYME.ui.custom_traits_editors import CBEditor
    
        return View(
                    Item('fit_type', editor=CBEditor(choices=self._fitter_choices)),
                    Item('_'),
                    Item(''),
                    Item('ensemble_test_values'),
                    )

@register_module('EnsembleFitProfiles')
class EnsembleFitProfiles(ModuleBase):
    inputName = Input('line_profiles')

    fit_type = CStr(list(profile_fitters.ensemble_fitters.keys())[0])
    ensemble_parameter_guess = Float(50.)
    hold_ensemble_parameter_constant = Bool(False)

    outputName = Output('fit_results')
    outputFitter = Output('fitter')

    def execute(self, namespace):


        inp = namespace[self.inputName]

        # generate LineProfileHandler from tables
        handler = LineProfileHandler()
        handler._load_profiles_from_list(inp)

        fit_class = profile_fitters.ensemble_fitters[self.fit_type]
        fitter = fit_class(handler)

        if self.hold_ensemble_parameter_constant:
            fitter.fit_profiles(self.ensemble_parameter_guess)
        else:
            fitter.ensemble_fit(self.ensemble_parameter_guess)

        res = tabular.RecArraySource(fitter.results)

        # propagate metadata, if present
        res.mdh = MetaDataHandler.NestedClassMDHandler(getattr(inp, 'mdh', None))

        res.mdh['EnsembleFitProfiles.FitType'] = self.fit_type
        res.mdh['EnsembleFitProfiles.EnsembleParameterGuess'] = self.ensemble_parameter_guess
        res.mdh['EnsembleFitProfiles.HoldEnsembleParamConstant'] = self.hold_ensemble_parameter_constant

        namespace[self.outputName] = res
        namespace[self.outputFitter] = fitter

    @property
    def _fitter_choices(self):
        return list(profile_fitters.ensemble_fitters.keys())


    @property
    def default_view(self):
        from traitsui.api import View, Item
        from PYME.ui.custom_traits_editors import CBEditor

        return View(Item('inputName', editor=CBEditor(choices=self._namespace_keys)),
                    Item('_'),
                    Item('fit_type', editor=CBEditor(choices=self._fitter_choices)),
                    Item('_'),
                    Item('ensemble_parameter_guess'),
                    Item('_'),
                    Item('hold_ensemble_parameter_constant'),
                    Item('_'),
                    Item('outputName'), buttons=['OK'])

    @property
    def pipeline_view(self):
        from traitsui.api import View, Item
        from PYME.ui.custom_traits_editors import CBEditor
    
        return View(Item('fit_type', editor=CBEditor(choices=self._fitter_choices)),
                    Item('_'),
                    Item('ensemble_parameter_guess'),
                    Item('_'),
                    Item('hold_ensemble_parameter_constant'),
                    )

    @property
    def dsview_view(self):
        from traitsui.api import View, Item
        from PYME.ui.custom_traits_editors import CBEditor

        return View(Item('fit_type', editor=CBEditor(choices=self._fitter_choices)),
                    Item('_'),
                    Item('ensemble_parameter_guess'),
                    Item('_'),
                    Item('hold_ensemble_parameter_constant'), buttons=['OK'])


@register_module('FitProfiles')
class FitProfiles(ModuleBase):
    inputName = Input('profiles')

    fit_type = CStr(list(profile_fitters.non_ensemble_fitters.keys())[0])

    outputName = Output('fit_results')
    outputFitter = Output('fitter')

    def execute(self, namespace):


        inp = namespace[self.inputName]

        # generate LineProfileHandler from tables
        handler = LineProfileHandler()
        handler._load_profiles_from_list(inp)

        fit_class = profile_fitters.non_ensemble_fitters[self.fit_type]
        fitter = fit_class(handler)

        fitter.fit_profiles()

        res = tabular.RecArraySource(fitter.results)

        # propagate metadata, if present
        res.mdh = MetaDataHandler.NestedClassMDHandler(getattr(inp, 'mdh', None))

        res.mdh['FitProfiles.FitType'] = self.fit_type

        namespace[self.outputName] = res
        namespace[self.outputFitter] = fitter

    @property
    def _fitter_choices(self):
        return list(profile_fitters.non_ensemble_fitters.keys())


    @property
    def default_view(self):
        from traitsui.api import View, Item
        from PYME.ui.custom_traits_editors import CBEditor

        return View(Item('inputName', editor=CBEditor(choices=self._namespace_keys)),
                    Item('_'),
                    Item('fit_type', editor=CBEditor(choices=self._fitter_choices)),
                    Item('_'),
                    Item('outputName'), buttons=['OK'])

    @property
    def pipeline_view(self):
        from traitsui.api import View, Item
        from PYME.ui.custom_traits_editors import CBEditor
    
        return View(
                    Item('fit_type', editor=CBEditor(choices=self._fitter_choices)),
                    )

@register_module('EnsembleFitROIs')
class EnsembleFitROIs(ModuleBase):  # Note that this should probably be moved somewhere else
    inputName = Input('ROIs')

    fit_type = CStr('LorentzianConvolvedSolidSphere_ensemblePSF')
    ensemble_parameter_guess = Float(50.)
    hold_ensemble_parameter_constant = Bool(False)

    outputName = Output('fit_results')
    outputFitter = Output('fitter')

    def execute(self, namespace):


        inp = namespace[self.inputName]

        # generate RegionHandler from tables
        handler = RegionHandler()
        handler._load_from_list(inp)

        fit_class = region_fitters.fitters[self.fit_type]
        fitter = fit_class(handler)

        if self.hold_ensemble_parameter_constant:
            fitter.fit_profiles(self.ensemble_parameter_guess)
        else:
            fitter.ensemble_fit(self.ensemble_parameter_guess)

        res = tabular.RecArraySource(fitter.results)

        # propagate metadata, if present
        res.mdh = MetaDataHandler.NestedClassMDHandler(getattr(inp, 'mdh', None))

        res.mdh['EnsembleFitROIs.FitType'] = self.fit_type
        res.mdh['EnsembleFitROIs.EnsembleParameterGuess'] = self.ensemble_parameter_guess
        res.mdh['EnsembleFitROIs.HoldEnsembleParamConstant'] = self.hold_ensemble_parameter_constant

        namespace[self.outputName] = res
        namespace[self.outputFitter] = fitter

    @property
    def _fitter_choices(self):
        return list(region_fitters.ensemble_fitters.keys()) #FIXME???


    @property
    def default_view(self):
        from traitsui.api import View, Item
        from PYME.ui.custom_traits_editors import CBEditor

        return View(Item('inputName', editor=CBEditor(choices=self._namespace_keys)),
                    Item('_'),
                    Item('fit_type', editor=CBEditor(choices=self._fitter_choices)),
                    Item('_'),
                    Item('ensemble_parameter_guess'),
                    Item('_'),
                    Item('hold_ensemble_parameter_constant'),
                    Item('_'),
                    Item('outputName'), buttons=['OK'])

    @property
    def pipeline_view(self):
        from traitsui.api import View, Item
        from PYME.ui.custom_traits_editors import CBEditor
    
        return View(
                    Item('fit_type', editor=CBEditor(choices=self._fitter_choices)),
                    Item('_'),
                    Item('ensemble_parameter_guess'),
                    Item('_'),
                    Item('hold_ensemble_parameter_constant'),
                    )

    @property
    def dsview_view(self):
        from traitsui.api import View, Item
        from PYME.ui.custom_traits_editors import CBEditor

        return View(Item('fit_type', editor=CBEditor(choices=self._fitter_choices)),
                    Item('_'),
                    Item('ensemble_parameter_guess'),
                    Item('_'),
                    Item('hold_ensemble_parameter_constant'), buttons=['OK'])
