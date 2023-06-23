#!/usr/bin/python

# lineProfilesOverlay.py
#
# Copyright Michael Graff and Andrew Barentine
#   graff@hm.edu
#   andrew.barentine@yale.edu
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

import wx

import PYME.ui.autoFoldPanel as afp

from nep_fitting.core.handlers import LineProfileHandler
from nep_fitting.core.rois import LineProfile
from nep_fitting.core import profile_fitters
import uuid
import os
import logging
logger = logging.getLogger(__name__)


class LineProfilesOverlay:
    PEN_COLOR_LINE = wx.GREEN
    PEN_COLOR_LABEL = wx.BLUE
    """
    This is the general handler for the gui stuff to handle lineprofiles.
    """

    def __init__(self, dsviewer):
        # FIXME - hard references to dsviewer etc ... will create circular references, Use plugin interface instead
        self._dsviewer = dsviewer
        self._view = dsviewer.view
        self._do = dsviewer.do
        self._image = dsviewer.image
        filename = self._image.filename
        self.image_name = filename.split('/')[-1].strip('.' + filename.split('.')[-1])
        # create LineProfileHandler with reference to ImageStack
        self._line_profile_handler = LineProfileHandler(self._image, image_name=self.image_name)
        
        dsviewer.AddMenuItem('Profiles', "Draw line\tCtrl-L", self._add_line)

        # add this overlay to the overlays to be rendered
        self._view.add_overlay(self.DrawOverlays, 'NEP Profiles')

        # add a gui panel to the window to control the values
        self._dsviewer.paneHooks.append(self.generate_panel)
        self._line_profile_handler.LIST_CHANGED_SIGNAL.connect(self._refresh)

    def _add_line(self, event=None):
        """
        This callback function is called, whenever a new line has been drawn using the selection tool
        and should be added
        """
        try:
            trace = self._dsviewer.do.selection_trace
        except AttributeError:  # PYME after 2022/11
            trace = self._dsviewer.do.selection.trace

        if len(trace) > 2:
            line = LineProfile(trace[0][0], trace[0][1], trace[-1][0], trace[-1][1],
                               identifier='line_profile_' + str(uuid.uuid4())[:8], image_name=self.image_name)  # len(self._line_profile_handler._line_profiles)+1)
            self._line_profile_handler.add_line_profile(line)

    def _get_multiaxis_widths(self):
        lateral_width = self._line_profile_handler.get_line_profile_width()
        return lateral_width, int(lateral_width * self._image.mdh['voxelsize.x'] / self._image.mdh['voxelsize.z'])

    def _interpolate_stack(self):
        raise NotImplementedError

    def _add_multiaxis(self, event=None):
        from nep_fitting.core import multiaxis_extraction
        if self._image.data.shape[2] == 1:
            logger.error('Cannot extract multiaxis profile from 2D data')
        try:
            trace = self._dsviewer.do.selection_trace
        except AttributeError:  # PYME after 2022/11
            trace = self._dsviewer.do.selection.trace

        if len(trace) > 2:
            interp_px = self._interpolate_stack()
            widths_px = self._get_multiaxis_widths()
            src = trace[0][0], trace[0][1]
            dst = trace[-1][0], trace[-1][1]
            # FIXME - currently only single color
            maprof = multiaxis_extraction.extract_multiaxis_profile(self._image.data[:, :, :, 0], interp_px,
                                                                    src, dst, widths_px, self._line_profile_handler.n)
            self._line_profile_handler.add_line_profile(maprof)

    def generate_panel(self, _pnl):
        """
        This method generates the panel to control the parameters of the overlay.
        :param _pnl: parent panel
        """
        item = afp.foldingPane(_pnl, -1, caption="Line Profile Selection", pinned=True)
        pan = wx.Panel(item, -1)
        v_sizer = wx.BoxSizer(wx.VERTICAL)
        
        #v_sizer.Add(wx.StaticText(''))

        h_sizer = wx.BoxSizer(wx.HORIZONTAL)

        h_sizer.Add(wx.StaticText(pan, -1, 'Line width:'))
        self.width_control = wx.TextCtrl(pan, -1, name='Line width',
                                         value=str(self._line_profile_handler.get_line_profile_width()))
        self.width_control.Bind(wx.EVT_KEY_UP, self._on_width_change)
        h_sizer.Add(self.width_control, 1, wx.EXPAND, 0)
        h_sizer.Add(wx.StaticText(pan, -1, 'px'))
        v_sizer.Add(h_sizer, 0, wx.EXPAND | wx.TOP | wx.BOTTOM, 2)

        btn_sizer = wx.GridSizer(rows=2, cols=2, hgap=3, vgap=2)

        add_btn = wx.Button(pan, -1, label='Add', style=wx.BU_EXACTFIT)
        add_btn.SetBitmap(wx.ArtProvider.GetBitmap(wx.ART_PLUS, wx.ART_TOOLBAR, (16,16)))
        add_btn.SetToolTipString('Add a profile corresponding to the current selection')
        btn_sizer.Add(add_btn, 0, wx.EXPAND)
        add_btn.Bind(wx.EVT_BUTTON, self._add_line)

        # add_multi_btn = wx.Button(pan, -1, label='Add Multi-Axis', style=wx.BU_EXACTFIT)
        # add_multi_btn.SetBitmap(wx.ArtProvider.GetBitmap(wx.ART_PLUS, wx.ART_TOOLBAR, (16, 16)))
        # add_multi_btn.SetToolTipString('Add a multi-axis profile corresponding to the current selection')
        # btn_sizer.Add(add_multi_btn, 0, wx.EXPAND)
        # add_multi_btn.Bind(wx.EVT_BUTTON, self._add_multiaxis)

        del_btn = wx.Button(pan, -1, label='Delete', style=wx.BU_EXACTFIT)
        del_btn.SetToolTipString('Delete the currently selected profile(s)')
        del_btn.SetBitmap(wx.ArtProvider.GetBitmap(wx.ART_MINUS, wx.ART_TOOLBAR, (16,16)))
        btn_sizer.Add(del_btn, 0, wx.EXPAND)
        del_btn.Bind(wx.EVT_BUTTON, lambda e: self._list_control.delete_line_profile())

        visibility_btn = wx.Button(pan, -1, label='Show / Hide', style=wx.BU_EXACTFIT)
        visibility_btn.SetToolTipString('Toggle the visibility of the selected profile(s)')
        btn_sizer.Add(visibility_btn, 0, wx.EXPAND)
        visibility_btn.Bind(wx.EVT_BUTTON, lambda e: self._list_control.change_visibility())

        # relabel_btn = wx.Button(pan, -1, label='Relabel')
        # btn_sizer.Add(relabel_btn, 0, wx.EXPAND)
        #relabel_btn.Bind(wx.EVT_BUTTON, lambda e: self._line_profile_handler.relabel())

        

        v_sizer.Add(btn_sizer, 0, wx.EXPAND|wx.TOP, 10)
        

        self._list_control = LineProfileList(self._line_profile_handler, pan, -1, size=(-1, 300),
                                             name='Line profiles')
        v_sizer.Add(self._list_control, 0, wx.EXPAND|wx.TOP|wx.BOTTOM, 2)


        bottom_btn_sizer = wx.GridSizer(rows=1, cols=2, hgap=3, vgap=2)

        save_btn = wx.Button(pan, -1, label='Save')
        save_btn.SetBitmap(wx.ArtProvider.GetBitmap(wx.ART_FILE_SAVE_AS, wx.ART_TOOLBAR, (16,16)))
        bottom_btn_sizer.Add(save_btn, 0, wx.EXPAND)
        save_btn.Bind(wx.EVT_BUTTON, lambda e: self._on_save())

        load_btn = wx.Button(pan, -1, label='Load')
        load_btn.SetBitmap(wx.ArtProvider.GetBitmap(wx.ART_FILE_OPEN, wx.ART_TOOLBAR, (16,16)))
        bottom_btn_sizer.Add(load_btn, 0, wx.EXPAND)
        load_btn.Bind(wx.EVT_BUTTON, lambda e: self._on_load())

        v_sizer.Add(bottom_btn_sizer)

        pan.SetSizerAndFit(v_sizer)
        pan.Layout()
        item.AddNewElement(pan)
        _pnl.AddPane(item)

        item = afp.foldingPane(_pnl, -1, caption="Independent Profile Fitting", pinned=True)
        pan = wx.Panel(item, -1)
        v_sizer = wx.BoxSizer(wx.VERTICAL)

        fit_btn = wx.Button(pan, -1, label='Fit Profiles')
        v_sizer.Add(fit_btn, 0, wx.EXPAND|wx.ALL, 2)
        fit_btn.Bind(wx.EVT_BUTTON, lambda e: self._on_fit())

        pan.SetSizerAndFit(v_sizer)
        pan.Layout()
        item.AddNewElement(pan)
        _pnl.AddPane(item)

        item = afp.foldingPane(_pnl, -1, caption="Ensemble Profile Fitting", pinned=True)
        pan = wx.Panel(item, -1)
        v_sizer = wx.BoxSizer(wx.VERTICAL)

        #btn_sizer = wx.GridSizer(rows=2, cols=2, hgap=3, vgap=2)

        

        ensemble_btn = wx.Button(pan, -1, label='Ensemble Fit Profiles')
        v_sizer.Add(ensemble_btn, 0, wx.EXPAND|wx.ALL, 2)
        ensemble_btn.Bind(wx.EVT_BUTTON, lambda e: self._on_ensemble_fit())

        ensemble_test_btn = wx.Button(pan, -1, label='Test Ensemble Values')
        v_sizer.Add(ensemble_test_btn, 0, wx.EXPAND|wx.ALL, 2)
        ensemble_test_btn.Bind(wx.EVT_BUTTON, lambda e: self._on_ensemble_test())

        #v_sizer.Add(btn_sizer)
        
        pan.SetSizerAndFit(v_sizer)
        pan.Layout()
        item.AddNewElement(pan)
        _pnl.AddPane(item)

    def DrawOverlays(self, view, dc):
        """
        This is a defined callback function to render the overlay image
        Parameters
        ----------
        view
        dc

        Returns
        -------

        """
    
        # draw lines again
        line_width = self._line_profile_handler.get_line_profile_width()
        dc.SetPen(wx.Pen(LineProfilesOverlay.PEN_COLOR_LINE, max(view.scale*line_width, 1)))
        dc.SetTextForeground(LineProfilesOverlay.PEN_COLOR_LABEL)
        
        self._line_profile_handler.update_names() #what does this do?
        
        for line_profile, visible in zip(self._line_profile_handler.get_line_profiles(),
                                         self._line_profile_handler.get_visibility_mask()):
            if visible:
                self._draw_line(dc, view, line_profile, label=str(self._line_profile_handler._names[line_profile.get_id()]))

    def _draw_line(self, dc, view, line, label=None):
        """
        This method actually draws a line given line on the image.

        Therefore it transforms the given pixel coordinates to screen coordinates.
        It doesn't snap to pixels so far.
        Parameters
        ----------
        line : LineProfile
            has a start and end coordinate in pixel coordinates

        Returns
        -------

        """
        screen_coordinates_start = view.pixel_to_screen_coordinates(*line.get_start())
        screen_coordinates_end = view.pixel_to_screen_coordinates(*line.get_end())

        dc.DrawLine(screen_coordinates_start[0],
                          screen_coordinates_start[1],
                          screen_coordinates_end[0],
                          screen_coordinates_end[1])
        if label is None:
            label = line.get_id()
            
        dc.DrawText(str(label),
                          screen_coordinates_start[0] +
                          (screen_coordinates_end[0] - screen_coordinates_start[0]) * 0.5,
                          screen_coordinates_start[1] +
                          (screen_coordinates_end[1] - screen_coordinates_start[1]) * 0.5)

    def _on_width_change(self, event):
        """

        :param event: wxpython event, the EventObject carries the value that should be used for the line width
        :return:
        """

        value = event.EventObject.Value
        try:
            # profile extraction takes an int, cast the GUI input and update the GUI to keep things in sync
            value = int(float(value))
            if value > 0:
                self._line_profile_handler.set_line_profile_width(value)
                self.width_control.SetValue(str(self._line_profile_handler.get_line_profile_width()))
                self.width_control.Refresh()
                self._refresh()
        except ValueError:
            pass

    def _refresh(self, sender=None, **kwargs):
        """
        This method should be called, when there's changes in the data model.
        That leads to updating the drawn lines on the overlay image and the list of in the gui.
        sender and **kwargs are unused, but need to be here to match the raised event.
        Returns
        -------

        """
 
        self._dsviewer.Refresh()

    def _on_ensemble_fit(self, event=None):
        from PYME.recipes.base import ModuleCollection
        from nep_fitting.recipe_modules import nep_fits
        from nep_fitting import reports
        from PYME.IO.ragged import RaggedCache
        from PYME.IO.FileUtils import nameUtils
        import webbrowser

        rec = ModuleCollection()
        fitting_module = nep_fits.EnsembleFitProfiles(rec, inputName='line_profiles',
                                                           fit_type=list(profile_fitters.ensemble_fitters.keys())[0],
                                                           hold_ensemble_parameter_constant=False, outputName='output', outputFitter='fitter')
        rec.add_module(fitting_module)

        # populate namespace with current profiles
        rec.namespace['line_profiles'] = RaggedCache(self._line_profile_handler.get_line_profiles())
        if not rec.configure_traits(view=rec.pipeline_view, kind='modal'):
            return  # handle cancel

        res = rec.execute()
        
        #print ensemble_results
        #print('Ensemble parameter fitted as: %3.2f+-%3.2f nm' %(res['ensemble_parameter'][0], res['ensemble_uncertainty'][0]))

        fdialog = wx.FileDialog(None, 'Save results as ...',
                                wildcard='hdf (*.hdf)|*.hdf', style=wx.FD_SAVE,
                                defaultDir=nameUtils.genShiftFieldDirectoryPath())  # , defaultFile=defFile)
        succ = fdialog.ShowModal()
        if (succ == wx.ID_OK):
            # extract the ensemble parameters into an easy to parse format
            e_res = res['ensemble_parameter'][0]
            ensemble_params = e_res.dtype.names
            ensemble_results = {name: (e_res[name], res['ensemble_uncertainty'][0][name]) for name in ensemble_params}

            base_path = os.path.splitext(fdialog.GetPath())[0]
            
            res.to_hdf(base_path + '.hdf', tablename='profile_fits') #table name changed to avoid conflicts with standard fit data

            # plot individual profiles
            # fitter = profile_fitters.ensemble_fitters[fitting_module.fit_type](self._line_profile_handler)
            #  fitter = rec.modules[0].fitter  # TODO - move plot_results out from class so we don't have to hack like this
            fitter = rec.namespace['fitter']
            profile_dir = base_path + '/'
            os.mkdir(profile_dir)
            fitter.plot_results(profile_dir)  # FIXME - need multi-plot output recipe module

            htmlfn = base_path + '.html'
            
            context = {'ensemble_results' : ensemble_results,
                       'results' : res,
                       'filename' : self._dsviewer.image.filename,
                       'fittype' : res.mdh['EnsembleFitProfiles.FitType'], #ef_mod.fit_type,
                       'img_schematic' : reports.img_as_strb64(reports.get_schematic(res.mdh['EnsembleFitProfiles.FitType']))
            }
            handler_names = self._line_profile_handler.get_image_names()
            if (len(handler_names) == 1) and (handler_names[0] == self.image_name):
                # if there's only a single image, include it in the report
                context['img_data'] = reports.img_as_strb64(self._dsviewer.view.GrabPNGToBuffer())
            
            reports.generate_and_save(htmlfn, context, template_name='single_data.html')
            
            webbrowser.open('file://' + htmlfn, 2)

    def _on_ensemble_test(self, event=None):
        from PYME.recipes.base import ModuleCollection
        from PYME.IO.FileUtils import nameUtils
        from nep_fitting.recipe_modules import nep_fits
        from nep_fitting import reports
        from PYME.IO.ragged import RaggedCache
        import webbrowser

        rec = ModuleCollection()

        rec.add_module(nep_fits.TestEnsembleParameters(rec, inputName='line_profiles',
                                                           fit_type=list(profile_fitters.ensemble_fitters.keys())[0],
                                                           hold_ensemble_parameter_constant=False, outputName='output'))

        # populate namespace with current profiles
        rec.namespace['line_profiles'] = RaggedCache(self._line_profile_handler.get_line_profiles())
        if not rec.configure_traits(view=rec.pipeline_view, kind='modal'):
            return  # handle cancel

        res = rec.execute()

        # generate report
        context = {
            #'ensemble_parameters': res.mdh['TestEnsembleParameters.EnsembleTestValues'],  # note this is a dict
            'results': res,
            'filename': self._dsviewer.image.filename,
            'fittype': res.mdh['TestEnsembleParameters.FitType'],
            'img_schematic': reports.img_as_strb64(reports.get_schematic(res.mdh['TestEnsembleParameters.FitType']))
        }
        handler_names = self._line_profile_handler.get_image_names()
        if (len(handler_names) == 1) and (handler_names[0] == self.image_name):
            # if there's only a single image, include it in the report
            context['img_data'] = reports.img_as_strb64(self._dsviewer.view.GrabPNGToBuffer())

        fdialog = wx.FileDialog(None, 'Save report as ...',
                                wildcard='html (*.html)|*.html', style=wx.FD_SAVE,
                                defaultDir=nameUtils.genShiftFieldDirectoryPath())
        succ = fdialog.ShowModal()
        if (succ == wx.ID_OK):
            fpath = os.path.splitext(fdialog.GetPath())[0] + '.html'
            reports.generate_and_save(fpath, context, template_name='ensemble_test.html')

            webbrowser.open('file://' + fpath, 2)

    def _on_fit(self, event=None):
        from PYME.recipes.base import ModuleCollection
        from nep_fitting.recipe_modules import nep_fits
        from PYME.IO.ragged import RaggedCache
        from PYME.IO.FileUtils import nameUtils
        import webbrowser
        from nep_fitting import reports

        rec = ModuleCollection()

        rec.add_module(nep_fits.FitProfiles(rec, inputName='line_profiles',
                                            fit_type=list(profile_fitters.non_ensemble_fitters.keys())[0], 
                                            outputName='output', outputFitter='fitter'))

        # populate namespace with current profiles
        rec.namespace['line_profiles'] = RaggedCache(self._line_profile_handler.get_line_profiles())
        if not rec.configure_traits(view=rec.pipeline_view, kind='modal'):
            return  # handle cancel

        res = rec.execute()

        fdialog = wx.FileDialog(None, 'Save results as ...',
                                wildcard='hdf (*.hdf)|*.hdf', style=wx.FD_SAVE,
                                defaultDir=nameUtils.genShiftFieldDirectoryPath())  # , defaultFile=defFile)
        succ = fdialog.ShowModal()
        if (succ == wx.ID_OK):
            base_path = os.path.splitext(fdialog.GetPath())[0]

            res.to_hdf(base_path + '.hdf', tablename='profile_fits')  # table name changed to avoid conflicts with standard fit data

            fitter = rec.namespace['fitter']  # TODO - adjust naming
            profile_dir = base_path + '/'
            os.mkdir(profile_dir)
            fitter.plot_results(profile_dir)

            htmlfn = base_path + '.html'

            context = {'results': res,
                       'filename': self._dsviewer.image.filename,
                       'fittype': res.mdh['FitProfiles.FitType'],
                       'img_schematic': reports.img_as_strb64(reports.get_schematic(res.mdh['FitProfiles.FitType']))
            }
            handler_names = self._line_profile_handler.get_image_names()
            if (len(handler_names) == 1) and (handler_names[0] == self.image_name):
                # if there's only a single image, include it in the report
                context['img_data'] = reports.img_as_strb64(self._dsviewer.view.GrabPNGToBuffer())

            reports.generate_and_save(htmlfn, context, template_name='single_data.html')

            webbrowser.open('file://' + htmlfn, 2)

    def _on_save(self, event=None):
        from PYME.IO.FileUtils import nameUtils
        
        fdialog = wx.FileDialog(None, 'Save/Append Line Profiles to ...',
                                wildcard='HDF5 Tables (*.hdf)|*.hdf', style=wx.FD_SAVE,
                                defaultDir=nameUtils.genHDFDataFilepath())  # , defaultFile=defFile)
        succ = fdialog.ShowModal()
        if (succ == wx.ID_OK):
            fpath = fdialog.GetPath()
            if os.path.exists(fpath):
                if not wx.MessageBox('Continue saving and append profiles to whatever else is in this file?', 'Output file exists', wx.YES|wx.CANCEL) == wx.YES:
                    return
            self._line_profile_handler.save_line_profiles(fpath)

    def _on_load(self, event=None):
        fdialog = wx.FileDialog(None, 'Load Line Profiles from ...',
                                wildcard='HDF5 Tables (*.hdf)|*.hdf| ImageJ roi (*.roi;*.zip)|*.roi;*.zip', style=wx.FD_OPEN)
        succ = fdialog.ShowModal()
        if (succ == wx.ID_OK):
            fpath = fdialog.GetPath()
            ext = os.path.splitext(fpath)[-1]
            if ext == '.roi' or ext == '.zip':
                self._line_profile_handler._load_profiles_from_imagej(fpath)
            else:
                self._line_profile_handler.open_line_profiles(fpath)

    def Unplug(self):
        self._do.overlays.remove(self.DrawOverlays)
        self._dsviewer.paneHooks.remove(self.generate_panel)

class LineProfileList(wx.ListCtrl):
    def __init__(self, line_profile_handler, *args, **kwargs):
        """
        This list control displays line profiles of a lineProfileHandler
        Parameters
        ----------
        line_profile_handler : handler whos list should be displayed
        args : parameters of wx.ListCtr
        kwargs : parameters of wx.ListCtr
        """
        super(LineProfileList, self).__init__(*args, style=wx.LC_REPORT|wx.BORDER_SUNKEN|wx.LC_VIRTUAL|wx.LC_VRULES, **kwargs)
        self._line_profile_handler = line_profile_handler

        self.InsertColumn(0, "Profile", width=120)
        self.InsertColumn(1, "Visible", width=50)
        self._line_profile_handler.LIST_CHANGED_SIGNAL.connect(self._update_list)

    def _update_list(self, sender=None, **kwargs):
        # relist all items in the gui
        # self.DeleteAllItems()
        # self._line_profile_handler.update_names()
        # for line_profile in self._line_profile_handler.get_line_profiles():
        #     lp_id = line_profile.get_id()
        #     index = self._line_profile_handler._names[lp_id]
        #     self.InsertStringItem(index, str(index) + ': ' + str(lp_id))
        
        self.SetItemCount(len(self._line_profile_handler.get_line_profiles()))
        
        self.Update()
        self.Refresh()
        
    def OnGetItemText(self, item, col):
        """
        Note that this is overriding the wxListCtrl method as required for wxLC_VIRTUAL style
        Parameters
        ----------
        item : long
            wx list item
        col : long
            column specifier for wxListCtrl

        Returns
        -------
        str : Returns string of column 'col' for item 'item'

        """
        line_profile = self._line_profile_handler.get_line_profiles()[item]
        lp_id = line_profile.get_id()
        index = self._line_profile_handler._names[lp_id]
        
        if col == 0:
            return str(index) + ': ' + str(lp_id)
        if col == 1:
            return str(self._line_profile_handler._visibility_mask[index])
        else:
            return ''

    def delete_line_profile(self):
        selected_indices = self.get_selected_items()

        for line_profile_index in reversed(selected_indices):
            self._line_profile_handler.remove_line_profile(line_profile_index)
            
        self._line_profile_handler.relabel()


    def change_visibility(self):
        selected_indices = self.get_selected_items()

        for line_profile_index in reversed(selected_indices):
            self._line_profile_handler.change_visibility(line_profile_index)

    def get_selected_items(self):
        selection = []
        current = -1
        next = 0
        while next != -1:
            next = self.get_next_selected(current)
            if next != -1:
                selection.append(next)
                current = next
        return selection

    def get_next_selected(self, current):
        return self.GetNextItem(current, wx.LIST_NEXT_ALL, wx.LIST_STATE_SELECTED)


def Plug(dsviewer):
    dsviewer.lineProfileOverlayer = LineProfilesOverlay(dsviewer)
    dsviewer.lineProfileOverlayer._line_profile_handler.LIST_CHANGED_SIGNAL.connect(log_event)

def Unplug(dsviewer):
    dsviewer.lineProfileOverlayer.Unplug()

def log_event(sender=None, **kwargs):
    logger.debug('LIST_CHANGED_SIGNAL appeared')
