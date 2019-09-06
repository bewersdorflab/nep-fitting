import wx
import logging
import wx
import uuid
import PYME.ui.autoFoldPanel as afp

logger = logging.getLogger(__name__)

from nep_fitting.core.handlers import RegionHandler
from nep_fitting.core.rois import RectangularROI

class RegionManager:
    PEN_COLOR_LINE = wx.GREEN
    PEN_COLOR_LABEL = wx.BLUE
    """
    This is the general handler for the gui stuff to handle lineprofiles.
    """

    def __init__(self, dsviewer):
        self._dsviewer = dsviewer
        self._view = dsviewer.view
        self._do = dsviewer.do
        self._image = dsviewer.image
        filename = self._image.filename
        self.image_name = filename.split('/')[-1].strip('.' + filename.split('.')[-1])
        # create LineProfileHandler with reference to ImageStack
        self._region_handler = RegionHandler(self._image)

        # add this overlay to the overlays to be rendered
        self._do.overlays.append(self.DrawOverlays)

        # add a gui panel to the window to control the values
        self._dsviewer.paneHooks.append(self.generate_panel)
        RegionHandler.LIST_CHANGED_SIGNAL.connect(self._refresh)

    def _add_roi(self):
        """
        This callback function is called, whenever a new line has been drawn using the selection tool
        and should be added
        """
        # fixme - currently only set up for rectangular ROI
        selection = self._dsviewer.do.GetSliceSelection()

        if len(selection) == 4:
            roi = RectangularROI(selection[0], selection[1], selection[2], selection[3],
                                 identifier='ROI' + str(uuid.uuid4())[:8], image_name=self.image_name)
            self._region_handler.add_roi(roi)

    def generate_panel(self, _pnl):
        """
        This method generates the panel to control the parameters of the overlay.
        :param _pnl: parent panel
        """
        item = afp.foldingPane(_pnl, -1, caption="MultilineSelection", pinned=True)
        pan = wx.Panel(item, -1)
        v_sizer = wx.BoxSizer(wx.VERTICAL)

        btn_sizer = wx.GridSizer(rows=3, cols=2, hgap=3, vgap=2)
        add_btn = wx.Button(pan, -1, label='Add')
        btn_sizer.Add(add_btn, 0, wx.EXPAND)
        add_btn.Bind(wx.EVT_BUTTON, lambda e: self._add_roi())
        del_btn = wx.Button(pan, -1, label='Delete')
        btn_sizer.Add(del_btn, 0, wx.EXPAND)
        del_btn.Bind(wx.EVT_BUTTON, lambda e: self._list_control.delete_roi())

        visibility_btn = wx.Button(pan, -1, label='Visibility')
        btn_sizer.Add(visibility_btn, 0, wx.EXPAND)
        visibility_btn.Bind(wx.EVT_BUTTON, lambda e: self._list_control.change_visibility())

        relabel_btn = wx.Button(pan, -1, label='Relabel')
        btn_sizer.Add(relabel_btn, 0, wx.EXPAND)
        relabel_btn.Bind(wx.EVT_BUTTON, lambda e: self._region_handler.relabel())

        fit_btn = wx.Button(pan, -1, label='Fit')
        btn_sizer.Add(fit_btn, 0, wx.EXPAND)
        fit_btn.Bind(wx.EVT_BUTTON, lambda e: self._on_fit())

        ensemble_btn = wx.Button(pan, -1, label='Ensemble Fit')
        btn_sizer.Add(ensemble_btn, 0, wx.EXPAND)
        ensemble_btn.Bind(wx.EVT_BUTTON, lambda e: self._on_ensemble_fit())

        v_sizer.Add(btn_sizer)
        h_sizer = wx.BoxSizer(wx.HORIZONTAL)


        v_sizer.Add(h_sizer)

        self._list_control = RegionList(self._region_handler, pan, -1, size=(-1, 300),
                                             style=wx.LC_REPORT | wx.BORDER_SUNKEN,
                                             name='Line profiles')
        v_sizer.Add(self._list_control)


        bottom_btn_sizer = wx.GridSizer(rows=1, cols=2, hgap=3, vgap=2)

        save_btn = wx.Button(pan, -1, label='Save')
        bottom_btn_sizer.Add(save_btn, 0, wx.EXPAND)
        save_btn.Bind(wx.EVT_BUTTON, lambda e: self._on_save())

        load_btn = wx.Button(pan, -1, label='Load')
        bottom_btn_sizer.Add(load_btn, 0, wx.EXPAND)
        load_btn.Bind(wx.EVT_BUTTON, lambda e: self._on_load())

        v_sizer.Add(bottom_btn_sizer)

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
        self._view = view
        self._dc = dc
        self._refresh()

    def _draw_roi(self, roi, label=None):
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
        # fixme - next bit only works for rectangular ROIs
        screen_coordinates_start = self._view._PixelToScreenCoordinates(*roi.get_origin())
        screen_coordinates_end = self._view._PixelToScreenCoordinates(*roi.get_second_corner())

        x = min([screen_coordinates_start[0], screen_coordinates_end[0]])
        y = min([screen_coordinates_start[1], screen_coordinates_end[1]])
        h = int(abs(screen_coordinates_end[1] - screen_coordinates_start[1]))
        w = int(abs(screen_coordinates_end[0] - screen_coordinates_start[0]))
        self._dc.DrawRectangle(x, y, w, h)
        # self._dc.DrawLine(screen_coordinates_start[0],
        #                   screen_coordinates_start[1],
        #                   screen_coordinates_end[0],
        #                   screen_coordinates_end[1])
        if label is None:
            label = roi.get_id()
        self._dc.DrawText(str(label),
                          screen_coordinates_start[0] +
                          (screen_coordinates_end[0] - screen_coordinates_start[0]) * 0.5,
                          screen_coordinates_start[1] +
                          (screen_coordinates_end[1] - screen_coordinates_start[1]) * 0.5)

    def _refresh(self, sender=None, **kwargs):
        """
        This method should be called, when there's changes in the data model.
        That leads to updating the drawn lines on the overlay image and the list of in the gui.
        sender and **kwargs are unused, but need to be here to match the raised event.
        Returns
        -------

        """
        if self._dc and self._dc is not None:

            # draw lines again
            self._dc.SetPen(
                wx.Pen(RegionManager.PEN_COLOR_LINE, self._view._PixelToScreenCoordinates(10, 0)[0]))
            self._dc.SetTextForeground(RegionManager.PEN_COLOR_LABEL)
            self._region_handler.update_names()
            for roi, visible in zip(self._region_handler.get_rois(),
                                                self._region_handler.get_visibility_mask()):
                if visible:
                    self._draw_roi(roi, label=str(self._region_handler._names[roi.get_id()]))

        self._dsviewer.Refresh()

    def _on_ensemble_fit(self, event=None):
        from PYME.recipes.base import ModuleCollection
        from nep_fitting.core import profile_fitters
        from PYME.IO.ragged import RaggedCache

        rec = ModuleCollection()

        rec.add_module(profile_fitters.EnsembleFitROIs(rec, inputName='ROIs',
                                                           fit_type='LorentzianConvolvedSolidSphere_ensemblePSF',
                                                           hold_ensemble_parameter_constant=False, outputName='output'))

        # populate namespace with current profiles
        rec.namespace['ROIs'] = RaggedCache(self._region_handler.get_rois())
        if not rec.configure_traits(view=rec.pipeline_view, kind='modal'):
            return  # handle cancel

        res = rec.execute()

    def _on_fit(self, event=None):
        raise NotImplementedError
        # from PYME.recipes.base import ModuleCollection
        # from PYME.recipes import profile_fitting
        # from PYME.DSView.modules import profileFitting
        # from PYME.IO.ragged import RaggedCache
        #
        # rec = ModuleCollection()
        #
        # rec.add_module(profile_fitting.FitProfiles(rec, inputerName='line_profiles',
        #                                            fit_type=profileFitting.fitters.keys()[0], outputName='output'))
        #
        # # populate namespace with current profiles
        # rec.namespace['line_profiles'] = RaggedCache(self._region_handler.get_rois())
        # if not rec.configure_traits(view=rec.pipeline_view, kind='modal'):
        #     return  # handle cancel
        #
        # res = rec.execute()

    def _on_save(self, event=None):
        from PYME.IO.FileUtils import nameUtils
        fdialog = wx.FileDialog(None, 'Save/Append ROIs to ...',
                                wildcard='HDF5 Tables (*.hdf)|*.hdf', style=wx.SAVE,
                                defaultDir=nameUtils.genHDFDataFilepath())  # , defaultFile=defFile)
        succ = fdialog.ShowModal()
        if (succ == wx.ID_OK):
            fpath = fdialog.GetPath()
            self._region_handler.save_rois(fpath)

    def _on_load(self, event=None):
        raise NotImplementedError

    def Unplug(self):
        self._do.overlays.remove(self.DrawOverlays)
        self._dsviewer.paneHooks.remove(self.generate_panel)

class RegionList(wx.ListCtrl):
    def __init__(self, region_handler, *args, **kwargs):
        """
        This list control displays line profiles of a lineProfileHandler
        Parameters
        ----------
        line_profile_handler : handler whos list should be displayed
        args : parameters of wx.ListCtr
        kwargs : parameters of wx.ListCtr
        """
        super(RegionList, self).__init__(*args, **kwargs)
        self._region_handler = region_handler

        self.InsertColumn(0, "Region of Interest", width=self.Size[1])
        RegionHandler.LIST_CHANGED_SIGNAL.connect(self._update_list)

    def _update_list(self, sender=None, **kwargs):
        # relist all items in the gui
        self.DeleteAllItems()
        self._region_handler.update_names()
        for roi in self._region_handler.get_rois():
            roi_id = roi.get_id()
            index = self._region_handler._names[roi_id]
            self.InsertStringItem(index, str(index) + ': ' + str(roi_id))
        self.Update()
        self.Refresh()

    def delete_roi(self):
        selected_indices = self.get_selected_items()

        for roi_index in reversed(selected_indices):
            self._region_handler.remove_roi(roi_index)


    def change_visibility(self):
        selected_indices = self.get_selected_items()

        for roi_index in reversed(selected_indices):
            self._region_handler.change_visibility(roi_index)

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
    dsviewer.regionManager = RegionManager(dsviewer)
    RegionHandler.LIST_CHANGED_SIGNAL.connect(log_event)

def Unplug(dsviewer):
    dsviewer.regionManager.Unplug()

def log_event(sender=None, **kwargs):
    logger.debug('LIST_CHANGED_SIGNAL appeared')