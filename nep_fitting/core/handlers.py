import logging
from skimage.measure import profile_line
import numpy as np

from PYME.IO.MetaDataHandler import NestedClassMDHandler
import dispatch

from nep_fitting.core import rois


logger = logging.getLogger(__name__)

class LineProfileHandler(object):
    LIST_CHANGED_SIGNAL = dispatch.Signal()

    def __init__(self, image_data=None, interpolation_order=3, image_name=None):
        self._line_profiles = []
        self._names = {}
        self._visibility_mask = []
        self.image = image_data
        self.image_name = image_name
        self.interpolation_order = interpolation_order
        self._width = 1.0
        if self.image is not None:
            self.mdh = NestedClassMDHandler(self.image.mdh)
        else:
            self.mdh = None

    def profiles_as_dict(self):
        if self._line_profiles[-1].get_data() is None:
            self.calculate_profiles()

        d = {}
        for p in self._line_profiles:
            d[p.get_id()] = p

        return d

    def update_names(self, relabel=False):
        if relabel:
            # sort self._line_profiles list according to visibility
            profiles, visibilities = [], []
            #self._line_profiles = [lp for (vis, lp) in sorted(zip(self.get_visibility_mask(), self._line_profiles), reverse=True)]
            for (vis, lp) in sorted(zip(self.get_visibility_mask(), self._line_profiles), reverse=True):
                profiles.append(lp)
                visibilities.append(vis)

            self._line_profiles = profiles
            self._visibility_mask = visibilities

        index = 0
        for lp in self.get_line_profiles():
            self._names[lp.get_id()] = index
            index += 1

    def LineProfile_from_array(self, parray, table_name=None):
        lp = rois.LineProfile(parray['r1'], parray['c1'], parray['r2'], parray['c2'],
                         parray['slice'], parray['width'], identifier=table_name)
        lp._profile = parray['profile']
        lp._distance = parray['distance']
        return lp

    def save_line_profiles(self, filename, tablename='profiles'):
        from PYME.IO import ragged

        if self._line_profiles[-1].get_data() is None:
            self.calculate_profiles()

        profs = ragged.RaggedCache(self._line_profiles)  # [lp.as_dict() for lp in self._line_profiles])

        profs.to_hdf(filename, tablename, mode='a')

    def _load_profiles_from_list(self, ragged):
        for lp in ragged:
            if isinstance(lp, rois.LineProfile):
                self.add_line_profile(lp, update=False)  # we might not be using a GUI
            elif type(lp) is dict:  # recreate the line profile
                if lp['class'] == 'LineProfile':
                    input_lp = lp.copy()
                    del input_lp['class']
                    self.add_line_profile(rois.LineProfile(**input_lp), update=False)

    def _load_profiles_from_array(self, parray):
        profile_count = parray.shape[0]
        for pi in range(profile_count):
            lp = rois.LineProfile(parray[pi]['r1'], parray[pi]['c1'], parray[pi]['r2'], parray[pi]['c2'],
                             parray[pi]['slice'], parray[pi]['width'], identifier=pi)
            lp._profile = parray[pi]['profile']
            lp._distance = parray[pi]['distance']
            # self.add_line_profile(lp)
            # skip GUI calls and just add profile
            self._line_profiles.append(lp)

    def _load_profiles_from_dict(self, d, table_name_filter='line_profile'):
        profile_keys = [key for key in d.keys() if key.startswith(table_name_filter)]
        for key in profile_keys:
            # skip GUI calls and just add profile
            #try:
            self._line_profiles.append(self.LineProfile_from_array(d[key], table_name=key))
            # assume that if something went wrong, something was pulled from the namespace which was not a profile
            #except:
            #    pass

    def open_line_profiles(self, hdfFile):
        import tables
        from PYME.IO import ragged
        if type(hdfFile) == tables.file.File:
            hdf = hdfFile
        else:
            hdf = tables.open_file(hdfFile)

        for t in hdf.list_nodes('/'):
            #print t.name, t.__class__
            if isinstance(t, tables.vlarray.VLArray):
                profs = ragged.RaggedVLArray(hdf, t.name)
                print profs[0], profs[0].__class__
                for i in range(len(profs)):# in profs[:]:
                    p = profs[i]
                    #print p.__class__
                    #print p
                    lp = rois.LineProfile(p['r1'], p['c1'], p['r2'], p['c2'], p['slice'], p['width'],
                                          identifier=p['identifier'])
                    lp._profile = p['profile']
                    lp._distance = p['distance']

                    self.add_line_profile(lp, False)
                    
        LineProfileHandler.LIST_CHANGED_SIGNAL.send(sender=self)

                # TODO - check if we need to worry about multiple profiles with the same id
        hdf.close()

    def get_line_profiles(self):
        try:
            self.calculate_profiles()
        except AttributeError:
            logger.debug('No image data provided to calculate profile from - may result in error if profile has not been calculated')
        return self._line_profiles

    def get_visibility_mask(self):
        return self._visibility_mask

    def change_visibility(self, index):
        self._visibility_mask[index] = not self._visibility_mask[index]
        self._on_list_changed()

    def relabel_line_profiles(self):
        self.update_names(relabel=True)
        # label_no = 1
        # for line_profile in self.get_line_profiles():
        #     line_profile.set_id(label_no)
        #     label_no += 1
        LineProfileHandler.LIST_CHANGED_SIGNAL.send(sender=self)

    def calculate_profiles(self):
        """
        Loops over all start/end points held by the handler and calculates the x- and y-axis of each line profile.
        Currently enforces that each line in the handler must have the same width.
        Returns
        -------

        """
        for lp in self._line_profiles:
            if np.any(lp._profile) and lp._width == self._width:
                continue
            lp._profile = profile_line(self.image.data.getSlice(lp._slice), (lp._r1, lp._c1), (lp._r2, lp._c2),
                                       order=self.interpolation_order, linewidth=self._width)
            lp.set_width(self._width)

            dist = np.sqrt((self.image.voxelsize.x * (lp._r2 - lp._r1)) ** 2 + (self.image.voxelsize.y * (lp._c2 - lp._c1)) ** 2)  # [nm]

            lp._distance = np.linspace(0, dist, lp._profile.shape[0])

    def add_line_profile(self, line_profile, update=True):
        self._line_profiles.append(line_profile)
        self._visibility_mask.append(line_profile.get_image_name() == self.image_name)
        if update:
            LineProfileHandler.LIST_CHANGED_SIGNAL.send(sender=self)


    def remove_line_profile(self, index):
        del self._line_profiles[index]
        del self._visibility_mask[index]
        LineProfileHandler.LIST_CHANGED_SIGNAL.send(sender=self)

    def set_line_profile_width(self, width):
        """

        Parameters
        ----------
        width : float
            width of the line profile in data-pixel

        Returns
        -------

        """
        self._width = width

    def get_line_profile_width(self):
        return self._width

    def _on_list_changed(self):
        LineProfileHandler.LIST_CHANGED_SIGNAL.send(sender=self)

class RegionHandler(object):
    LIST_CHANGED_SIGNAL = dispatch.Signal() #TODO - why is this a class level variable???

    def __init__(self, image_data=None, interpolation_order=3):
        self._rois = []
        self._names = {}
        self._visibility_mask = []
        self.image = image_data
        self.interpolation_order = interpolation_order
    
        if self.image is not None:
            self.mdh = NestedClassMDHandler(self.image.mdh)
        else:
            self.mdh = None

    def extract_rois(self):
        """

        """
        # fixme - this only works for rectangular ROIs
        for roi in self._rois:
            min_row, max_row = sorted([roi._r1, roi._r2])
            min_col, max_col = sorted([roi._c1, roi._c2])
            roi._data = self.image.data.getSlice(roi._slice)[min_row:max_row, min_col:max_col]
        
            rows = self.image.mdh['voxelsize.x'] * np.arange(min_row, max_row)
            columns = self.image.mdh['voxelsize.y'] * np.arange(min_col, max_col)
            roi._rows, roi._columns = np.meshgrid(rows, columns, indexing='ij')

    def get_rois(self):
        try:
            self.extract_rois()
        except AttributeError:
            logger.debug(
                'No image data provided to extract rois from - may result in error if data has not already been calculated')
        return self._rois

    def get_visibility_mask(self):
        return self._visibility_mask

    def change_visibility(self, index):
        self._visibility_mask[index] = not self._visibility_mask[index]
        self._on_list_changed()

    def relabel_line_profiles(self):
        self.update_names(relabel=True)
        # label_no = 1
        # for line_profile in self.get_line_profiles():
        #     line_profile.set_id(label_no)
        #     label_no += 1
        RegionHandler.LIST_CHANGED_SIGNAL.send(sender=self)

    def add_roi(self, roi, gui=True):
        self._rois.append(roi)
        if gui:
            self._visibility_mask.append(True)
            RegionHandler.LIST_CHANGED_SIGNAL.send(sender=self)

    def remove_roi(self, index):
        del self._rois[index]
        del self._visibility_mask[index]
        RegionHandler.LIST_CHANGED_SIGNAL.send(sender=self)

    def update_names(self, relabel=False):
        if relabel:
            # sort self._rois list according to visibility
            rois, visibilities = [], []
            #self._line_profiles = [lp for (vis, lp) in sorted(zip(self.get_visibility_mask(), self._line_profiles), reverse=True)]
            for (vis, lp) in sorted(zip(self.get_visibility_mask(), self._rois), reverse=True):
                rois.append(lp)
                visibilities.append(vis)
        
            self._rois = rois
            self._visibility_mask = visibilities
    
        index = 0
        for roi in self.get_rois():
            self._names[roi.get_id()] = index
            index += 1

    def save_rois(self, filename, tablename='ROIs'):
        from PYME.IO import ragged
    
        try:
            len(self._rois[-1].get_data())
        except TypeError:
            self.extract_rois()
    
        profs = ragged.RaggedCache(self._rois)  # [lp.as_dict() for lp in self._line_profiles])
    
        profs.to_hdf(filename, tablename, mode='a')

    def _load_from_list(self, ragged):
        for roi in ragged:
            if type(roi) is dict:
                # recreate the instance by dynamically loading the class
                mod = __import__(rois.module_lookup[roi['class']], fromlist=[roi['class']])
                input_roi = roi.copy()
                del input_roi['class']
                roi = getattr(mod, roi['class'])(**input_roi)
        
            # if roi is not a dictionary, assume it is already the correct class
            self.add_roi(roi, gui=False)

    def _on_list_changed(self):
        RegionHandler.LIST_CHANGED_SIGNAL.send(sender=self)