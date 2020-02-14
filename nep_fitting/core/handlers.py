import logging
from skimage.measure import profile_line
import numpy as np

from PYME.IO.MetaDataHandler import NestedClassMDHandler
import dispatch

from nep_fitting.core import rois


logger = logging.getLogger(__name__)

def min_max(profiles):
    normed = []
    for profile in profiles:
        peak, off = profile.max(), profile.min()
        normed.append((profile - off) / (peak - off))
    return normed


class BaseHandler(object):
    def __init__(self):
        self.LIST_CHANGED_SIGNAL = dispatch.Signal()

        self._visibility_mask = []
        self._names = {}
        self._rois = []
        self._partial_ensemble_parameters = np.array([])

    @property
    def n(self):
        return len(self._rois)

    @property
    def partial_ensemble_parameters(self):
        self.update_partial_ensemble_parameters()
        return self._partial_ensemble_parameters

    def update_partial_ensemble_parameters(self):
        params = []
        for ind, roi in enumerate(self._rois):
            try:
                params.append(roi.parameter_exchanges[1])
            except AttributeError:
                pass
        peps, indices = np.unique(params, return_index=True)
        self._partial_ensemble_parameters = {}
        for ind in range(len(peps)):
            self._partial_ensemble_parameters[peps[ind]] = indices

    def get_visibility_mask(self):
        return self._visibility_mask

    def change_visibility(self, index):
        self._visibility_mask[index] = not self._visibility_mask[index]
        self._on_list_changed()

    def _on_list_changed(self):
        self.LIST_CHANGED_SIGNAL.send(sender=self)

    def update_names(self, relabel=False):
        if relabel:
            # sort self._line_profiles list according to visibility
            rois, visibilities = [], []
            tie_breaker = range(len(self._rois))
            for (vis, tb, roi) in sorted(zip(self.get_visibility_mask(), tie_breaker, self._rois), reverse=True):
                rois.append(roi)
                visibilities.append(vis)

            self._rois = rois
            self._visibility_mask = visibilities

        index = 0
        for lp in self._rois:
            self._names[lp.get_id()] = index
            index += 1

    def relabel(self):
        self.update_names(relabel=True)
        # label_no = 1
        # for line_profile in self.get_line_profiles():
        #     line_profile.set_id(label_no)
        #     label_no += 1
        self.LIST_CHANGED_SIGNAL.send(sender=self)

class LineProfileHandler(BaseHandler):
    def __init__(self, image_data=None, interpolation_order=3, image_name=None):
        super(self.__class__, self).__init__()
        self.image = image_data
        self.image_name = image_name
        self.interpolation_order = interpolation_order
        self._width = 1.0
        if self.image is not None:
            self.mdh = NestedClassMDHandler(self.image.mdh)
        else:
            self.mdh = None

    @property
    def line_profiles(self):
        return self._rois

    @line_profiles.setter
    def line_profiles(self, profiles):
        self._rois = [p for p in profiles]  # type _rois as a list

    def get_rois(self):
        return self.get_line_profiles()

    def profiles_as_dict(self):
        if self._rois[-1].get_data() is None:
            self.calculate_profiles()

        d = {}
        for p in self._rois:
            d[p.get_id()] = p

        return d

    def LineProfile_from_array(self, parray, table_name=None):
        lp = rois.LineProfile(parray['r1'], parray['c1'], parray['r2'], parray['c2'],
                         parray['slice'], parray['width'], identifier=table_name)
        lp.set_profile(parray['profile'])
        lp.set_distance(parray['distance'])
        return lp

    def save_line_profiles(self, filename, tablename='profiles'):
        from PYME.IO import ragged

        if self.line_profiles[-1].get_data() is None:
            self.calculate_profiles()

        profs = ragged.RaggedCache(self.line_profiles)  # [lp.as_dict() for lp in self._line_profiles])

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
            lp.set_profile(parray[pi]['profile'])
            lp.set_distance(parray[pi]['distance'])
            # self.add_line_profile(lp)
            # skip GUI calls and just add profile
            self._rois.append(lp)

    def _load_profiles_from_dict(self, d, table_name_filter='line_profile'):
        profile_keys = [key for key in d.keys() if key.startswith(table_name_filter)]
        for key in profile_keys:
            # skip GUI calls and just add profile
            #try:
            self._rois.append(self.LineProfile_from_array(d[key], table_name=key))
            # assume that if something went wrong, something was pulled from the namespace which was not a profile
            #except:
            #    pass

    def _load_profiles_from_imagej(self, fn, zip_file=True):
        """
        Read line profiles from Fiji/ImageJ.

        Parameters
        ----------
            fn : str
                Filename containing line profiles.
            zip_file : bool
                Read profiles from zip file (default ImageJ output).
        """
        try:
            import read_roi
        except(ImportError):
            raise ImportError('Please install the read_roi package (https://pypi.org/project/read-roi/).')

        read_rois = read_roi.read_roi_zip
        if not zip_file:
            read_rois = read_roi.read_roi_file
        
        imagej_rois = read_rois(fn)

        for key in imagej_rois.keys():
            roi = imagej_rois[key]
            # x, y transposed in Fiji/ImageJ
            # TODO: Don't hardcode slice
            self._rois.append(rois.LineProfile(roi['y1'], roi['x1'],
                              roi['y2'], roi['x2'], slice=0,
                              width=roi['width'], identifier=roi['name'], 
                              image_name=self.image_name))

    def open_line_profiles(self, hdfFile, gui=True):
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
                for i in range(len(profs)):# in profs[:]:
                    p = profs[i]
                    identifier = str(i) if not p['identifier'] else p['identifier']
                    if p['class'] == 'LineProfile':
                        lp = rois.LineProfile(p['r1'], p['c1'], p['r2'], p['c2'], p['slice'], p['width'],
                                              identifier=identifier, image_name=p['image_name'])
                        lp.set_profile(p['profile'])
                        lp.set_distance(p['distance'])
                    else:  # p['class'] == 'MultiaxisProfile':
                        profiles = []
                        positions = []
                        n_prof = 0
                        for k in p.keys():
                            if len(k.split('~')) > 1:
                                n_prof = max(n_prof, int(k.split('~')[-1]) + 1)

                        for pi in range(n_prof):
                            profiles.append(np.asarray(p['profiles~%d' % pi]))
                            positions.append(np.asarray(p['positions~%d' % pi]))
                        lp = rois.MultiaxisProfile(profiles, positions, p['widths'], identifier=identifier,
                                                   image_name=p['image_name'])

                    self.add_line_profile(lp, False)

        if gui:
            self.update_names(relabel=True)
            self._on_list_changed()

                # TODO - check if we need to worry about multiple profiles with the same id
        hdf.close()

    def get_line_profiles(self):
        try:
            self.calculate_profiles()
        except AttributeError:
            logger.debug('No image data provided to calculate profile from - may result in error if profile has not been calculated')
        return self._rois

    def calculate_profiles(self):
        """
        Loops over all start/end points held by the handler and calculates the x- and y-axis of each line profile.
        Currently enforces that each line in the handler must have the same width.
        Returns
        -------

        """
        for lp in self._rois:
            # skip if already calculated, or if profile was extracted from different image
            if (np.any(lp._profile) and lp._width == self._width) or (lp.get_image_name() != self.image_name):
                continue
            lp.set_profile(profile_line(self.image.data.getSlice(lp._slice), (lp._r1, lp._c1), (lp._r2, lp._c2),
                                       order=self.interpolation_order, linewidth=self._width))
            lp.set_width(self._width)

            dist = np.sqrt((self.image.voxelsize.x * (lp._r2 - lp._r1)) ** 2 + (self.image.voxelsize.y * (lp._c2 - lp._c1)) ** 2)  # [nm]

            lp.set_distance(np.linspace(0, dist, lp.get_data().shape[0]))

    def add_line_profile(self, line_profile, update=True):
        self._rois.append(line_profile)
        self._visibility_mask.append(line_profile.get_image_name() == self.image_name)
        self.update_partial_ensemble_parameters()
        if update:
            self.update_names(relabel=True)
            self.LIST_CHANGED_SIGNAL.send(sender=self)


    def remove_line_profile(self, index):
        del self._rois[index]
        del self._visibility_mask[index]
        self.LIST_CHANGED_SIGNAL.send(sender=self)

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

    def get_image_names(self):
        """

        Returns
        -------
        names : array
            Array of unique image names from which profiles were extracted.
        """
        profs = self.get_line_profiles()
        names = np.unique([p.get_image_name() for p in profs])
        return names

    def profile_by_index(self, index):
        return self._rois[index]


    def _load_multiaxis_from_csv(self, filenames, delimiter=',', distance_in_um=False, min_max_normalize=True):
        from nep_fitting.core import rois
        data = [np.genfromtxt(fn, delimiter=delimiter, usemask=True) for fn in filenames]
        n = int(0.5 * data[0].shape[0])
        for pi in range(n):
            profiles = []
            positions = []
            for axis in range(len(data)):
                if distance_in_um:
                    positions.append(1e3 * data[axis][2 * pi, :].compressed())
                else:
                    positions.append(data[axis][2 * pi, :].compressed())
                profiles.append(data[axis][2 * pi + 1, :].compressed())

            if min_max_normalize:
                profiles = min_max(profiles)
            p = rois.MultiaxisProfile(profiles, positions)
            p.data = positions, profiles

            self.add_line_profile(p, update=False)  # fixme- should update=True once we fix sorting in handler on py3

class RegionHandler(BaseHandler):
    def __init__(self, image_data=None, interpolation_order=3):
        super(self.__class__, self).__init__()
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

    def add_roi(self, roi, gui=True):
        self._rois.append(roi)
        if gui:
            self._visibility_mask.append(True)
            self.LIST_CHANGED_SIGNAL.send(sender=self)

    def remove_roi(self, index):
        del self._rois[index]
        del self._visibility_mask[index]
        self.LIST_CHANGED_SIGNAL.send(sender=self)

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
