import numpy as np
import logging
logger = logging.getLogger(__name__)


class BaseROI(object):
    def __init__(self, identifier=None, image_name=None):
        """

        Parameters
        ----------
        identifier : int
            number assigned to line profile
        image_name : str
            stub of image filename which line profile is extracted from
        """
        self.set_id(identifier)
        self.set_image_name(image_name)
    
    def set_id(self, identifier):
        """
        Set method for number assigned to ROI as an identifier

        Parameters
        ----------
        identifier : int
            Unique integer used as an ID number for the ROI

        Returns
        -------
        Nothing
        """
        self._id = identifier

    def set_image_name(self, image_name):
        """
        Set method for image file stub

        Parameters
        ----------
        image_name : str
            stub of image name ROI was originally extracted from

        Returns
        -------
        Nothing
        """
        self._image_name = image_name
    
    def get_id(self):
        """
        Get method for number assigned to ROI as an identifier

        Returns
        -------
        identifier : int
            Unique integer used as an ID number for the ROI
        """
        return self._id
    
    def get_image_name(self):
        """
        Get method for name of image from which ROI was extracted

        Returns
        -------
        image_name : str
            stub of image name ROI was originally extracted from
        """
        return self._image_name
    
    # def set_data(self, data):
    #     # must be overridden in derived class
    #     raise NotImplementedError
    #
    # def get_data(self):
    #     return self._data
    #
    def get_coordinates(self):
        # must be overridden in derived class
        raise NotImplementedError
    
    def get_data(self):
        raise NotImplementedError
    
    def as_array(self):
        # must be overridden in derived class
        raise NotImplementedError
    
    def as_dict(self):
        # must be overridden in derived class
        raise NotImplementedError
    
    def to_JSON(self):
        import json
        d = self.as_dict()
        return json.dumps(d)


class LineProfile(BaseROI):
    """
    This class represents a lineprofile between two points.
    This line has a start and an end point. Additionally it has a width.
    These three information are enough to calculate the profile.

    For saving line profiles to file, see LineProfileHandler.save_line_profiles, which converts LineProfile objects to
    recarrays and saves each profile as a separate table in the same hdf file.
    """
    
    def __init__(self, r1=None, c1=None, r2=None, c2=None, slice=0, width=1.0, identifier=None, image_name=None,
                 profile=None, distance=None):
        """

        Parameters
        ----------
        r1 : int
            row position of first coordinate
        c1 : int
            column position of first coordinate
        r2 : int
            row position of second coordinate
        c2 : int
            column position of first coordinate
        slice : int
            z or time slice from which line profile is selected
        width : int
            width of pixels to average over after interpolation. See skimage.measure.profile_line linewidth parameter
        identifier : int
            number assigned to line profile
        image_name : str
            stub of image filename which line profile is extracted from
        profile : 1darray
            the line profile itself. Should be initiated with None unless a LineProfile instance is being created from a
             previously saved profile
        distance : 1darray
            the x-axis (distance) of the line-profile. Should be initiated with None unless a LineProfile instance is
            being created from a previously saved profile
        """
        super(self.__class__, self).__init__(identifier=identifier, image_name=image_name)
        self._r1 = r1
        self._c1 = c1
        self._r2 = r2
        self._c2 = c2
        self._slice = slice
        self._width = width
        self.set_profile(profile)
        self.set_distance(distance)
    
    def get_start(self):
        return self._r1, self._c1
    
    def get_end(self):
        return self._r2, self._c2
    
    def set_width(self, width):
        self._width = width
    
    def set_profile(self, profile):
        self._profile = np.asarray(profile)

    def set_distance(self, distance):
        self._distance = np.asarray(distance)
    
    def get_data(self):
        return self._profile
    
    def get_coordinates(self):
        return self._distance
    
    def as_array(self):
        try:
            plen = self._profile.shape[0]
        except TypeError:
            raise RuntimeError('Handler must calculate profile before executing this function')
        
        pdtype = [('r1', '<i4'), ('c1', '<i4'), ('r2', '<i4'), ('c2', '<i4'), ('slice', '<i4'),
                  ('width', '<i4'), ('profile', 'f4', (plen)), ('distance', 'f4', (plen))]
        
        parray = np.array(
            [(self._r1, self._c1, self._r2, self._c2, self._slice, self._width, self._profile, self._distance)],
            dtype=pdtype).view(np.recarray)
        return parray
    
    def as_dict(self):
        try:
            plen = self._profile.shape[0]
        except TypeError:
            raise RuntimeError('Handler must calculate profile before executing this function')
        
        d = {
            'r1': self._r1, 'c1': self._c1, 'r2': self._r2, 'c2': self._c2, 'slice': self._slice,
            'width': self._width, 'profile': self.get_data().tolist(), 'distance': self.get_coordinates().tolist(),
            'image_name': self._image_name, 'identifier': self._id,
            'class': self.__class__.__name__
        }
        return d


class RectangularROI(BaseROI):
    def __init__(self, r1=None, c1=None, r2=None, c2=None, slice=0, identifier=None, image_name=None, data=None,
                 rows=None, columns=None):
        """

        Parameters
        ----------
        r1 : int
            row position of first coordinate
        c1 : int
            column position of first coordinate
        r2 : int
            row position of second coordinate
        c2 : int
            column position of first coordinate
        slice : int
            z or time slice from which line profile is selected
        identifier : int
            number assigned to line profile
        image_name : str
            stub of image filename which line profile is extracted from
        data : ndarray

        interpolation_factor : float

        rows : ndarray

        columns : ndarray
        """
        super(self.__class__, self).__init__(identifier=identifier, image_name=image_name)
        self._r1 = r1
        self._c1 = c1
        self._r2 = r2
        self._c2 = c2
        self._slice = slice
        self._data = np.array(data)
        self._rows = np.array(rows)
        self._columns = np.array(columns)
    
    def get_origin(self):
        return self._r1, self._c1
    
    def get_second_corner(self):
        return self._r2, self._c2
    
    def set_data(self, data):
        self._data = data
    
    def get_data(self):
        return self._data
    
    def get_rows(self):
        return self._rows
    
    def get_columns(self):
        return self._columns
    
    def get_coordinates(self):
        return (self.get_rows(), self.get_columns())
    
    def as_dict(self):
        try:
            self._data.shape[0]
        except TypeError:
            raise RuntimeError('Handler must extract data before executing this function')
        
        d = {
            'r1': self._r1, 'c1': self._c1, 'r2': self._r2, 'c2': self._c2, 'slice': self._slice,
            'data': self._data.tolist(), 'rows': self._rows.tolist(), 'columns': self._columns.tolist(),
            'image_name': self._image_name, 'identifier': self._id,
            'class': self.__class__.__name__
        }
        return d


module_lookup = {
    'LineProfile': 'PYME.DSView.modules.roiExtraction',
    'RectangularROI': 'PYME.DSView.modules.roiExtraction'
}
