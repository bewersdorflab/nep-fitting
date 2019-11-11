
import numpy as np
from PYME.IO import image
from scipy import interpolate
import matplotlib.pyplot as plt
from skimage.measure import profile_line
from PYME.Acquire.Hardware.focus_locks.reflection_focus_lock import GaussFitter1D
from nep_fitting.core.handlers import LineProfileHandler
from nep_fitting.core.rois import MultiaxisProfile
import os
import multiprocessing
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def extract_multiaxis_profile(data, terp_px, src, dst, x_comp, y_comp, widths_px, pi, return_dictionary, root_dir):
    x_width_px, y_width_px, z_width_px = widths_px
    # just use pixel units
    # NOTE for scikits image, opposite xy convention is used than fiji / pyme
    xy_profile = profile_line(data.sum(axis=2).T, src, dst, linewidth=x_width_px)
    # get peak position, in pixels, from start
    peak1d = gauss1d.fit(np.arange(len(xy_profile)), xy_profile)[0][1]

    # x_peak_nm, y_peak_nm = np.array([xi[pi], yi[pi]]) + peak1d * np.array([x_comp[pi] * pixel_size[0], y_comp[pi] * pixel_size[1]])
    # plt.scatter(x_peak_nm / pixel_size[0], y_peak_nm / pixel_size[1])
    x_peak = (peak1d * x_comp) + src[0]
    y_peak = (peak1d * y_comp) + src[1]

    # extract z profile, centering it on the peak, and skewing it base on the angle
    xx_grid = np.arange(x_width_px) - 0.5 * (x_width_px - 1)
    yy_grid = np.arange(y_width_px) - 0.5 * (y_width_px - 1)
    # make grids more than 1 d
    xx_grid, yy_grid = np.meshgrid(xx_grid, yy_grid)
    # rotate and add offset
    # xx = xx_grid * x_comp[pi] + yy_grid * y_comp[pi] + x_peak
    # yy = - xx_grid * y_comp[pi] + yy_grid * x_comp[pi] + y_peak

    xx = xx_grid * x_comp - yy_grid * y_comp + x_peak
    yy = xx_grid * y_comp + yy_grid * x_comp + y_peak
    # plt.figure(0)
    # plt.scatter(xx, yy)

    z_profile = np.zeros(data.shape[2])
    # stack = np.zeros((xx.shape[0], xx.shape[1], zz.shape[0]))
    for zi in zz:
        # stack[:, :, zi] = np.reshape([terp_px((xxi, yyi, zi)) for xxi, yyi in zip(xx.ravel(), yy.ravel())], (5, 5))
        # interp2d = interpolate.interp2d((x, y, z))
        # TODO - x and y swapped here?
        z_profile[zi] = np.mean(terp_by_slice_px[zi](yy.ravel(), xx.ravel(), grid=False))
        # z_profile[zi] = np.mean([terp_px((xxi, yyi, zi)) for xxi, yyi in zip(xx.ravel(), yy.ravel())])

    # fit to find focal plane of this profile
    zpeak = int(gauss1d.fit(zz, z_profile)[0][1])
    print('profile %d peak on slice %d' % (pi, zpeak))
    mean_data = np.zeros((data.shape[0], data.shape[1]))
    # fixme - this bit is hopelessly slow.
    zsamples = zpeak + np.arange(z_width_px) - 0.5 * (z_width_px - 1)
    for indx in range(data.shape[0]):
        for indy in range(data.shape[1]):
            mean_data[indx, indy] = np.mean([terp_px((indx, indy, zsamp)) for zsamp in zsamples])

    # re-extract xy_profile
    xy_profile = profile_line(mean_data.T, src, dst, linewidth=x_width_px)


    xy_pos = pixel_size[0] * np.arange(len(xy_profile))
    z_pos = zz * pixel_size[2]
    plt.figure()
    plt.scatter(xy_pos, xy_profile)
    plt.scatter(z_pos, z_profile)
    plt.savefig(os.path.join(root_dir, 'profile%d.pdf' % pi))
    plt.clf()
    profile = MultiaxisProfile(profiles=(xy_profile, z_profile),
                               positions=(xy_pos, z_pos),
                               widths=(x_width_px * pixel_size[0], z_width_px * pixel_size[2]),
                               identifier=pi)
    return_dictionary[pi] = profile




if __name__ == '__main__':


    handler = LineProfileHandler()
    base_dir = '/home/smeagol/code/invivo-sted/2019-9-17/quantifying-3d-resolution-er-atto590/twophoton'
    root_dir = os.path.join(base_dir, '')
    filename = os.path.join(root_dir, 'twophoton-0051_Seq1_Ch1.tif')

    im = image.ImageStack(filename=filename, haveGUI=False)

    data = im.data[:,:,:,0].squeeze()

    # note that at the moment anisotropic xy pixels aren't compatible, need x and y to be the same unless we extract/
    # pass a more sophisticated xy position array to the multiaxis profile init
    pixel_size = (39.062500, 39.062500, 50.0)

    profile_width = 400  # nm
    x_width_px = round(profile_width / pixel_size[0])
    y_width_px = round(profile_width / pixel_size[1])
    z_width_px = round(profile_width / pixel_size[2])
    widths_px = (x_width_px, y_width_px, z_width_px)
    # profile_length = 2000

    roi_info = np.genfromtxt(os.path.join(root_dir, 'centerpoints-and-angles.csv'), delimiter=',')
    # profile_lengths = np.genfromtxt('/home/smeagol/code/invivo-sted/_fromMG/2019-9-3-xyz-profiles/3d-er/xy-profiles-areas-match.csv',
    #                                 delimiter=',', usemask=True)
    # profile_lengths = np.array([1e3 * prof.compressed()[-1] for prof in profile_lengths[::2]])  # [nm]
    # positions are in um, angle is in degrees where zero is aligned with positive x
    center_x, center_y, angle, profile_lengths = roi_info[1:, 5], roi_info[1:, 6], roi_info[1:, 11], roi_info[1:, -1]
    print('N: %d' % len(center_x))
    center_x *= 1e3  # um to nm
    center_y *= 1e3  # um to nm
    profile_lengths *= 1e3  # um to nm
    # xi, xf, yi, yf = endpoints_from_centers(center_x, center_y, angle, profile_length)
    rads = np.deg2rad(angle)
    x_comp = np.cos(rads)
    y_comp = -np.sin(rads)



    half_x_ext = 0.5 * profile_lengths * x_comp
    half_y_ext = 0.5 * profile_lengths * y_comp
    #
    xi, xf = center_x - half_x_ext, center_x + half_x_ext
    yi, yf = center_y - half_y_ext, center_y + half_y_ext
    # xi = start_x
    # xf = start_x + profile_lengths * x_comp
    # yi = start_y
    # yf = start_y + profile_lengths * y_comp

    xip = xi / pixel_size[0]
    xfp = xf / pixel_size[0]
    yip = yi / pixel_size[1]
    yfp = yf / pixel_size[1]

    src, dst = np.array([xip, yip]), np.array([xfp, yfp])

    # generate our interpolation
    x = np.arange(0, pixel_size[0] * data.shape[0], pixel_size[0])
    y = np.arange(0, pixel_size[1] * data.shape[1], pixel_size[1])
    z = np.arange(0, pixel_size[2] * data.shape[2], pixel_size[2])

    # terp_nm = interpolate.RegularGridInterpolator((x, y, z), data)
    terp_px = interpolate.RegularGridInterpolator((range(data.shape[0]), range(data.shape[1]), range(data.shape[2])),
                                                  data)
    zz = np.arange(data.shape[2])
    terp_by_slice_px = [interpolate.RectBivariateSpline(range(data.shape[0]), range(data.shape[1]), data[:,:,zi]) for zi in zz]

    gauss1d = GaussFitter1D()

    procs = []
    man = multiprocessing.Manager()
    return_dict = man.dict()


    for pi in range(center_x.shape[0]):

        # extract_multiaxis_profile(data, terp_px, src, dst, x_comp, y_comp, widths_px, pi, return_dictionary, root_dir)
        p = multiprocessing.Process(target=extract_multiaxis_profile, args=(data, terp_px, src[:, pi], dst[:, pi],
                                                                            x_comp[pi], y_comp[pi], widths_px, pi,
                                                                            return_dict, root_dir))
        p.start()
        procs.append(p)

    for pi, p in enumerate(procs):
        p.join()
        logger.debug('joining process %d' % pi)
        handler.add_line_profile(return_dict[pi], update=False)

    # # debug plotting
    # plt.figure(0)
    #
    #
    # for pi in range(center_x.shape[0]):
    #     plt.plot([xip[pi], xfp[pi]], [yip[pi], yfp[pi]])
    #     plt.scatter([xip[pi], xfp[pi]], [yip[pi], yfp[pi]], marker='$' + str(pi + 1) + '$')
    #
    # plt.show()

from PYME.IO.ragged import RaggedCache

rag = RaggedCache(handler.line_profiles)

rag.to_hdf(os.path.join(root_dir, 'multiaxis-profiles-%dnm.hdf' % int(profile_width)), 'line_profiles')
base_root = ''
rag.to_hdf(os.path.join(base_dir, 'all-multiaxis-profiles-%dnm.hdf' % int(profile_width)), 'line_profiles', mode='a')
