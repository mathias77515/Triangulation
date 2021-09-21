import numpy as np
import pickle
import matplotlib.pyplot as plt
import healpy as hp
import scipy

def save_pkl(dict, name):
    pickle.dump(dict, open('{}.pkl'.format(name), 'wb'))

def open_pkl(name):
    with open('{}.pkl'.format(name), 'rb') as f:
        data = pickle.load(f)

    return data

def cartesian_coord(phi_lon, phi_lat, R) :
    r_x = R * np.cos(phi_lon) * np.cos(phi_lat)
    r_y = R * np.sin(phi_lon) * np.cos(phi_lat)
    r_z = R * np.sin(phi_lat)
    return r_x, r_y, r_z


def confarea(tab, conf, NPIX, chi2) :
    chi2_conf = np.zeros(NPIX)
    somme = 0
    for i in range(NPIX) :
        somme += tab[i]
        ind = np.where(chi2 == tab[i])[0]
        chi2_conf[ind] = 1
        if somme > conf :
            break
    return chi2_conf

def change_coord(m, coord):
    npix = m.shape[-1]
    nside = hp.npix2nside(npix)
    ang = hp.pix2ang(nside, np.arange(npix))

    rot = hp.Rotator(coord=reversed(coord))

    new_ang = rot(*ang)
    new_pix = hp.ang2pix(nside, *new_ang)

    return m[..., new_pix]


def choice_sigmat2(tab, n_detector1, n_detector2) :
    s_12 = tab[n_detector1, n_detector2]
    return s_12

def choice_sigmat3(tab, n_detector1, n_detector2, n_detector3) :
    s_12 = tab[n_detector1, n_detector2]
    s_13= tab[n_detector1, n_detector3]
    s_23 = tab[n_detector2, n_detector3]
    return s_12, s_13, s_23

def choice_sigmat4(tab, n_detector1, n_detector2, n_detector3, n_detector4) :
    s_12 = tab[n_detector1, n_detector2]
    s_13 = tab[n_detector1, n_detector3]
    s_23 = tab[n_detector2, n_detector3]
    s_24 = tab[n_detector2, n_detector4]
    return s_12, s_13, s_23, s_24



class Coordinate:

    def __init__(self, ra, dec):
        self.RA = ra
        self.DEC = dec

    def cartesian_coords(self):
        x = np.cos(self.RA)*np.cos(self.DEC)
        y = np.sin(self.RA)*np.cos(self.DEC)
        z = np.sin(self.DEC)
        return np.array([x, y, z])

    def time_delay(self, tab, n, c = 3e8) :
        return np.dot(tab, n)/c

class Stats:

    def __init__(self, npix):
        self.oneSIG = 68
        self.twoSIG = 95
        self.threeSIG = 99.7
        self.NPIX = npix

    def chi2(self, t_obs, t, sig):
        return ((t_obs - t)/sig)**2

    def conf_area(self, tab):

        delta_chi2_1 = scipy.stats.distributions.chi2.ppf(self.oneSIG/100,2)
        delta_chi2_2 = scipy.stats.distributions.chi2.ppf(self.twoSIG/100,2)
        delta_chi2_3 = scipy.stats.distributions.chi2.ppf(self.threeSIG/100,2)

        min_tab = np.min(tab)

        d_chi2_1 = min_tab + delta_chi2_1
        d_chi2_2 = min_tab + delta_chi2_2
        d_chi2_3 = min_tab + delta_chi2_3

        conf_map = np.zeros(self.NPIX)

        ind_1 = np.where(tab <= d_chi2_1)[0]
        ind_2 = np.where(tab <= d_chi2_2)[0]
        ind_3 = np.where(tab <= d_chi2_3)[0]

        conf_map[ind_3] = self.threeSIG
        conf_map[ind_2] = self.twoSIG
        conf_map[ind_1] = self.oneSIG

        return conf_map

class Plots:

    def __init__(self, ra, dec):
        self.RA = ra
        self.DEC = dec

    def plot_with_differents_confs(self, tab, name):

        ind = tab == 0
        tab[ind] = np.nan

        plt.figure()
        hp.mollview(tab, cmap = 'tab10', flip = 'geo', cbar = True, title = 'Confidence area (in %)')
        hp.visufunc.projscatter(self.RA*180/np.pi, self.DEC*180/np.pi, c = 'black', marker = 'x', lonlat=True)
        if type(name) is str:
            print('Saving...')
            plt.savefig('{}.png'.format(name))
        plt.show()
