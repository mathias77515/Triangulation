# This program contain all of functions

import numpy as np
import healpy as hp

def cartesian_coord(phi_lon, phi_lat, R) :
    r_x = R * np.cos(phi_lon) * np.cos(phi_lat)
    r_y = R * np.sin(phi_lon) * np.cos(phi_lat)
    r_z = R * np.sin(phi_lat)
    return r_x, r_y, r_z

c = 3e8

def time_delay(tab, n) :
    return np.dot(tab, n)/c                                                # Here, we identify n with de position of the source


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
