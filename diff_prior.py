# This program show us the different prior that we can have.

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

def change_coord(m, coord):
    npix = m.shape[-1]
    nside = hp.npix2nside(npix)
    ang = hp.pix2ang(nside, np.arange(npix))

    rot = hp.Rotator(coord=reversed(coord))

    new_ang = rot(*ang)
    new_pix = hp.ang2pix(nside, *new_ang)

    return m[..., new_pix]

def normalize(x) :
    return x/np.sum(x)


#prior_original = np.loadtxt('old_prior')
#prior_new = np.loadtxt('prior_new')
prior_with_dust = np.load('prior_with_dust.npy')
#prior_with_dust = change_coord(prior_with_dust, ['G', 'C'])
#prior_with_dust = hp.pixelfunc.ud_grade(prior_with_dust, 64)
#prior_galactic_sources_32 = normalize(prior_galactic_sources_32)
#prior_galactic_sources = normalize(prior_galactic_sources)
prior_without_dust = np.load('prior_without_dust.npy')
#prior_without_dust = np.load('prior_density.npy')
#prior_without_dust = change_coord(prior_without_dust, ['G', 'C'])
#prior_density = normalize(prior_density)
#prior_without_dust = hp.pixelfunc.ud_grade(prior_without_dust, 64)
#prior_without_dust = normalize(prior_without_dust)



plt.figure(figsize = (5, 5))
hp.mollview(prior_without_dust, cmap = 'gnuplot2', flip = 'geo', title = "Densité d'étoile", unit = "Nombre d'étoile par pixels")
#hp.mollview(prior_galactic_sources, cmap = 'gnuplot2', flip = 'geo', sub = (1, 2, 1), title = "Star density in the milky way (NSIDE = 512)", unit = 'Number of stars / pixels')
#hp.mollview(prior_galactic_sources_32, cmap = 'gnuplot2', flip = 'geo', sub = (1, 2, 2), title = "Star density in the milky way (NSIDE = 64)", unit = 'Number of stars / pixels')
hp.graticule()
plt.show()
