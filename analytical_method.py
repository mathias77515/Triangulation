import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.stats
import healpy as hp
from tools import *


################################################################################
############################## Main program ####################################
################################################################################

NSIDE = 128
NPIX = hp.nside2npix(NSIDE)
area_pix = hp.nside2pixarea(NSIDE, degrees = True)

coord = np.array(hp.pix2ang(nside=NSIDE, ipix=range(0,NPIX))).transpose()
coord[:,0]=np.pi/2.-coord[:,0]                                                  # Declination = 90-colatitude

ra_true = (-94.4)*(np.pi/180.)
dec_true = (-28.94)*(np.pi/180.)

sourcepos = Coordinate(ra_true, dec_true).cartesian_coords()

dict = open_pkl('constant')
c = dict['c']
Rearth = dict['Rearth']
lonKM3, latKM3 = dict['coord_KM3']
lonIC, latIC = dict['coord_IC']
lonSK, latSK = dict['coord_SK']
lonHK, latHK = dict['coord_HK']
lonJUNO, latJUNO = dict['coord_JUNO']

xKM3, yKM3, zKM3 = dict['coord_cart_KM3']
xIC, yIC, zIC = dict['coord_cart_IC']
xSK, ySK, zSK = dict['coord_cart_SK']
xHK, yHK, zHK = dict['coord_cart_HK']
xJUNO, yJUNO, zJUNO = dict['coord_cart_JUNO']

sigmatICKM3 = dict['sig_IC_KM3']
sigmatICHK = dict['sig_IC_HK']
sigmatHKKM3 = dict['sig_HK_KM3']
sigmatKM3SK = dict['sig_KM3_SK']
sigmatICSK = dict['sig_IC_SK']
sigmatJUNOIC = dict['sig_JUNO_IC']
sigmatJUNOHK = dict['sig_JUNO_HK']
sigmatJUNOKM3 = dict['sig_JUNO_KM3']
sigmatSKJUNO = dict['sig_JUNO_SK']

posdiffICKM3 = np.array([xIC-xKM3, yIC-yKM3, zIC-zKM3])
posdiffHKKM3 = np.array([xSK-xKM3, ySK-yKM3, zSK-zKM3])
posdiffSKM3 = posdiffHKKM3
posdiffICHK = np.array([xIC-xSK, yIC-ySK, zIC-zSK])
posdiffICSK = posdiffICHK
posdiffJUNOKM3 = np.array([xJUNO-xKM3, yJUNO-yIC, zJUNO-zKM3])
posdiffJUNOIC = np.array([xJUNO-xIC, yJUNO-yIC, zJUNO-zIC])
posdiffJUNOHK = np.array([xJUNO-xSK, yJUNO-ySK, zJUNO-zSK])
posdiffJUNOSK = posdiffJUNOHK

t_ij_wf = np.zeros((6, NPIX))
t_ij = np.zeros((6, NPIX))



chi2_totN = np.zeros((NPIX, NPIX))
chi2_totN_f = np.zeros(NPIX)
chi2_totN_wf = np.zeros((NPIX, NPIX))
area_analytical_N = np.zeros(NPIX)
area_analytical_N_wf = np.zeros(NPIX)

tdelay_trueICKM3 = Coordinate(ra_true, dec_true).time_delay(posdiffICKM3, sourcepos)
tdelay_trueHKKM3 = Coordinate(ra_true, dec_true).time_delay(posdiffHKKM3, sourcepos)
tdelay_trueICHK = Coordinate(ra_true, dec_true).time_delay(posdiffICHK, sourcepos)
tdelay_trueJUNOIC = Coordinate(ra_true, dec_true).time_delay(posdiffJUNOIC, sourcepos)
tdelay_trueJUNOHK = Coordinate(ra_true, dec_true).time_delay(posdiffJUNOHK, sourcepos)
tdelay_trueJUNOKM3 = Coordinate(ra_true, dec_true).time_delay(posdiffJUNOKM3, sourcepos)

### Coordinates for each pixels

findposX = np.cos(coord[:, 1])*np.cos(coord[:, 0])                                                              # Pixel coordinates
findposY = np.sin(coord[:, 1])*np.cos(coord[:, 0])
findposZ = np.sin(coord[:, 0])

### Computing of t_{i,j}

tdelay_obsICKM3 = (posdiffICKM3[0]*findposX+posdiffICKM3[1]*findposY+posdiffICKM3[2]*findposZ)/c
tdelay_obsHKKM3 = (posdiffHKKM3[0]*findposX+posdiffHKKM3[1]*findposY+posdiffHKKM3[2]*findposZ)/c
tdelay_obsICHK = (posdiffICHK[0]*findposX+posdiffICHK[1]*findposY+posdiffICHK[2]*findposZ)/c
tdelay_obsJUNOIC = (posdiffJUNOIC[0]*findposX+posdiffJUNOIC[1]*findposY+posdiffJUNOIC[2]*findposZ)/c
tdelay_obsJUNOHK = (posdiffJUNOHK[0]*findposX+posdiffJUNOHK[1]*findposY+posdiffJUNOHK[2]*findposZ)/c
tdelay_obsJUNOKM3 = (posdiffJUNOKM3[0]*findposX+posdiffJUNOKM3[1]*findposY+posdiffJUNOKM3[2]*findposZ)/c

tab_tdelay_obs = np.array([tdelay_obsICKM3, tdelay_obsHKKM3, tdelay_obsICHK, tdelay_obsJUNOIC, tdelay_obsJUNOHK, tdelay_obsJUNOKM3])
tab_tdelay_true = np.array([tdelay_trueICKM3, tdelay_trueHKKM3, tdelay_trueICHK, tdelay_trueJUNOIC, tdelay_trueJUNOHK, tdelay_trueJUNOKM3])
tab_sigmat = np.array([sigmatICKM3, sigmatHKKM3, sigmatICHK, sigmatJUNOIC, sigmatJUNOHK, sigmatJUNOKM3])

t_ij = np.zeros((1, tab_tdelay_obs.shape[0]))

sum = np.zeros(NPIX)
for i in range(t_ij.shape[1]):
    t_ij[0, i] = np.random.normal(tab_tdelay_true[i], tab_sigmat[i])
    sum += ((tab_tdelay_obs[i] - t_ij[0, i])/tab_sigmat[i])**2


### To a normal map
conf_map = Stats(NPIX).conf_area(sum)


Plots(ra_true, dec_true).plot_with_differents_confs(conf_map, name = 'Analytical method : NSIDE = {}'.format(NSIDE))
