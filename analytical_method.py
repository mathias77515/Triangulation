import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.stats
import pandas as pd
import healpy as hp
from functions import cartesian_coord, time_delay, confarea, change_coord


prior = np.load('Galactic_plan_numbers_of_sources.npy')                         # Loading the prior
prior = hp.pixelfunc.ud_grade(prior, 32)                                        # Reduce the number of pixel
prior = change_coord(prior, ['G', 'C'])                                         # Change the coordinate system
prior = prior / np.sum(prior)                                                   # Normalisation


################################################################################
############################## Main program ####################################
################################################################################

NSIDE = int(input("\nNSIDE (64 rather than others) : "))
NPIX = hp.nside2npix(NSIDE)
print("Pixel numbers :", NPIX)
area_pix = hp.nside2pixarea(NSIDE, degrees = True)

coord = np.array(hp.pix2ang(nside=NSIDE, ipix=range(0,NPIX))).transpose()
coord[:,0]=np.pi/2.-coord[:,0]                                                  # Declination = 90-colatitude

ra_true = (-94.4)*(np.pi/180.)
dec_true = (-28.94)*(np.pi/180.)

xsource = np.cos(ra_true)*np.cos(dec_true)
ysource = np.sin(ra_true)*np.cos(dec_true)
zsource = np.sin(dec_true)
sourcepos = np.array([xsource, ysource, zsource])

c = 3e8                           # Light of speed in meter per second
Rearth = 6.4e6                    # Earth radius in meter


#estimated uncertainties

sigmatICKM3 = 0.00665
sigmatICHK = 0.00055
sigmatHKKM3 = 0.0067
sigmatKM3SK = 0.0074
sigmatICSK = 0.00195
sigmatJUNOIC = 0.00195
sigmatJUNOHK = 0.00199
sigmatJUNOKM3 = 0.0074
sigmatSKJUNO = 0.00275

lonKM3 = 16*(np.pi/180)
latKM3 = 0.632973
lonIC = -63.453056*(np.pi/180)
latIC = -89.99*(np.pi/180)
latSK = 36*(np.pi/180)
lonSK = 129*(np.pi/180)
latJUNO = 22.11827*(np.pi/180)
lonJUNO = 112.51867*(np.pi/180)


xKM3, yKM3, zKM3 = cartesian_coord(lonKM3, latKM3, Rearth)
xIC, yIC, zIC = cartesian_coord(lonIC, latIC, Rearth)
xSK, ySK, zSK = cartesian_coord(lonSK, latSK, Rearth)
xHK, yHK, zHK = cartesian_coord(lonSK, latSK, Rearth)
xJUNO, yJUNO, zJUNO = cartesian_coord(lonJUNO, latJUNO, Rearth)

xsource = np.cos(ra_true)*np.cos(dec_true)
ysource = np.sin(ra_true)*np.cos(dec_true)
zsource = np.sin(dec_true)


posKM3 = [xKM3, yKM3, zKM3]
posIC = [xIC, yIC, zIC]
posSK = [xSK, ySK, zSK]
posHK = [xHK, yHK, zHK]
posJUNO = [xJUNO,yJUNO,zJUNO]
sourcepos = np.array([xsource, ysource, zsource])

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

tdelay_trueICKM3 = time_delay(posdiffICKM3, sourcepos)
tdelay_trueHKKM3 = time_delay(posdiffHKKM3, sourcepos)
tdelay_trueICHK = time_delay(posdiffICHK, sourcepos)
tdelay_trueJUNOIC = time_delay(posdiffJUNOIC, sourcepos)
tdelay_trueJUNOHK = time_delay(posdiffJUNOHK, sourcepos)
tdelay_trueJUNOKM3 = time_delay(posdiffJUNOKM3, sourcepos)

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



t_ij = np.zeros(6)

t_ij[0] = np.random.normal(tdelay_trueICKM3, sigmatICKM3)
t_ij[1] = np.random.normal(tdelay_trueHKKM3, sigmatHKKM3)
t_ij[2] = np.random.normal(tdelay_trueICHK, sigmatICHK)
t_ij[3] = np.random.normal(tdelay_trueJUNOIC, sigmatJUNOIC)
t_ij[4] = np.random.normal(tdelay_trueJUNOHK, sigmatJUNOHK)
t_ij[5] = np.random.normal(tdelay_trueJUNOKM3, sigmatJUNOKM3)




chi2_ICKM3 = ((tdelay_obsICKM3 - t_ij[0])/sigmatICKM3)**2
chi2_HKKM3 = ((tdelay_obsHKKM3 - t_ij[1])/sigmatHKKM3)**2
chi2_ICHK = ((tdelay_obsICHK - t_ij[2])/sigmatICHK)**2
chi2_JUNOIC = ((tdelay_obsJUNOIC - t_ij[3])/sigmatJUNOIC)**2
chi2_JUNOHK = ((tdelay_obsJUNOHK - t_ij[4])/sigmatJUNOHK)**2
chi2_JUNOKM3 = ((tdelay_obsJUNOKM3 - t_ij[5])/sigmatJUNOKM3)**2

### Sum the contribution

chi2_totN = chi2_ICKM3 + chi2_HKKM3 + chi2_ICHK + chi2_JUNOIC + chi2_JUNOHK + chi2_JUNOKM3

chi2_map = chi2_totN.copy()

### To a normal map

delta_chi2 = scipy.stats.distributions.chi2.ppf(0.9,2)                   # For 90% -> ~4.60

min_chi2 = np.zeros(NPIX)

min_chi2 = np.min(chi2_totN)                        # Minimum

d_chi2 = min_chi2 + delta_chi2

area_ana = np.zeros(NPIX)

conf_map = np.zeros(NPIX)

area_ana = np.sum(chi2_totN <= d_chi2) * area_pix               # Sum over all pixel which verify the condition
ind_90 = np.where(chi2_totN <= d_chi2)[0]


conf_map[ind_90] = 90


plt.figure(figsize = (8, 8))
hp.mollview(conf_map, title = '', flip = 'geo', cmap = 'Blues')
hp.projscatter(np.pi/2. - dec_true, ra_true)
hp.graticule()
plt.show()
