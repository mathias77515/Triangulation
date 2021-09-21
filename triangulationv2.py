import numpy as np
import matplotlib.pyplot as plt
import random
import healpy as hp
from tqdm import tqdm
from tools import *


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

NSIDE = 128
ntoy = 80000
conf = 90


NPIX = 12*NSIDE**2
coord = np.array(hp.pix2ang(nside=NSIDE, ipix=range(0,NPIX))).transpose()
coord[:,0]=np.pi/2.-coord[:,0] #declination = 90-colatitude
area_pix = hp.pixelfunc.nside2pixarea(NSIDE)

### Definition of the True position of GC

ra_true = (-94.4)*(np.pi/180.)
dec_true = (-28.94)*(np.pi/180.)

sourcepos = Coordinate(ra_true, dec_true).cartesian_coords()

### Difference of coordinates

posdiffICKM3 = np.array([xIC-xKM3, yIC-yKM3, zIC-zKM3])
posdiffHKKM3 = np.array([xHK-xKM3, yHK-yKM3, zHK-zKM3])
posdiffSKM3 = np.array([xSK-xKM3, ySK-yKM3, zSK-zKM3])
posdiffICHK = np.array([xIC-xHK, yIC-yHK, zIC-zHK])
posdiffICSK = np.array([xIC-xSK, yIC-ySK, zIC-zSK])
posdiffJUNOKM3 = np.array([xJUNO-xKM3, yJUNO-yIC, zJUNO-zKM3])
posdiffJUNOIC = np.array([xJUNO-xIC, yJUNO-yIC, zJUNO-zIC])
posdiffJUNOHK = np.array([xJUNO-xHK, yJUNO-yHK, zJUNO-zHK])
posdiffJUNOSK = np.array([xJUNO-xSK, yJUNO-ySK, zJUNO-zSK])

### Time delay between two detectors if there are not uncertainties

tdelay_trueICKM3 = Coordinate(ra_true, dec_true).time_delay(posdiffICKM3, sourcepos)
tdelay_trueHKKM3 = Coordinate(ra_true, dec_true).time_delay(posdiffHKKM3, sourcepos)
tdelay_trueICHK = Coordinate(ra_true, dec_true).time_delay(posdiffICHK, sourcepos)
tdelay_trueJUNOIC = Coordinate(ra_true, dec_true).time_delay(posdiffJUNOIC, sourcepos)
tdelay_trueJUNOHK = Coordinate(ra_true, dec_true).time_delay(posdiffJUNOHK, sourcepos)
tdelay_trueJUNOKM3 = Coordinate(ra_true, dec_true).time_delay(posdiffJUNOKM3, sourcepos)

################################################################################
############################### Main loop ######################################
################################################################################

### Initialize the final tab

chi2_stat = np.zeros(NPIX)
chi2_f = np.zeros(NPIX)

### Coordinates for each pixels

findposX = np.cos(coord[:,1])*np.cos(coord[:,0])                                                              # Pixel coordinates
findposY = np.sin(coord[:,1])*np.cos(coord[:,0])
findposZ = np.sin(coord[:,0])

### Computing of t_{i,j}

tdelay_obsICKM3 = (posdiffICKM3[0]*findposX+posdiffICKM3[1]*findposY+posdiffICKM3[2]*findposZ)/c
tdelay_obsHKKM3 = (posdiffHKKM3[0]*findposX+posdiffHKKM3[1]*findposY+posdiffHKKM3[2]*findposZ)/c
tdelay_obsICHK = (posdiffICHK[0]*findposX+posdiffICHK[1]*findposY+posdiffICHK[2]*findposZ)/c
tdelay_obsJUNOIC = (posdiffJUNOIC[0]*findposX+posdiffJUNOIC[1]*findposY+posdiffJUNOIC[2]*findposZ)/c
tdelay_obsJUNOHK = (posdiffJUNOHK[0]*findposX+posdiffJUNOHK[1]*findposY+posdiffJUNOHK[2]*findposZ)/c
tdelay_obsJUNOKM3 = (posdiffJUNOKM3[0]*findposX+posdiffJUNOKM3[1]*findposY+posdiffJUNOKM3[2]*findposZ)/c


### Random delay

tdelay_ICKM3 = np.random.normal(tdelay_trueICKM3, sigmatICKM3, (ntoy))
tdelay_HKKM3 = np.random.normal(tdelay_trueHKKM3, sigmatHKKM3, (ntoy))
tdelay_ICHK = np.random.normal(tdelay_trueICHK, sigmatICHK, (ntoy))
tdelay_JUNOIC = np.random.normal(tdelay_trueJUNOIC, sigmatJUNOIC, (ntoy))
tdelay_JUNOHK = np.random.normal(tdelay_trueJUNOHK, sigmatJUNOHK, (ntoy))
tdelay_JUNOKM3 = np.random.normal(tdelay_trueJUNOKM3, sigmatJUNOKM3, (ntoy))

tab_tdelay = np.array([tdelay_ICKM3, tdelay_HKKM3, tdelay_ICHK, tdelay_JUNOIC, tdelay_JUNOHK, tdelay_JUNOKM3])
tab_obs_tdelay = np.array([tdelay_obsICKM3, tdelay_obsHKKM3, tdelay_obsICHK, tdelay_obsJUNOIC, tdelay_obsJUNOHK, tdelay_obsJUNOKM3])
tab_sigmat = np.array([sigmatICKM3, sigmatHKKM3, sigmatICHK, sigmatJUNOIC, sigmatJUNOHK, sigmatJUNOKM3])

### Experience

for itoy in tqdm(range(0,ntoy)):

    tab_chi2 = []

    ### Computing of chi^2
    for j in range(tab_tdelay.shape[0]):

        chi2 = Stats(conf).chi2(tab_obs_tdelay[j], tab_tdelay[j, itoy], tab_sigmat[j])
        tab_chi2.append(chi2)
        ### Sum the contribution

    sum_chi2 = np.sum(tab_chi2, axis=0)

    ### Add 1 at the place of the minimum for each experiment

    chi2_stat[np.where(sum_chi2 == np.amin(sum_chi2))[0]] += 1

chi2_f = chi2

#print()
#print("Execution time : ", np.round(interval, 2), "s")

### Conversion to percents

chi2_stat = chi2_stat/float(ntoy)*100

### Sorting from largest to smallest

data_sorted = np.sort(chi2_stat)[::-1]


################################################################################
############################# Confidence area 90 % #############################
################################################################################

tab_1sig = confarea(data_sorted, 68, NPIX, chi2_stat)
tab_2sig = confarea(data_sorted, 95, NPIX, chi2_stat)
tab_3sig = confarea(data_sorted, 99.7, NPIX, chi2_stat)

### Create the final table

chi2_tot_f = np.zeros(NPIX)

### Sum of contours

chi2_tot = tab_1sig + tab_2sig + tab_3sig


### Index of differents contours

ind_1sig = np.where(chi2_tot == 3)[0]
ind_2sig = np.where(chi2_tot == 2)[0]
ind_3sig = np.where(chi2_tot == 1)[0]

### Sets the value of the contour

chi2_tot_f[ind_3sig] = 99.7
chi2_tot_f[ind_2sig] = 95
chi2_tot_f[ind_1sig] = 68




Plots(ra_true, dec_true).plot_with_differents_confs(chi2_tot_f, name = '{} - {}'.format(NSIDE, ntoy))
