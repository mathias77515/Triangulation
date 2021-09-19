import numpy as np
import matplotlib.pyplot as plt
import random
import time
import healpy as hp
from functions import cartesian_coord, time_delay, confarea, choice_sigmat2, choice_sigmat3, choice_sigmat4, change_coord

print('\n################### Parameters ###################')
### Input
info_prior = str(input("\n Do you want to include the prior ? (y/n) "))
ntoy = int(input("\nEnter the number of experiences : "))
number = int(input("\nEnter the number of detectors that you want use : "))

NSIDE = int(input("\nNSIDE (64 rather than others) : "))
NPIX = hp.nside2npix(NSIDE)
print("Pixel numbers :", NPIX)
area_pix = hp.nside2pixarea(NSIDE, degrees = True)

detector = ['KM3', 'IC', 'SK', 'JUNO', 'HK']                                    # Name of all detectors

prior = np.load('Galactic_plan_numbers_of_sources.npy')                         # Loading the prior
prior = hp.pixelfunc.ud_grade(prior, NSIDE)                                        # Reduce the number of pixel
prior = change_coord(prior, ['G', 'C'])                                         # Change the coordinate system
prior = prior / np.sum(prior)                                                   # Normalisation

################################################################################
############################### Constants ######################################
################################################################################

coord = np.array(hp.pix2ang(nside=NSIDE, ipix=range(0,NPIX))).transpose()
coord[:,0]=np.pi/2.-coord[:,0]                                                  # Declination = 90-colatitude

area_pix = hp.nside2pixarea(NSIDE, degrees = True)
NPIX,_ = coord.shape

chi2_stat2 = np.zeros(NPIX)

### Definition of the True position of GC

ra_true = (-94.4)*(np.pi/180.)
dec_true = (-28.94)*(np.pi/180.)

### Cartesian coordinates of SN

xsource = np.cos(ra_true)*np.cos(dec_true)
ysource = np.sin(ra_true)*np.cos(dec_true)
zsource = np.sin(dec_true)

sourcepos = np.array([xsource, ysource, zsource])

c = 3e8                           # Light of speed in meter per second
Rearth = 6.4e6                    # Earth radius in meter

### Positions of detectors

lonKM3 = 16.1*(np.pi/180)
latKM3 = 36.27*(np.pi/180)
lonIC = -63.45*(np.pi/180)
latIC = -89.99*(np.pi/180)
latSK = 36.23*(np.pi/180)
lonSK = 137.18*(np.pi/180)
latHK = 36.42*(np.pi/180)
lonHK = 137.3*(np.pi/180)
latJUNO = 22.13*(np.pi/180)
lonJUNO = 112.51*(np.pi/180)

### Estimated uncertainties

sigmatICKM3 = 0.00665
sigmatICHK = 0.00055
sigmatHKKM3 = 0.0067
sigmatKM3SK = 0.0074
sigmatICSK = 0.00195
sigmatJUNOIC = 0.00195
sigmatJUNOHK = 0.00199
sigmatJUNOKM3 = 0.0074
sigmatSKJUNO = 0.00275
sigmatJUNODUNE = 0.0025

sigmat = np.array([[np.nan, sigmatICKM3, sigmatKM3SK, sigmatJUNOKM3, sigmatHKKM3]
, [sigmatICKM3, np.nan, sigmatICSK, sigmatJUNOIC, sigmatICHK]
, [sigmatKM3SK, sigmatICSK, np.nan, sigmatSKJUNO, np.nan]
, [sigmatJUNOKM3, sigmatJUNOIC, sigmatSKJUNO, np.nan, sigmatJUNOHK]
, [sigmatHKKM3, sigmatICHK, np.nan, sigmatJUNOHK, np.nan]])

findposX = np.cos(coord[:,1])*np.cos(coord[:,0])                                                              # Pixel coordinates
findposY = np.sin(coord[:,1])*np.cos(coord[:,0])
findposZ = np.sin(coord[:,0])

################################################################################
############################### Main loop ######################################
################################################################################



if number == 1 :
    raise ValueError("Bad choice of number")
elif number == 2 :

    print('\n ################### Dectectors ###################')

    detector1 = str(input("\nEnter the name of the first detector : "))
    detector2 = str(input("\nEnter the name of the second detector : "))

    if detector1 == 'IC' :
        x1, y1, z1 = cartesian_coord(lonIC, latIC, Rearth)
        detector1 = 1
    elif detector1 == 'KM3' :
        x1, y1, z1 = cartesian_coord(lonKM3, latKM3, Rearth)
        detector1 = 0
    elif detector1 == 'SK' :
        x1, y1, z1 = cartesian_coord(lonSK, latSK, Rearth)
        detector1 = 2
    elif detector1 == 'HK' :
        x1, y1, z1 = cartesian_coord(lonHK, latHK, Rearth)
        detector1 = 4
    elif detector1 == 'JUNO' :
        x1, y1, z1 = cartesian_coord(lonJUNO, latJUNO, Rearth)
        detector1 = 3
    else :
        raise ValueError("Bad name of detector")

    if detector2 == 'IC' :
        x2, y2, z2 = cartesian_coord(lonIC, latIC, Rearth)
        detector2 = 1
    elif detector2 == 'KM3' :
        x2, y2, z2 = cartesian_coord(lonKM3, latKM3, Rearth)
        detector2 = 0
    elif detector2 == 'SK' :
        x2, y2, z2 = cartesian_coord(lonSK, latSK, Rearth)
        detector2 = 2
    elif detector2 == 'HK' :
        x2, y2, z2 = cartesian_coord(lonHK, latHK, Rearth)
        detector2 = 4
    elif detector2 == 'JUNO' :
        x2, y2, z2 = cartesian_coord(lonJUNO, latJUNO, Rearth)
        detector2 = 3
    else :
        raise ValueError("Bad name of detector")

    posdiff12 = np.array([x1-x2, y1-y2, z1-z2])

    tdelay_true12 = time_delay(posdiff12, sourcepos)

    sigmat_12 = choice_sigmat2(sigmat, detector1, detector2)

    tdelay_12 = np.random.normal(tdelay_true12, sigmat_12, (1, ntoy))

    tdelay_obs12 = (posdiff12[0]*findposX+posdiff12[1]*findposY+posdiff12[2]*findposZ)/c

    print('\n################### Experience ###################')
    print()

    for itoy in range(0,ntoy):

        print('Computing : ' + str(int(itoy*100/ntoy)) + ' %', end = '\r')


        ### Computing of chi^2

        chi2_12 = ((tdelay_obs12 - tdelay_12[0][itoy])/sigmat_12)**2

        ### Add 1 at the place of the minimum for each experiment

        chi2_stat2[np.where(chi2_12 == np.amin(chi2_12))[0]] += 1


elif number == 3 :

    print('\n################### Dectectors ###################')

    detector1 = str(input("\nEnter the name of the first detector : "))
    detector2 = str(input("\nEnter the name of the second detector : "))
    detector3 = str(input("\nEnter the name of the third detector : "))

    if detector1 == 'IC' :
        x1, y1, z1 = cartesian_coord(lonIC, latIC, Rearth)
        detector1 = 1
    elif detector1 == 'KM3' :
        x1, y1, z1 = cartesian_coord(lonKM3, latKM3, Rearth)
        detector1 = 0
    elif detector1 == 'SK' :
        x1, y1, z1 = cartesian_coord(lonSK, latSK, Rearth)
        detector1 = 2
    elif detector1 == 'HK' :
        x1, y1, z1 = cartesian_coord(lonHK, latHK, Rearth)
        detector1 = 4
    elif detector1 == 'JUNO' :
        x1, y1, z1 = cartesian_coord(lonJUNO, latJUNO, Rearth)
        detector1 = 3
    else :
        raise ValueError("Bad name of detector")

    if detector2 == 'IC' :
        x2, y2, z2 = cartesian_coord(lonIC, latIC, Rearth)
        detector2 = 1
    elif detector2 == 'KM3' :
        x2, y2, z2 = cartesian_coord(lonKM3, latKM3, Rearth)
        detector2 = 0
    elif detector2 == 'SK' :
        x2, y2, z2 = cartesian_coord(lonSK, latSK, Rearth)
        detector2 = 2
    elif detector2 == 'HK' :
        x2, y2, z2 = cartesian_coord(lonHK, latHK, Rearth)
        detector2 = 4
    elif detector2 == 'JUNO' :
        x2, y2, z2 = cartesian_coord(lonJUNO, latJUNO, Rearth)
        detector2 = 3
    else :
        raise ValueError("Bad name of detector")

    if detector3 == 'IC' :
        x3, y3, z3 = cartesian_coord(lonIC, latIC, Rearth)
        detector3 = 1
    elif detector3 == 'KM3' :
        x3, y3, z3 = cartesian_coord(lonKM3, latKM3, Rearth)
        detector3 = 0
    elif detector3 == 'SK' :
        x3, y3, z3 = cartesian_coord(lonSK, latSK, Rearth)
        detector3 = 2
    elif detector3 == 'HK' :
        x3, y3, z3 = cartesian_coord(lonHK, latHK, Rearth)
        detector3 = 4
    elif detector3 == 'JUNO' :
        x3, y3, z3 = cartesian_coord(lonJUNO, latJUNO, Rearth)
        detector3 = 3
    else :
        raise ValueError("Bad name of detector")

    posdiff13 = np.array([x1-x3, y1-y3, z1-z3])
    posdiff23 = np.array([x2-x3, y2-y3, z2-z3])
    posdiff12 = np.array([x1-x2, y1-y2, z1-z2])

    tdelay_true13 = time_delay(posdiff13, sourcepos)
    tdelay_true23 = time_delay(posdiff23, sourcepos)
    tdelay_true12 = time_delay(posdiff12, sourcepos)

    sigmat_12, sigmat_13, sigmat_23 = choice_sigmat3(sigmat, detector1, detector2, detector3)

    ### Random delay

    tdelay_13 = np.random.normal(tdelay_true13, sigmat_13, (1, ntoy))
    tdelay_23 = np.random.normal(tdelay_true23, sigmat_23, (1, ntoy))
    tdelay_12 = np.random.normal(tdelay_true12, sigmat_12, (1, ntoy))

    tdelay_obs13 = (posdiff13[0]*findposX+posdiff13[1]*findposY+posdiff13[2]*findposZ)/c
    tdelay_obs23 = (posdiff23[0]*findposX+posdiff23[1]*findposY+posdiff23[2]*findposZ)/c
    tdelay_obs12 = (posdiff12[0]*findposX+posdiff12[1]*findposY+posdiff12[2]*findposZ)/c

    print('\n ################### Experience ###################')
    print()

    for itoy in range(0,ntoy):

        print('Computing : ' + str(int(itoy*100/ntoy)) + ' %', end = '\r')


        ### Computing of chi^2

        chi2_13 = ((tdelay_obs13 - tdelay_13[0][itoy])/sigmat_13)**2
        chi2_23 = ((tdelay_obs23 - tdelay_23[0][itoy])/sigmat_23)**2
        chi2_12 = ((tdelay_obs12 - tdelay_12[0][itoy])/sigmat_12)**2

        ### Sum the contribution

        chi2 = chi2_13 + chi2_23 + chi2_12

        ### Add 1 at the place of the minimum for each experiment

        chi2_stat2[np.where(chi2 == np.amin(chi2))[0]] += 1



elif number == 4 :

    print('\n ################### Dectectors ###################')

    detector1 = str(input("\nEnter the name of the first detector : "))
    detector2 = str(input("\nEnter the name of the second detector : "))
    detector3 = str(input("\nEnter the name of the third detector : "))
    detector4 = str(input("\nEnter the name of the fourth detector : "))

    if detector1 == 'IC' :
        x1, y1, z1 = cartesian_coord(lonIC, latIC, Rearth)
        detector1 = 1
    elif detector1 == 'KM3' :
        x1, y1, z1 = cartesian_coord(lonKM3, latKM3, Rearth)
        detector1 = 0
    elif detector1 == 'SK' :
        x1, y1, z1 = cartesian_coord(lonSK, latSK, Rearth)
        detector1 = 2
    elif detector1 == 'HK' :
        x1, y1, z1 = cartesian_coord(lonHK, latHK, Rearth)
        detector1 = 4
    elif detector1 == 'JUNO' :
        x1, y1, z1 = cartesian_coord(lonJUNO, latJUNO, Rearth)
        detector1 = 3
    else :
        raise ValueError("Bad name of detector")

    if detector2 == 'IC' :
        x2, y2, z2 = cartesian_coord(lonIC, latIC, Rearth)
        detector2 = 1
    elif detector2 == 'KM3' :
        x2, y2, z2 = cartesian_coord(lonKM3, latKM3, Rearth)
        detector2 = 0
    elif detector2 == 'SK' :
        x2, y2, z2 = cartesian_coord(lonSK, latSK, Rearth)
        detector2 = 2
    elif detector2 == 'HK' :
        x2, y2, z2 = cartesian_coord(lonHK, latHK, Rearth)
        detector2 = 4
    elif detector2 == 'JUNO' :
        x2, y2, z2 = cartesian_coord(lonJUNO, latJUNO, Rearth)
        detector2 = 3
    else :
        raise ValueError("Bad name of detector")

    if detector3 == 'IC' :
        x3, y3, z3 = cartesian_coord(lonIC, latIC, Rearth)
        detector3 = 1
    elif detector3 == 'KM3' :
        x3, y3, z3 = cartesian_coord(lonKM3, latKM3, Rearth)
        detector3 = 0
    elif detector3 == 'SK' :
        x3, y3, z3 = cartesian_coord(lonSK, latSK, Rearth)
        detector3 = 2
    elif detector3 == 'HK' :
        x3, y3, z3 = cartesian_coord(lonHK, latHK, Rearth)
        detector3 = 4
    elif detector3 == 'JUNO' :
        x3, y3, z3 = cartesian_coord(lonJUNO, latJUNO, Rearth)
        detector3 = 3
    else :
        raise ValueError("Bad name of detector")

    if detector4 == 'IC' :
        x4, y4, z4 = cartesian_coord(lonIC, latIC, Rearth)
        detector4 = 1
    elif detector4 == 'KM3' :
        x4, y4, z4 = cartesian_coord(lonKM3, latKM3, Rearth)
        detector4 = 0
    elif detector4 == 'SK' :
        x4, y4, z4 = cartesian_coord(lonSK, latSK, Rearth)
        detector4 = 2
    elif detector4 == 'HK' :
        x4, y4, z4 = cartesian_coord(lonHK, latHK, Rearth)
        detector4 = 4
    elif detector4 == 'JUNO' :
        x4, y4, z4 = cartesian_coord(lonJUNO, latJUNO, Rearth)
        detector4 = 3
    else :
        raise ValueError("Bad name of detector")

    posdiff13 = np.array([x1-x3, y1-y3, z1-z3])
    posdiff23 = np.array([x2-x3, y2-y3, z2-z3])
    posdiff12 = np.array([x1-x2, y1-y2, z1-z2])
    posdiff24 = np.array([x2-x4, y2-y4, z2-z4])

    tdelay_true13 = time_delay(posdiff13, sourcepos)
    tdelay_true23 = time_delay(posdiff23, sourcepos)
    tdelay_true12 = time_delay(posdiff12, sourcepos)
    tdelay_true24 = time_delay(posdiff24, sourcepos)

    sigmat_12, sigmat_13, sigmat_23, sigmat_24 = choice_sigmat4(sigmat, detector1, detector2, detector3, detector4)

    ### Random delay

    tdelay_13 = np.random.normal(tdelay_true13, sigmat_13, (1, ntoy))
    tdelay_23 = np.random.normal(tdelay_true23, sigmat_23, (1, ntoy))
    tdelay_12 = np.random.normal(tdelay_true12, sigmat_12, (1, ntoy))
    tdelay_24 = np.random.normal(tdelay_true24, sigmat_24, (1, ntoy))

    ### Computing of t_{i,j}

    tdelay_obs13 = (posdiff13[0]*findposX+posdiff13[1]*findposY+posdiff13[2]*findposZ)/c
    tdelay_obs23 = (posdiff23[0]*findposX+posdiff23[1]*findposY+posdiff23[2]*findposZ)/c
    tdelay_obs12 = (posdiff12[0]*findposX+posdiff12[1]*findposY+posdiff12[2]*findposZ)/c
    tdelay_obs24 = (posdiff24[0]*findposX+posdiff24[1]*findposY+posdiff24[2]*findposZ)/c

    print('\n################### Experience ###################')
    print()

    for itoy in range(0,ntoy):

        print('Computing : ' + str(int(itoy*100/ntoy)) + ' %', end = '\r')


        ### Computing of chi^2

        chi2_13 = ((tdelay_obs13 - tdelay_13[0][itoy])/sigmat_13)**2
        chi2_23 = ((tdelay_obs23 - tdelay_23[0][itoy])/sigmat_23)**2
        chi2_12 = ((tdelay_obs12 - tdelay_12[0][itoy])/sigmat_12)**2
        chi2_24 = ((tdelay_obs24 - tdelay_24[0][itoy])/sigmat_24)**2

        ### Sum the contribution

        chi2 = chi2_13 + chi2_23 + chi2_12 + chi2_24

        ### Add 1 at the place of the minimum for each experiment

        chi2_stat2[np.where(chi2 == np.amin(chi2))[0]] += 1

elif number == 5 :
    xKM3, yKM3, zKM3 = cartesian_coord(lonKM3, latKM3, Rearth)
    xIC, yIC, zIC = cartesian_coord(lonIC, latIC, Rearth)
    xSK, ySK, zSK = cartesian_coord(lonSK, latSK, Rearth)
    xHK, yHK, zHK = cartesian_coord(lonSK, latHK, Rearth)
    xJUNO, yJUNO, zJUNO = cartesian_coord(lonJUNO, latJUNO, Rearth)

    posdiffICKM3 = np.array([xIC-xKM3, yIC-yKM3, zIC-zKM3])
    posdiffHKKM3 = np.array([xHK-xKM3, yHK-yKM3, zHK-zKM3])
    posdiffSKM3 = np.array([xSK-xKM3, ySK-yKM3, zSK-zKM3])
    posdiffICHK = np.array([xIC-xHK, yIC-yHK, zIC-zHK])
    posdiffICSK = np.array([xIC-xSK, yIC-ySK, zIC-zSK])
    posdiffJUNOKM3 = np.array([xJUNO-xKM3, yJUNO-yIC, zJUNO-zKM3])
    posdiffJUNOIC = np.array([xJUNO-xIC, yJUNO-yIC, zJUNO-zIC])
    posdiffJUNOHK = np.array([xJUNO-xHK, yJUNO-yHK, zJUNO-zHK])
    posdiffJUNOSK = np.array([xJUNO-xSK, yJUNO-ySK, zJUNO-zSK])

    tdelay_trueICKM3 = time_delay(posdiffICKM3, sourcepos)
    tdelay_trueHKKM3 = time_delay(posdiffHKKM3, sourcepos)
    tdelay_trueICHK = time_delay(posdiffICHK, sourcepos)
    tdelay_trueJUNOIC = time_delay(posdiffJUNOIC, sourcepos)
    tdelay_trueJUNOHK = time_delay(posdiffJUNOHK, sourcepos)
    tdelay_trueJUNOKM3 = time_delay(posdiffJUNOKM3, sourcepos)

    tdelay_obsICKM3 = (posdiffICKM3[0]*findposX+posdiffICKM3[1]*findposY+posdiffICKM3[2]*findposZ)/c
    tdelay_obsHKKM3 = (posdiffHKKM3[0]*findposX+posdiffHKKM3[1]*findposY+posdiffHKKM3[2]*findposZ)/c
    tdelay_obsICHK = (posdiffICHK[0]*findposX+posdiffICHK[1]*findposY+posdiffICHK[2]*findposZ)/c
    tdelay_obsJUNOIC = (posdiffJUNOIC[0]*findposX+posdiffJUNOIC[1]*findposY+posdiffJUNOIC[2]*findposZ)/c
    tdelay_obsJUNOHK = (posdiffJUNOHK[0]*findposX+posdiffJUNOHK[1]*findposY+posdiffJUNOHK[2]*findposZ)/c
    tdelay_obsJUNOKM3 = (posdiffJUNOKM3[0]*findposX+posdiffJUNOKM3[1]*findposY+posdiffJUNOKM3[2]*findposZ)/c

    ### Random delay

    tdelay_ICKM3 = np.random.normal(tdelay_trueICKM3, sigmatICKM3, (1, ntoy))
    tdelay_HKKM3 = np.random.normal(tdelay_trueHKKM3, sigmatHKKM3, (1, ntoy))
    tdelay_ICHK = np.random.normal(tdelay_trueICHK, sigmatICHK, (1, ntoy))
    tdelay_JUNOIC = np.random.normal(tdelay_trueJUNOIC, sigmatJUNOIC, (1, ntoy))
    tdelay_JUNOHK = np.random.normal(tdelay_trueJUNOHK, sigmatJUNOHK, (1, ntoy))
    tdelay_JUNOKM3 = np.random.normal(tdelay_trueJUNOKM3, sigmatJUNOKM3, (1, ntoy))

    chi2_ICKM3 = np.zeros((NPIX, NPIX))
    chi2_HKKM3 = np.zeros((NPIX, NPIX))
    chi2_ICHK = np.zeros((NPIX, NPIX))
    chi2_JUNOIC = np.zeros((NPIX, NPIX))
    chi2_JUNOHK = np.zeros((NPIX, NPIX))
    chi2_JUNOKM3 = np.zeros((NPIX, NPIX))

    print('\n ################### Experience ###################')
    print()

    for itoy in range(0,ntoy):


        print('Computing : ' + str(int(itoy*100/ntoy)) + ' %', end = '\r')


        ### Computing of chi^2

        chi2_ICKM3 = ((tdelay_obsICKM3 - tdelay_ICKM3[0][itoy])/sigmatICKM3)**2
        chi2_HKKM3 = ((tdelay_obsHKKM3 - tdelay_HKKM3[0][itoy])/sigmatHKKM3)**2
        chi2_ICHK = ((tdelay_obsICHK - tdelay_ICHK[0][itoy])/sigmatICHK)**2
        chi2_JUNOIC = ((tdelay_obsJUNOIC - tdelay_JUNOIC[0][itoy])/sigmatJUNOIC)**2
        chi2_JUNOHK = ((tdelay_obsJUNOHK - tdelay_JUNOHK[0][itoy])/sigmatJUNOHK)**2
        chi2_JUNOKM3 = ((tdelay_obsJUNOKM3 - tdelay_JUNOKM3[0][itoy])/sigmatJUNOKM3)**2

        ### Sum the contribution

        chi2 = chi2_ICKM3 + chi2_HKKM3 + chi2_ICHK + chi2_JUNOIC + chi2_JUNOHK + chi2_JUNOKM3

        ### Add 1 at the place of the minimum for each experiment

        chi2_stat2[np.where(chi2 == np.amin(chi2))[0]] += 1


############################# Without prior ################################

### Conversion to percents

ind_0 = np.where(chi2_stat2 != 0)[0]

chi2_stat2 = chi2_stat2/float(ntoy)*100

### Sorting from largest to smallest

data_sorted = np.sort(chi2_stat2)[::-1]

print()
print('\n ################### Confidence area ###################')
print()

conf = int(input("\nEnter the level of confidence area : "))

tab_90 = confarea(data_sorted, conf, NPIX, chi2_stat2)

### Create the final table

chi2_tot_f = np.zeros(NPIX)

### Sum of contours

chi2_tot = tab_90

### Index of differents contours

ind_90 = np.where(chi2_tot == 1)[0]

### Sets the value of the contour

chi2_tot_f[ind_90] = conf

ind_90 = np.where(chi2_tot_f == conf)[0]
surface_90 = len(ind_90) * area_pix

############################### With prior #################################

if info_prior == 'y' :

    ### Conversion to percents

    chi2_stat2_wp = (chi2_stat2/100 * prior) / np.sum((chi2_stat2/100) * prior)

    ### Sorting from largest to smallest

    data_sorted_wp = np.sort(chi2_stat2_wp)[::-1]


    conf_wp = conf/100

    tab_90_wp = confarea(data_sorted_wp, conf_wp, NPIX, chi2_stat2_wp)

    ### Create the final table

    chi2_tot_f_wp = np.zeros(NPIX)

    ### Sum of contours

    chi2_tot_wp = tab_90_wp

    ### Index of differents contours

    ind_90_wp = np.where(chi2_tot_wp == 1)[0]

    ### Sets the value of the contour

    chi2_tot_f_wp[ind_90_wp] = conf

    surface_90_wp = len(ind_90_wp) * area_pix

else :
    pass


print()
print('\n ################### Plot ###################')
print()
################################################################################
#################################### Plot ######################################
################################################################################

if number == 2 :
    if info_prior == 'y' :
        plt.figure(figsize = (10, 10))
        hp.mollview(chi2_stat2, cmap = 'Blues', flip = 'geo', title = str(detector[detector1]) + ' - ' + str(detector[detector2]), sub = (1, 3, 1))
        hp.mollview(chi2_tot_f, norm=None, min=0, max=100, unit = 'Confidence, %', cmap='Blues', title= r'90 % area : ' + str(np.round( surface_90, 3)) + r' $deg^2$', flip='geo', sub = (1, 3, 2))
        hp.mollview(chi2_tot_f_wp, norm=None, min=0, max=100, unit = 'Confidence, %', cmap='Blues', title= r'90 % area : ' + str(np.round( surface_90_wp, 3)) + r' $deg^2$', flip='geo', sub = (1, 3, 3))
        hp.graticule()
        plt.show()
    else :
        plt.figure(figsize = (10, 10))
        hp.mollview(prior, cmap = 'gnuplot2', flip = 'geo', title = 'Prior', sub = (1, 3, 1))
        hp.mollview(chi2_stat2, cmap = 'Blues', flip = 'geo', title = str(detector[detector1]) + ' - ' + str(detector[detector2]), sub = (1, 3, 2))
        hp.mollview(chi2_tot_f, norm=None, min=0, max=100, unit = 'Confidence, %', cmap='Blues', title= r'90 % area : ' + str(np.round( surface_90, 3)) + r' $deg^2$', flip='geo', sub = (1, 3, 3))
        hp.graticule()
        plt.show()

elif number == 3 :
    if info_prior == 'y' :
        plt.figure(figsize = (10, 10))
        hp.mollview(chi2_stat2, cmap = 'Blues', flip = 'geo', title = str(detector[detector1]) + ' - ' + str(detector[detector2]) + ' - ' + str(detector[detector3]), sub = (1, 3, 1))
        hp.mollview(chi2_tot_f, norm=None, min=0, max=100, unit = 'Confidence, %', cmap='Blues', title= r'90 % area : ' + str(np.round( surface_90, 3)) + r' $deg^2$', flip='geo', sub = (1, 3, 2))
        hp.mollview(chi2_tot_f_wp, norm=None, min=0, max=100, unit = 'Confidence, %', cmap='Blues', title= r'90 % area : ' + str(np.round( surface_90_wp, 3)) + r' $deg^2$', flip='geo', sub = (1, 3, 3))
        hp.graticule()
        plt.show()
    else :
        plt.figure(figsize = (10, 10))
        hp.mollview(prior, cmap = 'gnuplot2', flip = 'geo', title = 'Prior', sub = (1, 3, 1))
        hp.mollview(chi2_stat2, cmap = 'Blues', flip = 'geo', title = str(detector[detector1]) + ' - ' + str(detector[detector2]) + ' - ' + str(detector[detector3]), sub = (1, 3, 2))
        hp.mollview(chi2_tot_f, norm=None, min=0, max=100, unit = 'Confidence, %', cmap='Blues', title= r'90 % area : ' + str(np.round( surface_90, 3)) + r' $deg^2$', flip='geo', sub = (1, 3, 3))
        hp.graticule()
        plt.show()

elif number == 4 :
    if info_prior == 'y' :
        plt.figure(figsize = (10, 10))
        hp.mollview(chi2_stat2, cmap = 'Blues', flip = 'geo', title = str(detector[detector1]) + ' - ' + str(detector[detector2]) + ' - ' + str(detector[detector3]) + ' - ' + str(detector[detector4]), sub = (1, 3, 1))
        hp.mollview(chi2_tot_f, norm=None, min=0, max=100, unit = 'Confiance, %', cmap='Blues', title= r'90 % area : ' + str(np.round( surface_90, 1)) + r' $deg^2$', flip='geo', sub = (1, 3, 2))
        hp.mollview(chi2_tot_f_wp, norm=None, min=0, max=100, unit = 'Confiance, %', cmap='Blues', title= r'90 % area : ' + str(np.round( surface_90_wp, 1)) + r' $deg^2$', flip='geo', sub = (1, 3, 3))
        hp.graticule()
        plt.show()
    else :
        plt.figure(figsize = (10, 10))
        hp.mollview(prior, cmap = 'gnuplot2', flip = 'geo', title = 'Prior', sub = (1, 3, 1))
        hp.mollview(chi2_stat2, cmap = 'Blues', flip = 'geo', title = str(detector[detector1]) + ' - ' + str(detector[detector2]) + ' - ' + str(detector[detector3]) + ' - ' + str(detector[detector4]), sub = (1, 3, 2))
        hp.mollview(chi2_tot_f, norm=None, min=0, max=100, unit = 'Confidence, %', cmap='Blues', title= r'90 % area : ' + str(np.round( surface_90, 3)) + r' $deg^2$', flip='geo', sub = (1, 3, 3))
        hp.graticule()
        plt.show()

else :
    if info_prior == 'y' :
        plt.figure(figsize = (10, 10))
        hp.mollview(chi2_stat2, cmap = 'CMRmap', flip = 'geo', title = 'All of detectors', unit = 'in %', sub = (1, 3, 1))
        hp.mollview(chi2_tot_f, norm=None, min=0, max=100, unit = '', cmap='Blues', title= 'Result without the prior', flip='geo', sub = (1, 3, 2))
        hp.mollview(chi2_tot_f_wp, norm=None, min=0, max=100, unit = '', cmap='Blues', title= 'Result with the prior', flip='geo', sub = (1, 3, 3))
        hp.graticule()
        plt.show()
    else :
        plt.figure(figsize = (10, 10))
        hp.mollview(prior, cmap = 'gnuplot2', flip = 'geo', title = 'Prior', sub = (1, 3, 1))
        hp.mollview(chi2_stat2, cmap = 'CMRmap', flip = 'geo', title = 'All of detectors', sub = (1, 3, 2))
        hp.mollview(chi2_tot_f, norm=None, min=0, max=100, unit = 'Confidence, %', cmap='Blues', title= r'90 % area : ' + str(np.round( surface_90, 3)) + r' $deg^2$', flip='geo', sub = (1, 3, 3))
        hp.graticule()
        plt.show()

plt.figure(figsize = (10, 10))
hp.mollview(np.round(np.log10(chi2),2), flip = 'geo', cmap = 'CMRmap', title = r'Carte de $\chi^2$', unit = r'$\log(\chi^2)$')
hp.graticule()
plt.show()
