import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.stats
import pandas as pd
import healpy as hp
from functions import cartesian_coord, time_delay, confarea, change_coord

print('\n################## Parameters ###################')
print()

NSIDE = int(input("Enter the NSIDE (32 max) : "))
NPIX = hp.nside2npix(NSIDE)

### Donwloading of the prior

prior = np.load('Galactic_plan_numbers_of_sources.npy')
prior = change_coord(prior, ['G', 'C'])                                         # Transform the galactic coordinates in celestial coordinates
prior = hp.pixelfunc.ud_grade(prior, NSIDE)                                     # Reduce the resolution
prior = prior / np.sum(prior)                                                   # Normalization


################################################################################
############################## Main program ####################################
################################################################################

print()
print("We have two methods :")
print()
print("     - Monte-Carlo method (enter MC)")
print("     - Analytical method (enter A)")
print()
method = str(input("\n Enter the name of the method : "))


coord = np.array(hp.pix2ang(nside=NSIDE, ipix=range(0,NPIX))).transpose()
coord[:,0]=np.pi/2. - coord[:,0]

dec_true = coord[:, 0]
ra_true = coord[:, 1]

area_pix = hp.nside2pixarea(NSIDE, degrees = True)

c = 3e8                                                                         # Light of speed in meter per second
Rearth = 6.4e6                                                                  # Earth radius in meter

### Coordinates of the detectors (longitude/latitude)

lonKM3 = 16*(np.pi/180)
latKM3 = 0.632973
lonIC = -63.453056*(np.pi/180)
latIC = -89.99*(np.pi/180)
latSK = 36*(np.pi/180)
lonSK = 129*(np.pi/180)
latJUNO = 22.11827*(np.pi/180)
lonJUNO = 112.51867*(np.pi/180)

### Computing of the cartesian coordinates

xKM3, yKM3, zKM3 = cartesian_coord(lonKM3, latKM3, Rearth)
xIC, yIC, zIC = cartesian_coord(lonIC, latIC, Rearth)
xSK, ySK, zSK = cartesian_coord(lonSK, latSK, Rearth)
xHK, yHK, zHK = cartesian_coord(lonSK, latSK, Rearth)
xJUNO, yJUNO, zJUNO = cartesian_coord(lonJUNO, latJUNO, Rearth)

xsource = np.cos(ra_true)*np.cos(dec_true)
ysource = np.sin(ra_true)*np.cos(dec_true)
zsource = np.sin(dec_true)

### Tables of positions

posKM3 = [xKM3, yKM3, zKM3]
posIC = [xIC, yIC, zIC]
posSK = [xSK, ySK, zSK]
posHK = [xHK, yHK, zHK]
posJUNO = [xJUNO,yJUNO,zJUNO]
sourcepos = np.array([xsource, ysource, zsource])

### Differences between two detectors

posdiffICKM3 = np.array([xIC-xKM3, yIC-yKM3, zIC-zKM3])
posdiffHKKM3 = np.array([xSK-xKM3, ySK-yKM3, zSK-zKM3])
posdiffSKM3 = posdiffHKKM3
posdiffICHK = np.array([xIC-xSK, yIC-ySK, zIC-zSK])
posdiffICSK = posdiffICHK
posdiffJUNOKM3 = np.array([xJUNO-xKM3, yJUNO-yIC, zJUNO-zKM3])
posdiffJUNOIC = np.array([xJUNO-xIC, yJUNO-yIC, zJUNO-zIC])
posdiffJUNOHK = np.array([xJUNO-xSK, yJUNO-ySK, zJUNO-zSK])
posdiffJUNOSK = posdiffJUNOHK

### Computing of the time delay

tdelay_trueICKM3 = time_delay(posdiffICKM3, sourcepos)
tdelay_trueHKKM3 = time_delay(posdiffHKKM3, sourcepos)
tdelay_trueICHK = time_delay(posdiffICHK, sourcepos)
tdelay_trueJUNOIC = time_delay(posdiffJUNOIC, sourcepos)
tdelay_trueJUNOHK = time_delay(posdiffJUNOHK, sourcepos)
tdelay_trueJUNOKM3 = time_delay(posdiffJUNOKM3, sourcepos)

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

### Each coordinates of the sources

dec_true = coord[:, 0]
ra_true = coord[:, 1]

xsource = np.cos(ra_true)*np.cos(dec_true)
ysource = np.sin(ra_true)*np.cos(dec_true)
zsource = np.sin(dec_true)
sourcepos = np.array([xsource, ysource, zsource])

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


### Computing of the observed time

tdelay_obsICKM3 = (posdiffICKM3[0]*findposX+posdiffICKM3[1]*findposY+posdiffICKM3[2]*findposZ)/c
tdelay_obsHKKM3 = (posdiffHKKM3[0]*findposX+posdiffHKKM3[1]*findposY+posdiffHKKM3[2]*findposZ)/c
tdelay_obsICHK = (posdiffICHK[0]*findposX+posdiffICHK[1]*findposY+posdiffICHK[2]*findposZ)/c
tdelay_obsJUNOIC = (posdiffJUNOIC[0]*findposX+posdiffJUNOIC[1]*findposY+posdiffJUNOIC[2]*findposZ)/c
tdelay_obsJUNOHK = (posdiffJUNOHK[0]*findposX+posdiffJUNOHK[1]*findposY+posdiffJUNOHK[2]*findposZ)/c
tdelay_obsJUNOKM3 = (posdiffJUNOKM3[0]*findposX+posdiffJUNOKM3[1]*findposY+posdiffJUNOKM3[2]*findposZ)/c

if method == 'MC' :

    ntoy = int(input("\nEnter the number of experience per pixels : "))

    tdelay_ICKM3 = np.zeros((ntoy, NPIX))
    tdelay_HKKM3 = np.zeros((ntoy, NPIX))
    tdelay_ICHK = np.zeros((ntoy, NPIX))
    tdelay_JUNOIC = np.zeros((ntoy, NPIX))
    tdelay_JUNOHK = np.zeros((ntoy, NPIX))
    tdelay_JUNOKM3 = np.zeros((ntoy, NPIX))

    ### Random number as time delay

    for i in range(ntoy) :
        tdelay_ICKM3[i] = np.random.normal(tdelay_trueICKM3, sigmatICKM3)
        tdelay_HKKM3[i] = np.random.normal(tdelay_trueHKKM3, sigmatHKKM3)
        tdelay_ICHK[i] = np.random.normal(tdelay_trueICHK, sigmatICHK)
        tdelay_JUNOIC[i] = np.random.normal(tdelay_trueJUNOIC, sigmatJUNOIC)
        tdelay_JUNOHK[i] = np.random.normal(tdelay_trueJUNOHK, sigmatJUNOHK)
        tdelay_JUNOKM3[i] = np.random.normal(tdelay_trueJUNOKM3, sigmatJUNOKM3)


    print()
    print('################## Computing ###################')
    print()

    chi2_tot1 = np.zeros((NPIX, NPIX))

    ### Experience

    for i in range(NPIX) :

        print('Computing : ', str(int(i*100/NPIX)) + ' %', end = '\r')

        for j in range(ntoy) :


            ### Computing of chi^2

            chi2_ICKM3 = ((tdelay_obsICKM3 - tdelay_ICKM3[j, i])/sigmatICKM3)**2
            chi2_HKKM3 = ((tdelay_obsHKKM3 - tdelay_HKKM3[j, i])/sigmatHKKM3)**2
            chi2_ICHK = ((tdelay_obsICHK - tdelay_ICHK[j, i])/sigmatICHK)**2
            chi2_JUNOIC = ((tdelay_obsJUNOIC - tdelay_JUNOIC[j, i])/sigmatJUNOIC)**2
            chi2_JUNOHK = ((tdelay_obsJUNOHK - tdelay_JUNOHK[j, i])/sigmatJUNOHK)**2
            chi2_JUNOKM3 = ((tdelay_obsJUNOKM3 - tdelay_JUNOKM3[j, i])/sigmatJUNOKM3)**2

            ### Sum the contribution

            chi2 = chi2_ICKM3 + chi2_HKKM3 + chi2_ICHK + chi2_JUNOIC + chi2_JUNOHK + chi2_JUNOKM3

            chi2_tot1[i][np.where(chi2 == np.amin(chi2))[0]] += 1

    print()

    data_sorted = np.zeros((NPIX, NPIX))
    data_sortedbis = np.zeros((NPIX, NPIX))
    chi2_tot1bis = np.zeros((NPIX, NPIX))
    prob = np.zeros((NPIX, NPIX))
    integral = np.zeros(NPIX)
    integralbis = np.zeros(NPIX)
    area_mc = np.zeros(NPIX)
    area_mcbis = np.zeros(NPIX)
    tab = chi2_tot1.copy()

    chi2_tot1 = tab/float(ntoy)*100

    print()
    print()

    ### Sorting the tables (without prior)

    for i in range(NPIX) :
        print('Sorting (without prior) : ', str(int(i*100/NPIX)) + ' %', end = '\r')
        data_sorted[i] = np.sort(chi2_tot1[i])[::-1]

    print()

    ### Computing the confidence area (without prior)

    for j in range(NPIX) :
        print('Computing of confidence area (without prior) : ', str(int(j*100/NPIX)) + ' %', end = '\r')

        for k, i in enumerate(data_sorted[j]) :
            integral[j] += i
            area_mc[j] += np.round(area_pix, 1)
            if integral[j] > 90 :
                #print("90% conf area : ", area, 'deg^2')
                break

    print()
    print()

    chi2_tot1bis = tab/float(ntoy)

    ### Sorting the tables (with prior)

    for i in range(NPIX) :
        print('Sorting (with prior) : ', str(int(i*100/NPIX)) + ' %', end = '\r')
        prob[i] = (chi2_tot1bis[i] * prior)
        prob[i] = prob[i]/np.sum(chi2_tot1bis[i] * prior)
        data_sortedbis[i] = np.sort(prob[i])[::-1]

    print()

    ### Computing the confidence area (with prior)

    for j in range(NPIX) :
        print('Computing of confidence area (with prior) : ', str(int(j*100/NPIX)) + ' %', end = '\r')

        for k, i in enumerate(data_sortedbis[j]) :
            #print("calc:",i)
            integralbis[j] += i
            area_mcbis[j] += np.round(area_pix, 1)
            if integralbis[j] > 0.90 :
                #print("90% conf area : ", area, 'deg^2')
                break

    print()
    print('\n################## Plot ###################')
    print()

    ### Plots

    plt.figure(figsize = (10, 10))
    #hp.mollview(prior, flip = 'geo', cmap = 'gnuplot2', sub = (2, 2, 1), title = 'Prior (Number of sources in each pixels)')
    hp.mollview(area_mc, flip = 'geo', cmap = 'gnuplot2', sub = (1, 2, 1), unit = r"Value of the area (in $deg^2$)", title = r"Confidence area distribution")
    hp.mollview(area_mcbis, flip = 'geo', cmap = 'gnuplot2', sub = (1, 2, 2), unit = r"Value of the area (in $deg^2$)", title = "Confidence area distribution (with prior)")
    #hp.mollview((area_mcbis - area_mc)*100/area_mcbis, flip = 'geo', cmap = 'bwr', sub = (2, 2, 4), title = "Relative fluctuations")
    plt.show()

    plt.figure(figsize = (10, 10))
    hp.mollview(np.round((area_mcbis - area_mc)*100/area_mcbis, 1), flip = 'geo', unit = r'Deacrease/Increase (in $\%$)', cmap = 'bwr', title = "Relative fluctuations")
    plt.show()

if method == 'A' :

    chi2_totN = np.zeros((NPIX, NPIX))
    chi2_totN_f = np.zeros(NPIX)
    chi2_totN_wf = np.zeros((NPIX, NPIX))
    area_analytical_N = np.zeros(NPIX)
    area_analytical_N_wf = np.zeros(NPIX)

    print()
    print('################## Computing ###################')
    print()

    t_ij = np.zeros((6, NPIX))

    t_ij[0] = tdelay_trueICKM3
    t_ij[1] = tdelay_trueHKKM3
    t_ij[2] = tdelay_trueICHK
    t_ij[3] = tdelay_trueJUNOIC
    t_ij[4] = tdelay_trueJUNOHK
    t_ij[5] = tdelay_trueJUNOKM3


    for i in range(NPIX) :

        print('Computing : ', str(int(i*100/NPIX)) + ' %', end = '\r')

        ### Computing of chi^2

        chi2_ICKM3 = ((tdelay_obsICKM3 - t_ij[0, i])/sigmatICKM3)**2
        chi2_HKKM3 = ((tdelay_obsHKKM3 - t_ij[1, i])/sigmatHKKM3)**2
        chi2_ICHK = ((tdelay_obsICHK - t_ij[2, i])/sigmatICHK)**2
        chi2_JUNOIC = ((tdelay_obsJUNOIC - t_ij[3, i])/sigmatJUNOIC)**2
        chi2_JUNOHK = ((tdelay_obsJUNOHK - t_ij[4, i])/sigmatJUNOHK)**2
        chi2_JUNOKM3 = ((tdelay_obsJUNOKM3 - t_ij[5, i])/sigmatJUNOKM3)**2

        ### Sum the contribution

        chi2_totN[i] = chi2_ICKM3 + chi2_HKKM3 + chi2_ICHK + chi2_JUNOIC + chi2_JUNOHK + chi2_JUNOKM3


    chi2_map = chi2_totN.copy()

    ### To a normal map

    delta_chi2 = scipy.stats.distributions.chi2.ppf(0.9,2)                  # For 90% -> ~4.60

    min_chi2 = np.zeros(NPIX)

    print()
    print("Done")

    min_chi2 = np.min(chi2_totN, axis = 1)                                  # Minimum

    print()
    print("Done")

    d_chi2 = min_chi2 + delta_chi2

    area_ana = np.zeros(NPIX)

    print()
    print("Done")

    for i in range(NPIX) :
        area_ana[i] = np.sum(chi2_totN[i] <= d_chi2[i]) * area_pix               # Sum over all pixel which verify the condition

    print()
    print("Done")

    area_analytical_N += area_ana

    ### Plots

    plt.figure(figsize = (10, 10))
    hp.mollview(np.round(area_analytical_N, 1), flip = 'geo', cmap = 'gnuplot2', unit = r"Valeur de l'aire à $90 \%$", title = r"Distribution de l'aire de confiance à $90 \%$")
    plt.show()
    
