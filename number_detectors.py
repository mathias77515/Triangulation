import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.stats
import pandas as pd

ntoy = 100000

coord = np.loadtxt('coord')
area_pix = np.loadtxt('area_pix')
NPIX,_ = coord.shape

nb_detector = 2                       # Enter 2, 3, or 6


################################################################################
################################ Constants #####################################
################################################################################


# Definition of the True position of GC

ra_true = (-94.4)*(np.pi/180.)
dec_true = (-28.94)*(np.pi/180.)

c = 3e8                           # Light of speed in meter per second
Rearth = 6.4e6                    # Earth radius in meter

lonKM3 = 16*(np.pi/180)
latKM3 = 0.632973
lonIC = -63.453056*(np.pi/180)
latIC = -89.99*(np.pi/180)
latSK = 36*(np.pi/180)
lonSK = 129*(np.pi/180)
latJUNO = 22.11827*(np.pi/180)
lonJUNO = 112.51867*(np.pi/180)

################################################################################
################################ Functions #####################################
################################################################################

def cartesian_coord(phi_lon, phi_lat) :
    r_x = Rearth * np.cos(phi_lon) * np.cos(phi_lat)
    r_y = Rearth * np.sin(phi_lon) * np.cos(phi_lat)
    r_z = Rearth * np.sin(phi_lat)
    return r_x, r_y, r_z

def time_delay(tab, n) :
    return np.dot(tab, n)/c                                                # Here, we identify n with de position of the source

################################################################################
############################## Main program ####################################
################################################################################

### Position of detectors

xKM3, yKM3, zKM3 = cartesian_coord(lonKM3, latKM3)
xIC, yIC, zIC = cartesian_coord(lonIC, latIC)
xSK, ySK, zSK = cartesian_coord(lonSK, latSK)
xHK, yHK, zHK = cartesian_coord(lonSK, latSK)
xJUNO, yJUNO, zJUNO = cartesian_coord(lonJUNO, latJUNO)

### Position of the source

xsource = np.cos(ra_true)*np.cos(dec_true)
ysource = np.sin(ra_true)*np.cos(dec_true)
zsource = np.sin(dec_true)

### Positions tables

posKM3 = [xKM3, yKM3, zKM3]
posIC = [xIC, yIC, zIC]
posSK = [xSK, ySK, zSK]
posHK = [xHK, yHK, zHK]
posJUNO = [xJUNO,yJUNO,zJUNO]
sourcepos = np.array([xsource, ysource, zsource])

### Difference of cartesian coordinates

posdiffICKM3 = np.array([xIC-xKM3, yIC-yKM3, zIC-zKM3])
posdiffHKKM3 = np.array([xSK-xKM3, ySK-yKM3, zSK-zKM3])
posdiffSKM3 = posdiffHKKM3
posdiffICHK = np.array([xIC-xSK, yIC-ySK, zIC-zSK])
posdiffICSK = posdiffICHK
posdiffJUNOKM3 = np.array([xJUNO-xKM3, yJUNO-yIC, zJUNO-zKM3])
posdiffJUNOIC = np.array([xJUNO-xIC, yJUNO-yIC, zJUNO-zIC])
posdiffJUNOHK = np.array([xJUNO-xSK, yJUNO-ySK, zJUNO-zSK])
posdiffJUNOSK = posdiffJUNOHK

### Time delay between two detectors (theorical)

tdelay_trueICKM3 = time_delay(posdiffICKM3, sourcepos)
tdelay_trueHKKM3 = time_delay(posdiffHKKM3, sourcepos)
tdelay_trueICHK = time_delay(posdiffICHK, sourcepos)
tdelay_trueJUNOIC = time_delay(posdiffJUNOIC, sourcepos)
tdelay_trueJUNOHK = time_delay(posdiffJUNOHK, sourcepos)
tdelay_trueJUNOKM3 = time_delay(posdiffJUNOKM3, sourcepos)


### Estimated uncertainties

sigmatKM3DUNE = 0.00158
sigmatJUNODUNE = 0.0030
sigmatHKDUNE = 0.0020
sigmatICDUNE = 0.00120
sigmatSKDUNE = 0.0053
sigmatICKM3 = 0.00665
sigmatICHK = 0.00055
sigmatHKKM3 = 0.0067
sigmatKM3SK = 0.0074
sigmatICSK = 0.00195
sigmatJUNOIC = 0.00195
sigmatJUNOHK = 0.0031
sigmatJUNOKM3 = 0.0074
sigmatSKJUNO = 0.00275

### Creation of tables according to the number of detectors

chi2_stat = np.zeros((4, NPIX))
chi2 = np.zeros((4, NPIX))
if nb_detector == 6 :
    chi2_stat = np.zeros(NPIX)

chi2_density = np.zeros(NPIX)

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


tdelay_ICKM3 = np.random.normal(tdelay_trueICKM3, sigmatICKM3, (1, ntoy))
tdelay_HKKM3 = np.random.normal(tdelay_trueHKKM3, sigmatHKKM3, (1, ntoy))
tdelay_ICHK = np.random.normal(tdelay_trueICHK, sigmatICHK, (1, ntoy))
tdelay_JUNOIC = np.random.normal(tdelay_trueJUNOIC, sigmatJUNOIC, (1, ntoy))
tdelay_JUNOHK = np.random.normal(tdelay_trueJUNOHK, sigmatJUNOHK, (1, ntoy))
tdelay_JUNOKM3 = np.random.normal(tdelay_trueJUNOKM3, sigmatJUNOKM3, (1, ntoy))

### Main loop

for itoy in range(0,ntoy):

    if itoy%1000 == 0 :
        print(itoy/1000, ' %')

    ### Computing of chi^2

    chi2_ICKM3 = ((tdelay_obsICKM3 - tdelay_ICKM3[0][itoy])/sigmatICKM3)**2
    chi2_HKKM3 = ((tdelay_obsHKKM3 - tdelay_HKKM3[0][itoy])/sigmatHKKM3)**2
    chi2_ICHK = ((tdelay_obsICHK - tdelay_ICHK[0][itoy])/sigmatICHK)**2
    chi2_JUNOIC = ((tdelay_obsJUNOIC - tdelay_JUNOIC[0][itoy])/sigmatJUNOIC)**2
    chi2_JUNOHK = ((tdelay_obsJUNOHK - tdelay_JUNOHK[0][itoy])/sigmatJUNOHK)**2
    chi2_JUNOKM3 = ((tdelay_obsJUNOKM3 - tdelay_JUNOKM3[0][itoy])/sigmatJUNOKM3)**2

    if nb_detector == 2 :

        ### Add 1 at the place of the minimum

        chi2_stat[0][np.where(chi2_ICKM3 == np.amin(chi2_ICKM3))[0]] += 1
        chi2_stat[1][np.where(chi2_HKKM3 == np.amin(chi2_HKKM3))[0]] += 1
        chi2_stat[2][np.where(chi2_ICHK == np.amin(chi2_ICHK))[0]] += 1
        chi2_stat[3][np.where(chi2_JUNOIC == np.amin(chi2_JUNOIC))[0]] += 1

    elif nb_detector == 3 :

        ### Sum the contribution

        chi2[0] = chi2_ICKM3 + chi2_JUNOIC + chi2_JUNOKM3
        chi2[1] = chi2_HKKM3 + chi2_JUNOHK + chi2_JUNOKM3
        chi2[2] = chi2_ICKM3 + chi2_HKKM3 + chi2_ICHK
        chi2[3] = chi2_ICHK + chi2_JUNOIC + chi2_JUNOHK

        ### Add 1 at the place of the minimum

        chi2_stat[0][np.where(chi2[0] == np.amin(chi2[0]))[0]] += 1
        chi2_stat[1][np.where(chi2[1] == np.amin(chi2[1]))[0]] += 1
        chi2_stat[2][np.where(chi2[2] == np.amin(chi2[2]))[0]] += 1
        chi2_stat[3][np.where(chi2[3] == np.amin(chi2[3]))[0]] += 1

    else :

        chi2 = chi2_ICKM3 + chi2_HKKM3 + chi2_ICHK + chi2_JUNOIC

        ### Add 1 at the place of the minimum

        chi2_stat[np.where(chi2 == np.amin(chi2))[0]] += 1



### Conversion to percents

chi2_stat = chi2_stat/float(ntoy)*100

for i in range(4) :
    chi2_density += chi2_stat[i]


### Sorting from largest to smallest


if nb_detector == 6 :
    data_sorted = np.sort(chi2_stat)[::-1]
else :
    data_sorted = np.zeros((4, NPIX))
    for i in range(4) :
        data_sorted[i] = np.sort(chi2_stat[i])[::-1]



### Compute the confidence area


if nb_detector == 6 :
    integral = 0
    area = 0
    for i in data_sorted:
        #print("calc:",i)
        integral += i
        area += area_pix
        if integral > 90 :
            print("90 % conf area : ", area, 'deg^2')
            break
else :
    integral = np.zeros(4)
    area = np.zeros(4)
    for j in range(4) :
        for k, i in enumerate(data_sorted) :
            #print("calc:",i)
            integral[j] += data_sorted[j, k]
            area[j] += area_pix
            if integral[j] > 90 :
                break
    print("90 % conf area : ", area[0], 'deg^2')

if nb_detector == 2 :
    np.savetxt('chi2_tot2', chi2_stat)
elif nb_detector == 3 :
    np.savetxt('chi2_tot3', chi2_stat)
else :
    np.savetxt('chi2_tot_all', chi2_stat)
