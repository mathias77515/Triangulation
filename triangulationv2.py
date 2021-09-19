import numpy as np
import matplotlib.pyplot as plt
import random
import time
import healpy as hp
from functions import cartesian_coord, time_delay, confarea

### Number of iterations

ntoy = int(input("Enter the number of experiences : "))

################################################################################
############################### Constants ######################################
################################################################################

coord = np.loadtxt('coord')
area_pix = np.loadtxt('area_pix')
NPIX,_ = coord.shape

### Definition of the True position of GC

ra_true = (-94.4)*(np.pi/180.)
dec_true = (-28.94)*(np.pi/180.)

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

################################################################################
############################### Main loop ######################################
################################################################################

### Cartesian coordinates of detectors

xKM3, yKM3, zKM3 = cartesian_coord(lonKM3, latKM3, Rearth)
xIC, yIC, zIC = cartesian_coord(lonIC, latIC, Rearth)
xSK, ySK, zSK = cartesian_coord(lonSK, latSK, Rearth)
xHK, yHK, zHK = cartesian_coord(lonSK, latHK, Rearth)
xJUNO, yJUNO, zJUNO = cartesian_coord(lonJUNO, latJUNO, Rearth)

### Cartesian coordinates of SN

xsource = np.cos(ra_true)*np.cos(dec_true)
ysource = np.sin(ra_true)*np.cos(dec_true)
zsource = np.sin(dec_true)

### Tables of coordinates

sourcepos = np.array([xsource, ysource, zsource])

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

### Initialize the final tab

chi2_stat = np.zeros(NPIX)
chi2_f = np.zeros(NPIX)

start_time = time.time()

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

tdelay_ICKM3 = np.random.normal(tdelay_trueICKM3, sigmatICKM3, (1, ntoy))
tdelay_HKKM3 = np.random.normal(tdelay_trueHKKM3, sigmatHKKM3, (1, ntoy))
tdelay_ICHK = np.random.normal(tdelay_trueICHK, sigmatICHK, (1, ntoy))
tdelay_JUNOIC = np.random.normal(tdelay_trueJUNOIC, sigmatJUNOIC, (1, ntoy))
tdelay_JUNOHK = np.random.normal(tdelay_trueJUNOHK, sigmatJUNOHK, (1, ntoy))
tdelay_JUNOKM3 = np.random.normal(tdelay_trueJUNOKM3, sigmatJUNOKM3, (1, ntoy))

### Experience

for itoy in range(0,ntoy):

    if itoy%1000 == 0 :
        print(itoy, end = '\r')

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

    chi2_stat[np.where(chi2 == np.amin(chi2))[0]] += 1

chi2_f = chi2
interval = time.time() - start_time

print()
print("Execution time : ", np.round(interval, 2), "s")

### Conversion to percents

chi2_stat = chi2_stat/float(ntoy)*100

### Sorting from largest to smallest

data_sorted = np.sort(chi2_stat)[::-1]


################################################################################
############################# Confidence area 90 % #############################
################################################################################

'''conf = int(input("Enter the level of confidence area : "))

tab_90 = confarea(data_sorted, conf, NPIX, chi2_stat)

### Create the final table

chi2_tot_f = np.zeros(NPIX)

### Sum of contours

chi2_tot = tab_90

### Index of differents contours

ind_90 = np.where(chi2_tot == 1)[0]

### Sets the value of the contour

chi2_tot_f[ind_90] = conf



ind_90 = np.where(chi2_tot_f == conf)[0]
surface_90 = len(ind_90) * area_pix'''


plt.figure(figsize = (9, 9))
#hp.mollview(chi2_stat, norm=None, min=0, max=np.amax(chi2_stat), unit = 'Fitted values distribution, %', cmap='Blues', title=r'$\chi^2$ map', flip='geo', bgcolor = (255/255, 255/255, 255/255), sub = (1, 2, 1))
#hp.mollview(chi2_tot_f, norm=None, min=0, max=100, unit = 'Confidence, %', cmap='Blues', title= r'90 % area : ' + str(np.round( surface_90, 3)) + r' $deg^2$', flip='geo', bgcolor = (255/255, 255/255, 255/255), sub = (1, 2, 2))
hp.mollview(chi2_f, cmap = 'Blues', norm=None, min=0, max=np.amax(chi2_f), unit = '', title=r'$\chi^2$ map', flip='geo', bgcolor = (255/255, 255/255, 255/255))
hp.graticule()
hp.projscatter(np.pi/2. - dec_true, ra_true, color='black', marker = 'x', alpha = 1) #colatitude and longitude in radian
plt.annotate('Minimum', xy=(-1, -0.4), xytext=(-2, -1), arrowprops=dict(facecolor='black', shrink=0.05))
plt.show()
