# Program to write a dictionary

import numpy as np
import pickle
from tools import *


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

### Cartesian coordinates of detectors

xKM3, yKM3, zKM3 = cartesian_coord(lonKM3, latKM3, Rearth)
xIC, yIC, zIC = cartesian_coord(lonIC, latIC, Rearth)
xSK, ySK, zSK = cartesian_coord(lonSK, latSK, Rearth)
xHK, yHK, zHK = cartesian_coord(lonSK, latHK, Rearth)
xJUNO, yJUNO, zJUNO = cartesian_coord(lonJUNO, latJUNO, Rearth)



dict = {'c' : c,
        'Rearth' : Rearth,
        'coord_KM3' : (lonKM3, latKM3),
        'coord_IC' : (lonIC, latIC),
        'coord_SK' : (lonSK, latSK),
        'coord_HK' : (lonHK, latHK),
        'coord_JUNO' : (lonJUNO, latJUNO),
        'sig_IC_KM3' : sigmatICKM3,
        'sig_IC_HK' : sigmatICHK,
        'sig_HK_KM3' : sigmatHKKM3,
        'sig_KM3_SK' : sigmatKM3SK,
        'sig_IC_SK' : sigmatICSK,
        'sig_JUNO_IC' : sigmatJUNOIC,
        'sig_JUNO_HK' : sigmatJUNOHK,
        'sig_JUNO_SK' : sigmatSKJUNO,
        'sig_JUNO_KM3' : sigmatJUNOKM3,
        'coord_cart_KM3' : (xKM3, yKM3, zKM3),
        'coord_cart_IC' : (xIC, yIC, zIC),
        'coord_cart_SK' : (xSK, ySK, zSK),
        'coord_cart_HK' : (xHK, yHK, zHK),
        'coord_cart_JUNO' : (xJUNO, yJUNO, zJUNO)
}
