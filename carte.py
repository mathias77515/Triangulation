# Charger les modules cartopy avec la commande : conda install -c conda-forge cartopy


import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs


def main():
    ax = plt.axes(projection=ccrs.Mollweide(central_longitude=0))

    # make the map global rather than have it zoom in to
    # the extents of any plotted data
    ax.set_global()

    ax.stock_img()
    ax.coastlines()

    ax.plot(16, (0.632973/np.pi)*180, marker = '.', color = 'red', markersize = 15, transform=ccrs.PlateCarree())
    ax.plot(-63.453056, -90, marker = '.', color = 'red', markersize = 15, transform=ccrs.PlateCarree())
    ax.plot(137.2999, 36.4123, marker = '.', color = 'red', markersize = 15, transform=ccrs.PlateCarree())
    ax.plot(112.51, 22.13, marker = '.', color = 'red', markersize = 15, transform=ccrs.PlateCarree())
    #ax.plot(-88.25, 41.83, marker = '.', color = 'red', markersize = 15, transform=ccrs.PlateCarree())
    ax.text(16 - 3, (0.632973/np.pi)*180 - 12, 'KM3NeT',horizontalalignment='center', fontsize=8, c = 'red', fontstretch = 'ultra-expanded', transform=ccrs.PlateCarree())
    #ax.text(-88.25 - 15, 41.83 + 10, 'DUNE',horizontalalignment='center', fontsize=8, c = 'red', fontstretch = 'ultra-expanded', transform=ccrs.PlateCarree())
    ax.text(-63.453056 + 30, -90 + 20, 'IC', horizontalalignment='center', fontsize=8, c = 'red', fontstretch = 'ultra-expanded', transform=ccrs.PlateCarree())
    ax.text(137.2999 - 35, 36.4123 - 10, 'JUNO',horizontalalignment='center', fontsize=8, c = 'red', fontstretch = 'ultra-expanded', transform=ccrs.PlateCarree())
    ax.text(112.51 + 35, 22.13 + 10, 'SK',horizontalalignment='center', fontsize=8, c = 'red', fontstretch = 'ultra-expanded', transform=ccrs.PlateCarree())

    plt.show(block = False)


if __name__ == '__main__':
    main()
