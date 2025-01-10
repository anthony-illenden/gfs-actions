import numpy as np
import metpy.calc as mpcalc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from siphon.catalog import TDSCatalog
import xarray as xr
from scipy.ndimage import gaussian_filter
from metpy.units import units
import time
import matplotlib.colors as mcolors

script_start = time.time()

print('---------------------------------------')
print('GFS Isentropic Surface Script - Script started.')
print('---------------------------------------')

def find_time_dim(ds, var_name):
    possible_time_dims = ['time', 'time1', 'time2', 'time3']
    time_dim = None
    for dim in possible_time_dims:
        if dim in ds[var_name].dims:
            time_dim = dim
            break
    if time_dim is None:
        raise ValueError('Could not find the time dimension')
    return time_dim
    
def find_press_dim(ds, var_name):
    possible_iso_dims = ['isobaric', 'isobaric1', 'isobaric2', 'isobaric3']
    iso_dim = None
    for dim in possible_iso_dims:
        if dim in ds[var_name].dims:
            iso_dim = dim
            break
    if iso_dim is None:
        raise ValueError('Could not find the iso dimension')
    return iso_dim

# Helper function for finding proper time variable
def find_time_var(var, time_basename='time'):
    for coord_name in var.coords:
        if coord_name.startswith(time_basename):
            return var.coords[coord_name]
    raise ValueError('No time variable found for ' + var.name)

def find_press_var(var, time_basename='isobaric'):
    for coord_name in var.coords:
        if coord_name.startswith(time_basename):
            return var.coords[coord_name]
    raise ValueError('No time variable found for ' + var.name)

tds_gfs = TDSCatalog('https://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p25deg/latest.html')
gfs_ds = tds_gfs.datasets[0]
ds = xr.open_dataset(gfs_ds.access_urls['OPENDAP'])
ds_latlon = ds.sel(lat=slice(60, 15), lon=slice(360-140, 360-50))

dims_to_keep = ['lat', 'lon', 'time', 'time1', 'time2', 'time3', 'isobaric']

ds_latlon = ds_latlon.drop_dims([dim for dim in ds_latlon.dims if dim not in dims_to_keep])

time_dim = find_time_var(ds_latlon['Temperature_isobaric'])
iso_dim = find_press_var(ds_latlon['Temperature_isobaric'])

init_time = time_dim[0].values

target_length = len(time_dim)  

# Initialize a variable to store the matching dimension name
matching_dim = None

# Loop over the dimensions and check their lengths
for dim, size in ds_latlon.dims.items():
    if size == target_length:
        matching_dim = dim
        break  # Exit loop once a match is found

for i in range(0, 29, 2):
    iteration_start = time.time()
    ds = ds_latlon.isel(**{matching_dim: i})

    isentlevs = [295] * units.kelvin

    # Extract the variables
    T = ds['Temperature_isobaric'] * units.kelvin
    U = ds['u-component_of_wind_isobaric'] * units.meters / units.seconds
    V = ds['v-component_of_wind_isobaric'] * units.meters / units.seconds
    Q = ds['Specific_humidity_isobaric'] * units.kilogram / units.kilogram
    Z = ds['Geopotential_height_isobaric'] * units.meter

    isent_data = mpcalc.isentropic_interpolation_as_dataset(isentlevs, T,  U, V, Q, Z)

    isent_data['Relative_humidity'] = mpcalc.relative_humidity_from_specific_humidity(isent_data['pressure'], isent_data['temperature'], isent_data['Specific_humidity_isobaric']).metpy.convert_units('percent')

    lons, lats = ds['lon'], ds['lat']

    smoothed_pres = gaussian_filter(isent_data['pressure'][0, :, :], sigma=3)
    smoothed_q = gaussian_filter(isent_data['Specific_humidity_isobaric'][0, :, :], sigma=1) * 1000 # units g/kg
    smoothed_rh = gaussian_filter(isent_data['Relative_humidity'][0, :, :], sigma=3) # units percent

    u_wnd = isent_data['u-component_of_wind_isobaric'][0, :, :] * 1.94384 # units knots
    v_wnd = isent_data['v-component_of_wind_isobaric'][0, :, :] * 1.94384 # units knots

    #levels = np.arange(4, 15, 1)
    #colors = ['#c3e8fa', '#8bc5e9', '#5195cf', '#49a283', '#6cc04b', '#d8de5a', '#f8b348', '#f46328', '#dc352b', '#bb1b24', '#911618']
    #cmap = mcolors.ListedColormap(colors)

    levels = np.arange(0, 101, 5)  # Levels from 0 to 100 in steps of 10
    colors = ['#c3e8fa', '#8bc5e9', '#5195cf', '#49a283', '#6cc04b', '#d8de5a', '#f8b348', '#f46328', '#dc352b', '#bb1b24', '#911618']
    cmap = mcolors.ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(12, 9), subplot_kw={'projection': ccrs.LambertConformal()})
    ax.set_extent([-125, -66.9, 23, 49.4])
    ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='gray', linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)

    clevisent = np.arange(0, 1000, 25)
    cs = ax.contour(isent_data['lon'], isent_data['lat'], smoothed_pres, clevisent, colors='k', linewidths=1.0, linestyles='solid', transform=ccrs.PlateCarree())
    try:
        plt.clabel(cs, fontsize=10, inline=1, inline_spacing=7, fmt='%i', rightside_up=True, use_clabeltext=True)
    except IndexError:
        print("No contours to label for isobars.")
    cf = ax.contourf(isent_data['lon'], isent_data['lat'], smoothed_rh, levels=levels, cmap='gist_earth_r', extend='max', transform=ccrs.PlateCarree())

    #Add a colorbar
    plt.colorbar(cf, ax=ax, orientation='horizontal', label='Relative Humidity', pad=0.05, aspect=50, ticks=np.arange(0, 101, 10))

    #Plot wind barbs on the isentropic surface
    step = 10
    ax.barbs(u_wnd['lon'][::step], u_wnd['lat'][::step], u_wnd.values[::step, ::step], v_wnd.values[::step, ::step], length=6, color='black', transform=ccrs.PlateCarree())

    isentropic_sfc = plt.Line2D([0], [0], color='black', linewidth=1, label='Isentropic Surface Pressure (hPa)')
    ax.legend(handles=[isentropic_sfc], loc='upper right')

    hour_difference = (ds_latlon[matching_dim][i] - init_time) / np.timedelta64(1, 'h')

    plt.title(f"{ds_latlon[matching_dim][0].dt.strftime('%H00 UTC').item()} GFS 295K Isentropic Surface Pressure, Relative Humidity, and Winds | {ds_latlon[matching_dim][i].dt.strftime('%Y-%m-%d %H00 UTC').item()} | FH: {hour_difference:.0f}", fontsize=12)
    plt.tight_layout()
    plt.savefig(f'plots/isentropic_295k/{hour_difference:.0f}.png', dpi=450, bbox_inches='tight')
    iteration_end = time.time()
    print(f'Iteration {i} Processing Time:', round((iteration_end - iteration_start), 2), 'seconds.')

print('\nTotal Processing Time:', round((time.time() - script_start), 2), 'seconds.')
