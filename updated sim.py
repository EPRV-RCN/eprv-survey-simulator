# imports
import csv
import calendar
import random
import re
import os
import warnings

import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime, timedelta

from astropy import units as u
from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord, EarthLocation
from astroplan import Observer, FixedTarget
from scipy.interpolate import InterpolatedUnivariateSpline, RegularGridInterpolator

from astroquery.simbad import Simbad

from stellar_gp.argo_model import GPModel, GPData
from stellar_gp.argo_model import GranulationKernel,OscillationKernel,QPKernel,PerKernel,M52Kernel,M32Kernel,WNKernel, SEKernel,M52PDKernel
from stellar_gp.argo_model import covariance_matrix,generate_ts, get_stellar_kernels
from stellar_gp.stellar_scalings import get_stellar_hypers, calc_Pg





# USER INPUT / CONFIGURATION 

# optionally provide 'teff', 'vmag', 'vsini', and 'dec'.
# If a parameter is not provided, the code will try to find it via SIMBAD, then fall back to the HWO CSV file.
user_stars_input = [
    {'name': 'HD 166'},
    # {'name': 'HD ', 'teff': }, 
    # {'name': 'HD ', 'vmag':, 'vsini':},
]

# File paths 
hwo_file = r"C:/Users/shire/Downloads/DI_STARS_EXEP_2025.06.20_11.46.50.csv"
weather_file = r"C:\Users\shire\Downloads\KPNO.txt"
output_timestamps_file = r"C:\Users\shire\Downloads\JUST TIMESTAMPS.txt"
path_to_grid = r"C:\Users\shire\Downloads\neid etc"

# SCHEDULING OPTIONS (Only one of these should be set)
cadence_days = None # None = use the even_spread_flag
even_spread_flag = True # If True and cadence_days is None = spread observations evenly




# Function to look up star data using astroquery (SIMBAD)
def query_star_data_astroquery(star_name):
    """
    Queries Simbad for star data (RA, Dec, vsini, Vmag). Teff is not available in SIMBAD.
    Returns a dict with keys: hd_id, dec, vsini, vmag (maybe None if missing).
    """
    Simbad.add_votable_fields('rot', 'flux(V)')
    Simbad.TIMEOUT = 60
    data = {'hd_id': None, 'dec': None, 'vsini': None, 'vmag': None}

    try:
        print(f"Attempting to query data for {star_name} via astroquery...")
        result_table = Simbad.query_object(star_name)

        name_match = re.search(r'HD\s*(\d+[A-Za-z]*)', star_name, re.IGNORECASE)
        data['hd_id'] = name_match.group(1).strip() if name_match else star_name.replace(' ', '_')

        if result_table is None or len(result_table) == 0:
            print(f"Query failed: No result found for {star_name}.")
            return data

        row = result_table[0]

        # DEC
        if 'DEC' in row.colnames and row['DEC'] is not np.ma.masked:
            data['dec'] = float(row['DEC'])

        # Vmag
        if 'FLUX_V' in row.colnames and row['FLUX_V'] is not np.ma.masked:
            data['vmag'] = float(row['FLUX_V'])

        # vsini
        if 'ROT_Vsini' in row.colnames and row['ROT_Vsini'] is not np.ma.masked:
            vs = float(row['ROT_Vsini'])
            # NEID condition: minimum of 1.0 km/s
            data['vsini'] = vs if vs >= 1.0 else 1.0

        print(f"Successfully retrieved from SIMBAD: hd_id={data['hd_id']}, dec={data['dec']}, vsini={data['vsini']}, vmag={data['vmag']}")
        return data

    except Exception as e:
        print(f"An error occurred during astroquery for {star_name}: {e}")
        return data

def load_hwo_candidates(filepath):
    """
    Loads star data from the HWO candidates CSV file including vsini, declination, and Vmag.
    Returns a tuple containing:
                           - hd_set (set): A set of HD star identifiers.
                           - teff_dict (dict): Dictionary mapping HD to effective temperature (Teff) values.
                           - vsini_dict (dict): Dictionary mapping HD to vsini values.
                           - dec_dict (dict): Dictionary mapping HD to declination values.
                           - vmag_dict (dict): Dictionary mapping HD to Vmag values from HWO.
    """
    hd_set, teff_dict, vsini_dict, dec_dict, vmag_dict = set(), {}, {}, {}, {}
    expected_min_cols = 60
    row_counter = 0

    try:
        with open(filepath, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            header = next(reader, None)
            if header:
                row_counter += 1

            for row_num_in_file, row in enumerate(reader, start=2):
                row_counter += 1

                if len(row) < expected_min_cols:
                    continue # Skip this row

                raw_hd = row[4].strip()      # Column E
                dec_str = row[11].strip()    # Column L
                hwo_vmag_str = row[15].strip() # Column P
                teff_str = row[25].strip()   # Column Z
                vsini_str = row[59].strip()  # Column BH

                hd_identifier = ""
                if raw_hd.upper().startswith("HD "):
                    hd_identifier = raw_hd[3:].strip()
                else:
                    continue

                if not hd_identifier:
                    continue

                hd_set.add(hd_identifier)

                try:
                    teff_dict[hd_identifier] = float(teff_str)
                except ValueError:
                    teff_dict[hd_identifier] = None
                try:
                    vmag_val = float(hwo_vmag_str)
                    vmag_dict[hd_identifier] = round(vmag_val, 2)
                except ValueError:
                    vmag_dict[hd_identifier] = None

                try:
                    if vsini_str and vsini_str != "0":
                        vsini_val = float(vsini_str)
                        if vsini_val < 1.0:
                            vsini_val = 1.0 #neid condition
                        vsini_dict[hd_identifier] = round(vsini_val, 1)
                    else:
                        vsini_dict[hd_identifier] = None
                except ValueError:
                    vsini_dict[hd_identifier] = None

                try:
                    dec_dict[hd_identifier] = float(dec_str)
                except ValueError:
                    dec_dict[hd_identifier] = None

    except IOError:
        print(f"Error: HWO candidates file not found at {filepath}")
        return set(), {}, {}, {}, {}
    except Exception as e:
        print(f"An unexpected error occurred loading HWO candidates data: {e}. Last processed row (approx): {row_counter}")
        return hd_set, teff_dict, vsini_dict, dec_dict, vmag_dict

    print(f"HWO Fallback data loaded: {len(hd_set)} stars found.")
    return hd_set, teff_dict, vsini_dict, dec_dict, vmag_dict

def get_star_profile(star_input, hwo_fallback_data):
    """
    Buildnig a complete star profile using a tiered data retrieval system:
    1. User-provided data
    2. SIMBAD query
    3. HWO CSV fallback
    Returns a dictionary with the complete profile or None if essential data is missing.
    """
    hd_set, hwo_teff, hwo_vsini, hwo_dec, hwo_vmag = hwo_fallback_data
    star_name = star_input.get('name')
    if not star_name:
        print("Star input is missing a 'name'. Skipping.")
        return None

    # profile with user-provided data
    profile = {
        'name': star_name,
        'hd_id': None,
        'teff': star_input.get('teff'),
        'dec': star_input.get('dec'),
        'vsini': star_input.get('vsini'),
        'vmag': star_input.get('vmag')
    }

    #Fill in missing data from SIMBAD
    if any(val is None for val in profile.values()):
        simbad_data = query_star_data_astroquery(star_name)
        if simbad_data:
            if profile['hd_id'] is None and simbad_data.get('hd_id'):
                profile['hd_id'] = simbad_data['hd_id']
            if profile['dec'] is None and simbad_data.get('dec'):
                profile['dec'] = simbad_data['dec']
            if profile['vsini'] is None and simbad_data.get('vsini'):
                profile['vsini'] = simbad_data['vsini']
            if profile['vmag'] is None and simbad_data.get('vmag'):
                profile['vmag'] = simbad_data['vmag']

    # making sure  we have an HD ID for HWO lookup
    if profile['hd_id'] is None:
        name_match = re.search(r'HD\s*(\d+[A-Za-z]*)', star_name, re.IGNORECASE)
        if name_match:
            profile['hd_id'] = name_match.group(1).strip()
        else:
            print(f"Could not determine HD identifier for '{star_name}' from input or SIMBAD. Cannot use HWO fallback.")

    # remaining missing data from HWO fallback
    if profile['hd_id'] and profile['hd_id'] in hd_set:
        hd_id = profile['hd_id']
        if profile['teff'] is None:
            profile['teff'] = hwo_teff.get(hd_id)
        if profile['dec'] is None:
            profile['dec'] = hwo_dec.get(hd_id)
        if profile['vsini'] is None:
            profile['vsini'] = hwo_vsini.get(hd_id)
        if profile['vmag'] is None:
            profile['vmag'] = hwo_vmag.get(hd_id)

    # Final check
    if profile['vsini'] is not None and profile['vsini'] < 1.0:
        profile['vsini'] = 1.0 # neid condition

    # Validate for completeness
    required_keys = ['hd_id', 'teff', 'dec', 'vsini', 'vmag']
    missing_keys = [key for key in required_keys if profile.get(key) is None]

    if missing_keys:
        print(f"Incomplete profile for {star_name} (HD {profile['hd_id']}). Missing: {', '.join(missing_keys)}")
        return None

    print(f"Successfully built profile for {star_name} (HD {profile['hd_id']}).")
    return profile



#scheduling functions

def observable_window(
    star_name=None,
    ra=None,
    dec=None,
    observatory="Kitt Peak",
    airmass_limit=1.5,
    min_hours=2.0,
    year=2025,
    plot=False
):
    """
    Identify days of the year when a target is above 'airmass_limit',
    for at least 'min_hours' hours, during astronomical night, 
    from a specified observatory.
    
    Parameters
    ----------
    star_name : str, optional
        Name of the target star (resolved with astropy name resolver).
    ra, dec : float, optional
        Right Ascension and Declination in degrees (used if star_name not given).
    observatory : str
        Observatory name (e.g., "Kitt Peak", "Keck").
    airmass_limit : float
        Maximum airmass (e.g., 1.5).
    min_hours : float
        Minimum required hours per night.
    year : int
        Year to compute for.
    plot : bool
        Generates plot showing number of hours observable as a function of day of the year
    
    Returns
    -------
    observable_days : int or None
        array of days of the year when conditions are met
    """
    # --- Setup observer ---
    location = EarthLocation.of_site(observatory)
    observer = Observer(location=location, name=observatory)
    
    # --- Setup target ---
    if star_name is not None:
        target = FixedTarget.from_name(star_name)
    elif ra is not None and dec is not None:
        target = FixedTarget(coord=SkyCoord(ra*u.deg, dec*u.deg), name="CustomTarget")
    else:
        raise ValueError("Please provide either star_name or (ra, dec).")
    
    # --- Altitude cutoff ---
    altitude_limit = np.degrees(np.arcsin(1/airmass_limit))
    
    days = np.arange(1, 367)  # day of year
    observable_hours = []

    for day in days:
        # Reference midnight UTC
        time_midnight = Time(f"{year}-01-01 00:00:00") + (day-1)*u.day
        midnight = observer.midnight(time_midnight, which="nearest")

        # Twilight boundaries
        try:
            evening_twilight = observer.twilight_evening_astronomical(midnight, which="nearest")
            morning_twilight = observer.twilight_morning_astronomical(midnight, which="next")
        except:
            observable_hours.append(0.0)
            continue

        # Grid of times across the night
        n_points = 200
        delta_t = np.linspace(0, (morning_twilight - evening_twilight).to(u.hour).value, n_points)*u.hour
        times = evening_twilight + delta_t

        # Altitudes
        altitudes = observer.altaz(times, target).alt.deg

        # Count observable time
        mask = altitudes >= altitude_limit
        hours = np.sum(mask) * (delta_t[1]-delta_t[0]).to(u.hour).value
        observable_hours.append(hours)

    observable_hours = np.array(observable_hours)
    
    if plot==True:
        # --- Optional Plot ---
        plt.figure(figsize=(10,6))
        plt.plot(days, observable_hours, color="darkblue")
        plt.fill_between(days, 0, observable_hours, alpha=0.3, color="skyblue")
        plt.xlabel("Day of Year")
        plt.ylabel("Observable Hours After Sundown")
        plt.title(f"Observable Hours of {target.name} from KPNO in {year}")
        plt.grid(alpha=0.4)
        plt.show()

    # Days where conditions are met
    good_days = days[observable_hours >= min_hours]
    if len(good_days) == 0:
        return None
    else:
        return good_days


def generate_observation_timestamps(
    stars,
    dec_dict,
    n_observations_per_star,
    start_day=0,
    end_day=365,
    cadence_days=None,
    even_spread=False,
    seed=42
):
    """
    Generate observation timestamps over a given period based on cadence or even spread,
    respecting the star's observable window.

    Parameters:
        stars (list): List of star names/identifiers.
        dec_dict (dict): Dictionary mapping star HD IDs to their declination.
        n_observations_per_star (dict): Dict mapping each star to number of observations.
        start_day (int): Start of the survey (in day-of-year).
        end_day (int): End of the survey (in day-of-year).
        cadence_days (int or None): Fixed cadence in days between observations.
        even_spread (bool): If True, observations are evenly spaced over the duration.
        seed (int): Random seed for reproducibility. 

    Returns:
        Dict mapping star names to lists of observation days.
    """
    np.random.seed(seed)
    random.seed(seed)

    observation_timestamps = {}

    for star in stars:
        n_obs = n_observations_per_star.get(star, 0)
        if n_obs <= 0:
            continue

        # Determine the observable days for the star -
        star_dec = dec_dict.get(star)
        observable_days_of_year = None
        if star_dec is not None:
            try:
                observable_days_of_year = observable_window(
                    star_name=f"HD {star}",
                    observatory="Kitt Peak",
                    airmass_limit=1.5,
                    min_hours=2.0,
                    year=2025, # Match the main script's year
                    plot=False
                )
            except Exception as e:
                print(f"Could not calculate observable window for HD {star}: {e}. Defaulting to full year.")
                observable_days_of_year = None

        if observable_days_of_year is None or len(observable_days_of_year) == 0:
            print(f"Warning: No observable window found for HD {star}. Using full year for scheduling.")
            observable_days_of_year = np.arange(start_day + 1, end_day + 1)

        # Schedule observations within the observable window 
        if cadence_days is not None:
            # Use fixed cadence, but only select days within the observable window
            obs_days_raw = list(range(start_day, end_day, cadence_days))
            obs_days = [day for day in obs_days_raw if day in observable_days_of_year]
            obs_days = obs_days[:n_obs] if len(obs_days) >= n_obs else obs_days
            
        elif even_spread:
            # Use even spread across the *observable* days
            if n_obs > 1:
                indices = np.round(np.linspace(0, len(observable_days_of_year) - 1, num=n_obs)).astype(int)
                obs_days = list(observable_days_of_year[indices])
            elif n_obs == 1:
                obs_days = [observable_days_of_year[len(observable_days_of_year) // 2]]
            else: # n_obs == 0
                obs_days = []
        else:
            # Fallback to random choice from the *observable* days
            obs_days = sorted(np.random.choice(observable_days_of_year, size=n_obs, replace=False))

        observation_timestamps[star] = obs_days

    return observation_timestamps


def plot_observations(observation_timestamps):
    """
    Plots a raster of observation timestamps.

    Parameters:
        observation_timestamps (dict): Mapping of star -> list of day-of-year timestamps.
    """
    sorted_stars = sorted(observation_timestamps.keys(), key=lambda x: len(observation_timestamps[x]), reverse=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    for i, star in enumerate(sorted_stars):
        obs_days = observation_timestamps[star]
        ax.plot(obs_days, [i] * len(obs_days), 'o', markersize=3, label=star)

    ax.set_yticks(range(len(sorted_stars)))
    ax.set_yticklabels(sorted_stars)
    ax.set_xlabel('Day of the Year')
    ax.set_ylabel('Star (sorted by observation count)')
    ax.set_title('Generated Observation Timestamps Over the Year')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()




#neid functions 

def vsini_scaling(vsini=2.0):
    """
    scale the RV precision based to account for rotational broadening effects
    vsini: Projected stellar rotational velocity (km s-1)

    returns a scaling ratio
    """
    precision_ratio = 0.000103 * vsini**4. - 0.004042 * vsini**3 + 0.048354 * vsini ** 2. - 0.014283 * vsini + 0.868

    return precision_ratio

def NEID_exptime_RV(teff, vmag, rv_precision, seeing=0.8, vsini=2.0, use_order=False, order=0):
    """
    calculate exposure time required to achieve specified precision for given inputs
    teff:            Effective Temperature (K)
    vmag:            V-band magnitude
    rv_precision:    Desired Radial Velocity Precision (m/s)
    seeing:          Atmospheric seeing (arcsec)
    vsini:           Projected stellar rotational velocity (km/s)

    set use_order=True to calculate exposure time for a specific order
    """
    exptime_grid = fits.open(os.path.join(path_to_grid,'photon_grid_exptime.fits'))[0].data
    teff_grid = fits.open(os.path.join(path_to_grid,'photon_grid_teff.fits'))[0].data
    vmag_grid = fits.open(os.path.join(path_to_grid,'photon_grid_vmag.fits'))[0].data
    seeing_grid=np.array([0.3,0.5,0.7,0.8,0.9,1.1,1.3,1.5,1.7,1.9])
    logexp=np.log10(exptime_grid)

    bound_test=True
    if teff<np.min(teff_grid) or teff>np.max(teff_grid):
        print("Temperature out of bounds. The allowed range is %d K to %d K." % (np.min(teff_grid), np.max(teff_grid)))
        bound_test=False
    if vmag<np.min(vmag_grid) or vmag>np.max(vmag_grid):
        print("Magnitude out of bounds. The allowed range is V = %d to V = %d." % (np.min(vmag_grid), np.max(vmag_grid)))
        bound_test=False
    if seeing<0.3 or seeing>1.9:
        print("Seeing out of bounds. The allowed range is 0.3\" to 1.9\".")
        bound_test=False
    if bound_test==False:
        return np.nan

    if use_order==True:
        order_grid = fits.open(os.path.join(path_to_grid,'order_wvl_centers.fits'))[0].data[0]
        order_loc=np.where(order_grid==order)[0][0]
        rvprec_grid=[]
        for s in seeing_grid:
            rvprec_grid_order = fits.open(os.path.join(path_to_grid,'dv_uncertainty_master_order_seeing','dv_uncertainty_master_order_'+str(s)+'.fits'))[0].data
            grid_s=rvprec_grid_order[order_loc]
            rvprec_grid.append(grid_s)
    else:
        rvprec_grid=[]
        for s in seeing_grid:
            grid_s=fits.open(os.path.join(path_to_grid,'dv_uncertainty_master_seeing','dv_uncertainty_master_'+str(s)+'.fits'))[0].data
            rvprec_grid.append(grid_s)
        rvprec_grid=np.array(rvprec_grid)

    teff_index=InterpolatedUnivariateSpline(teff_grid,
                                            np.arange(len(teff_grid), dtype=np.double))(teff)
    vmag_index=InterpolatedUnivariateSpline(vmag_grid,
                                            np.arange(len(vmag_grid), dtype=np.double))(vmag)
    seeing_index=InterpolatedUnivariateSpline(seeing_grid,
                                            np.arange(len(seeing_grid), dtype=np.double))(seeing)

    j=0
    eta=1e10
    while eta>rv_precision:
        exptime=2*(j+6)
        if exptime>np.max(exptime_grid):
            # print("\nMaximum Exposure Time Exceeded (t>3600s).\n")
            return np.nan
        exptime_index=InterpolatedUnivariateSpline(logexp, np.arange(len(exptime_grid),
                                                                     dtype=np.double))(np.log10(exptime))
        rvprec_interpolator=RegularGridInterpolator((np.arange(len(seeing_grid)),
                                           np.arange(len(exptime_grid)),
                                           np.arange(len(vmag_grid)),
                                           np.arange(len(teff_grid))),
                                                  rvprec_grid)
        inputs=[seeing_index, exptime_index, vmag_index, teff_index]
        eta=rvprec_interpolator(inputs)[0]*vsini_scaling(vsini)
        j+=1

    return exptime

def calculate_exposures_and_uniform_cycles(stars_to_process, teff_dict, vmag_dict, vsini_dict,
                                             target_nights_for_uniformity=20, hours_per_night=10):
    """
    Calculates exposure times for a given list of stars and determines max uniform observation cycles.
    """
    exposure_times = {}
    total_time_per_cycle_seconds = 0
    print("\nCalculating observation times for selected stars (accounting for vsini):")

    valid_selected_stars = []

    for hd in stars_to_process:
        teff = teff_dict.get(hd)
        vmag = vmag_dict.get(hd)
        vsini = vsini_dict.get(hd, 1.0)

        if teff is None or vmag is None or vsini is None:
            print(f"Warning: Missing data for HD {hd} (Teff/Vmag/vsini). Skipping exposure calculation.")
            continue

        exptime = NEID_exptime_RV(teff, vmag, 0.5, seeing=1.0, vsini=vsini)

        if not np.isnan(exptime):
            total_obs_duration = exptime
            exposure_times[hd] = total_obs_duration
            total_time_per_cycle_seconds += total_obs_duration
            valid_selected_stars.append(hd)
            print(f"HD {hd}: T_eff={teff:.0f}K, Vmag={vmag:.2f}, vsini={vsini:.1f} km/s -> {total_obs_duration:.2f} s")
        else:
            print(f"Warning: Could not calculate observation time for HD {hd}. Skipping.")

    print(f"\nTotal time for one observation cycle (one of each selected star): {total_time_per_cycle_seconds / 3600:.2f} hours")

    total_available_time_for_uniformity = target_nights_for_uniformity * hours_per_night * 3600
    max_uniform_cycles_calculated = 0
    if total_time_per_cycle_seconds > 0:
        max_uniform_cycles_calculated = int(total_available_time_for_uniformity // total_time_per_cycle_seconds)

    print(f"Maximum uniform observation cycles of all selected stars in {target_nights_for_uniformity} nights ({hours_per_night} hours/night): {max_uniform_cycles_calculated}")

    return valid_selected_stars, exposure_times, max_uniform_cycles_calculated



#weather functions

def read_monthly_weather_stats(weather_file):
    """
    Reads monthly clear night fractions from a file.
    """
    try:
        with open(weather_file, 'r') as f:
            monthly_fractions = [float(line.strip()) for line in f if line.strip()]
        if len(monthly_fractions) != 12:
            raise ValueError("Weather file must contain exactly 12 monthly clear night fractions.")
        return monthly_fractions
    except IOError:
        print(f"Error: Weather file not found at {weather_file}")
        return []
    except Exception as e:
        print(f"Error reading weather stats: {e}")
        return []

def select_observing_nights_with_weather(monthly_weather_fractions, start_year, num_years=1):
    """
    Selects potential observing nights over a period of years based on monthly clear night fractions.
    """
    observing_dates = []

    for year_offset in range(num_years):
        current_year = start_year + year_offset
        for month_idx, fraction in enumerate(monthly_weather_fractions):
            days_in_current_month = calendar.monthrange(current_year, month_idx + 1)[1]
            for day in range(1, days_in_current_month + 1):
                if random.random() <= fraction:
                    date_str = f"{current_year}-{month_idx + 1:02d}-{day:02d} 00:00:00"
                    observing_dates.append(Time(date_str, format='iso', scale='utc'))
    return sorted(observing_dates)

def generate_observing_schedule(selected_stars, exposure_times, dec_dict, weather_file,
                                     start_year=2025, num_years=1, clear_night_threshold=0.6,
                                     target_obs_per_star=None,
                                     total_sim_nights_limit=None,
                                     cadence_days=None,
                                     even_spread_flag=False):
    """
    Generates a simulated observing schedule attempting to achieve a target number of observations per star,
    respecting specified cadence or even spread.
    """
    location = Observer.at_site('kitt peak')
    monthly_weather_fractions = read_monthly_weather_stats(weather_file)
    potential_observing_dates = select_observing_nights_with_weather(monthly_weather_fractions, start_year, num_years)

    # 1. Determine target observation days based on cadence/spread logic
    stars_for_cadence = {hd: target_obs_per_star for hd in selected_stars}
    
    # Calculate desired observation days in the year (Day 0 to 365)
    desired_obs_days = generate_observation_timestamps(
        selected_stars,
        dec_dict,
        stars_for_cadence,
        start_day=0,
        end_day=365 * num_years,
        cadence_days=cadence_days,
        even_spread=even_spread_flag
    )
    
    # Map desired observation days to potential observation nights
    potential_obs_mjd_map = {obs_date.datetime.timetuple().tm_yday: obs_date for obs_date in potential_observing_dates}
    
    star_observations = {hd: [] for hd in selected_stars}
    observations_count = {hd: 0 for hd in selected_stars}

    print("\nGenerating visibility windows and observations for each target on potential observing nights:")
    
    processed_nights_count = 0
    
    # Combine all unique desired days and sort them
    all_desired_days = sorted(list(set(day for days in desired_obs_days.values() for day in days)))
    
    # Iterate over potential observing dates
    for obs_date in potential_observing_dates:
        if total_sim_nights_limit is not None and processed_nights_count >= total_sim_nights_limit:
            print(f"\nReached total_sim_nights_limit of {total_sim_nights_limit} nights")
            break

        if target_obs_per_star is not None and all(count >= target_obs_per_star for count in observations_count.values()):
            print("\nAll stars have reached their target number of observations")
            break

        processed_nights_count += 1
        
        # Determine the day-of-year for this potential night
        current_day_of_year = obs_date.datetime.timetuple().tm_yday

        try:
            night_start = location.twilight_evening_nautical(obs_date, which='next')
            night_end = location.twilight_morning_nautical(night_start, which='next')
            if night_end < night_start:
                night_end += 1 * u.day

            actual_night_end = night_start + TimeDelta(10 * u.hour) # 10 hours max observing window
            if actual_night_end > night_end:
                actual_night_end = night_end

            if actual_night_end <= night_start:
                continue

            night_start_mjd = night_start.mjd
            night_end_mjd = actual_night_end.mjd

        except Exception as e:
            continue

        stars_to_observe_this_night = []
        for hd in selected_stars:
            
            # Check if this night is one of the desired observation days for this star
            is_desired_night = current_day_of_year in desired_obs_days.get(hd, [])
            
            if is_desired_night and (target_obs_per_star is None or observations_count[hd] < target_obs_per_star):
                target = FixedTarget.from_name(f"HD {hd}")

                # Check visibility
                test_times_in_window = np.linspace(night_start_mjd, night_end_mjd, 5)
                is_visible_this_night = False
                for t_mjd in test_times_in_window:
                    time_point = Time(t_mjd, format='mjd')
                    if location.target_is_up(time_point, target) and \
                                   location.altaz(time_point, target).secz < 1.5: #airmass condition (<1.5 threshold)
                        is_visible_this_night = True
                        break

                if is_visible_this_night:
                    stars_to_observe_this_night.append(hd)

        random.shuffle(stars_to_observe_this_night)

        for hd in stars_to_observe_this_night:
            if target_obs_per_star is None or observations_count[hd] < target_obs_per_star:
                exptime = exposure_times.get(hd)
                if exptime is None:
                    continue

                star_observations[hd].append(night_start + TimeDelta(random.uniform(0, actual_night_end.mjd - night_start.mjd) * u.day))
                observations_count[hd] += 1

    print("\n Summary of Simulated Observations ---")
    total_simulated_observations = 0
    for hd, obs_list in star_observations.items():
        print(f"HD {hd}: {len(obs_list)} observations")
        total_simulated_observations += len(obs_list)

    print(f"\nTotal simulated observations across all selected stars: {total_simulated_observations}")
    print(f"Total potential observing nights (based on weather threshold): {len(potential_observing_dates)}")
    print(f"Actual nights processed in simulation (up to limit): {processed_nights_count}")

    return star_observations

def plot_observations_over_year(simulated_observations_data):
    """
    Plots all generated observation timestamps as a scatter plot:
        x-axis: Day of the year (1–365/366)
        y-axis: Star names (sorted by HD number in descending order)
    """
    if not simulated_observations_data:
        print("No observation data to plot.")
        return

    # Extract the numeric part of the HD identifier for sorting
    def get_hd_number(hd_str):
        match = re.match(r'(\d+)', hd_str)
        if match:
            return int(match.group(1))
        # Handle cases like "131156 A" by treating the non-numeric part as secondary sort key
        return float('-inf')

    star_counts = {hd: len(times) for hd, times in simulated_observations_data.items()}
    
    # sort stars by HD number (descending)
    sorted_stars = sorted(
        star_counts.keys(), 
        key=lambda s: get_hd_number(s), 
        reverse=False # Set reverse=True for ascending HD number
    )
    
    star_to_yindex = {star: idx for idx, star in enumerate(sorted_stars)}

    x_days = []
    y_indices = []
    labels = []

    for star in sorted_stars:
        for obs_time in simulated_observations_data[star]:
            x_days.append(obs_time.datetime.timetuple().tm_yday)
            y_indices.append(star_to_yindex[star])
            labels.append(star)

    plt.figure(figsize=(12, 8))
    plt.scatter(x_days, y_indices, s=20, alpha=0.7, c='blue', edgecolors='black')
    plt.yticks(range(len(sorted_stars)), [f"HD {s}" for s in sorted_stars])
    plt.xlabel("Day of the Year")
    plt.ylabel("Star (sorted by HD number, descending)")
    plt.title("Generated Observation Timestamps Over the Year")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()




# Main Execution

if __name__ == "__main__":
    if not user_stars_input:
        print("no stars input")
    else:
        warnings.filterwarnings('ignore') #cleaning output

        # loading HWO data for fallback lookup
        hwo_fallback_data = load_hwo_candidates(hwo_file)

        # processing user-provided stars to fill in missing data
        selected_stars_list = []
        final_teff_dict, final_vmag_dict, final_vsini_dict, final_dec_dict = {}, {}, {}, {}

        print("\nProcessing user-provided star list...")
        for star_input in user_stars_input:
            profile = get_star_profile(star_input, hwo_fallback_data)
            
            if profile:
                hd_id = profile['hd_id']
                selected_stars_list.append(hd_id)
                final_teff_dict[hd_id] = profile['teff']
                final_vmag_dict[hd_id] = profile['vmag']
                final_vsini_dict[hd_id] = profile['vsini']
                final_dec_dict[hd_id] = profile['dec']

        # run the ETC and scheduling
        if selected_stars_list:
            selected_stars, exposure_times, max_uniform_cycles_calculated = \
                calculate_exposures_and_uniform_cycles(
                    selected_stars_list,
                    final_teff_dict,
                    final_vmag_dict,
                    final_vsini_dict,
                    target_nights_for_uniformity=20,
                    hours_per_night=10
                )

            simulated_observations_data = generate_observing_schedule(
                selected_stars,
                exposure_times,
                final_dec_dict,
                weather_file,
                start_year=2025,
                num_years=1,
                clear_night_threshold=0.6,
                target_obs_per_star=max_uniform_cycles_calculated,
                cadence_days=cadence_days,
                even_spread_flag=even_spread_flag # Pass the desired cadence flag
            )


            try:
                with open(output_timestamps_file, 'w', newline='') as outfile:
                    outfile.write("Observation Timestamp\n")
                    for hd_number, obs_times in simulated_observations_data.items():
                        for t in obs_times:
                            outfile.write(f"{t.iso}\n")
                print(f"Observation timestamps saved to '{output_timestamps_file}'")

            except IOError as e:
                print(f"Error writing to file {output_timestamps_file}: {e}")

            print("\n plotting observation timestamps over the year ---")
            plot_observations_over_year(simulated_observations_data)

        else:
            print("\nExecution complete. No stars from the user input could be processed.")
        
        print("\nExecution complete.")