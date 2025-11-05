# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 23:59:05 2025

@author: shire
"""

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
from astroquery.simbad.core import Simbad as SimbadCore

from stellar_gp.argo_model import GPModel, GPData
from stellar_gp.argo_model import GranulationKernel,OscillationKernel,QPKernel,PerKernel,M52Kernel,M32Kernel,WNKernel, SEKernel,M52PDKernel
from stellar_gp.stellar_scalings import get_stellar_hypers, calc_Pg



# USER INPUT / CONFIGURATION 

# optionally provide 'teff', 'vmag', 'vsini', 'dec' (if not provided -> SIMBAD -> HWO CSV)
user_stars_input = [ {'name': 'HD 166'},
 # e.g. {'name': 'HD ', 'vmag':, 'vsini':},
]

# File paths 
hwo_file = r"C:/Users/shire/Downloads/DI_STARS_EXEP_2025.06.20_11.46.50.csv"
weather_file = r"C:\Users\shire\Downloads\KPNO.txt"
output_timestamps_file = r"C:\Users\shire\Downloads\JUST TIMESTAMPS.txt"
path_to_grid = r"C:\Users\shire\Downloads\neid etc"

# SCHEDULING OPTIONS (set only one)
cadence_days = None # None = use even_spread_flag
even_spread_flag = True # True = spread observations evenly



# ASTROQUERY FUNCTIONS
def get_ra_dec(name):
    simbad = Simbad()
    result = simbad.query_object(name)
    if result is None:
        return None, None
    colnames = result.colnames
    ra_col = None
    dec_col = None
    for c in colnames:
        cl = c.lower()
        if ra_col is None and "ra" in cl:
            if cl == "ra" or cl.endswith(".ra") or cl == "raj2000" or cl == "ra_str":
                ra_col = c
                break
    if ra_col is None:
        for c in colnames:
            if "ra" in c.lower():
                ra_col = c
                break
    for c in colnames:
        cl = c.lower()
        if dec_col is None and "dec" in cl:
            if cl == "dec" or cl.endswith(".dec") or cl == "decj2000" or cl == "dec_str":
                dec_col = c
                break
    if dec_col is None:
        for c in colnames:
            if "dec" in c.lower():
                dec_col = c
                break
    if ra_col is None or dec_col is None:
        return None, None
    ra_val = result[ra_col][0]
    dec_val = result[dec_col][0]
    if isinstance(ra_val, str) or isinstance(dec_val, str):
        try:
            sc = SkyCoord(f"{ra_val} {dec_val}", frame="icrs")
            return float(sc.ra.deg), float(sc.dec.deg)
        except Exception:
            try:
                sc = SkyCoord(ra_val, dec_val, unit=(u.hourangle, u.deg), frame="icrs")
                return float(sc.ra.deg), float(sc.dec.deg)
            except Exception:
                return None, None
    try:
        return float(ra_val), float(dec_val)
    except Exception:
        return None, None

def get_vmag(name):
    """
    Retrieve Johnson V-band magnitude for `name` using SIMBAD's allfluxes votable field.
    Returns a float (V magnitude) or None if not found/parsable.
    """
    simbad = Simbad()
    # Request the full set of flux/magnitude fields
    simbad.add_votable_fields("allfluxes")
    result = simbad.query_object(name)
    if result is None:
        return None

    # Helper to parse numeric prefix from strings like "6.07 [0.01]" or "6.07"
    def extract_number(x):
        if x is None:
            return None
        s = str(x).strip()
        if s == '' or s == '--':
            return None
        # look for a leading float or integer
        m = re.search(r'[-+]?\d+(\.\d+)?', s)
        if m:
            try:
                return float(m.group(0))
            except Exception:
                return None
        return None

    # First pass: look for clear names like FLUX_V, *_V, or something containing 'v' and 'flux'/'mag'
    for col in result.colnames:
        cl = col.lower()
        # common patterns: 'flux_v', 'allfluxes_v', 'v_mag', 'flux.v', 'v'
        tokens = re.split(r'[^a-z0-9]+', cl)
        if ('v' in tokens or cl.endswith('_v') or cl.endswith('.v')) and ('flux' in cl or 'mag' in cl or 'allfluxes' in cl):
            val = result[col][0]
            num = extract_number(val)
            if num is not None:
                return num

    # Second pass: tolerant search for any column that contains '_v' or endswith 'v' and looks numeric
    for col in result.colnames:
        cl = col.lower()
        if '_v' in cl or cl.endswith('v'):
            val = result[col][0]
            num = extract_number(val)
            if num is not None:
                return num

    # Third pass: fallback — inspect any column whose values look like photometric magnitudes (small positive numbers ~0-30)
    for col in result.colnames:
        for i in range(len(result)):
            val = result[col][i]
            num = extract_number(val)
            if num is not None and  -5.0 < num < 40.0:
                # Heuristic: photometric magnitudes are typically in this range
                # To avoid false positives, prefer columns whose name contains common photometric keywords
                if any(k in col.lower() for k in ('flux', 'mag', 'allfluxes', 'v', 'vmag')):
                    return num
    return None

def get_vsini(name):
    """Retrieve the most recent vsini (rotational velocity) from mesRot table."""
    simbad = Simbad()
    simbad.add_votable_fields("mesRot")
    result = simbad.query_object(name)
    if result is None:
        return None

    # identify available rotation measurement columns
    possible_cols = [c for c in result.colnames if "mesrot.vsini" in c.lower()]
    if not possible_cols:
        return None
    vsini_col = possible_cols[0]
    bib_col = "mesrot.bibcode" if "mesrot.bibcode" in result.colnames else None
    if bib_col is None:
        return None

    # multiple rows may be returned; pick latest year
    rows = []
    for i in range(len(result)):
        val = result[vsini_col][i]
        bib = result[bib_col][i]
        if val is None or bib is None:
            continue
        match = re.match(r"(\d{4})", str(bib))
        if match:
            year = int(match.group(1))
            rows.append((year, float(val)))
    if not rows:
        return None
    latest = max(rows, key=lambda x: x[0])
    return latest[1]

def get_teff(name):
    """Retrieve the most recent Teff (effective temperature) from mesFe_h table."""
    simbad = Simbad()
    simbad.add_votable_fields("mesFe_h")
    result = simbad.query_object(name)
    if result is None:
        return None

    teff_col = None
    bib_col = None
    for c in result.colnames:
        if "mesfe_h.teff" in c.lower():
            teff_col = c
        if "mesfe_h.bibcode" in c.lower():
            bib_col = c
    if teff_col is None or bib_col is None:
        return None

    rows = []
    for i in range(len(result)):
        val = result[teff_col][i]
        bib = result[bib_col][i]
        if val is None or bib is None:
            continue
        match = re.match(r"(\d{4})", str(bib))
        if match:
            year = int(match.group(1))
            rows.append((year, float(val)))
    if not rows:
        return None
    latest = max(rows, key=lambda x: x[0])
    return latest[1]

# DATA PROCESSING FUNCTIONS
def query_star_data_astroquery(star_name):
    """
    Queries Simbad for star data (RA, Dec, vsini, Vmag, Teff) using the new functions.
    Returns a dict with keys: hd_id, dec, vsini, vmag, teff (maybe None if missing).
    """
    data = {'hd_id': None, 'dec': None, 'vsini': None, 'vmag': None, 'teff': None}

    try:
        # Extract HD ID from star name
        name_match = re.search(r'HD\s*(\d+[A-Za-z]*)', star_name, re.IGNORECASE)
        data['hd_id'] = name_match.group(1).strip() if name_match else star_name.replace(' ', '_')

        # Get RA and Dec
        ra, dec = get_ra_dec(star_name)
        if dec is not None:
            data['dec'] = dec

        # Get V magnitude
        vmag = get_vmag(star_name)
        if vmag is not None:
            data['vmag'] = vmag

        # Get vsini
        vsini = get_vsini(star_name)
        if vsini is not None:
            data['vsini'] = vsini  

        # Get Teff
        teff = get_teff(star_name)
        if teff is not None:
            data['teff'] = teff

        print(f"Successfully retrieved from SIMBAD: hd_id={data['hd_id']}, dec={data['dec']}, vsini={data['vsini']}, vmag={data['vmag']}, teff={data['teff']}")
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

    return hd_set, teff_dict, vsini_dict, dec_dict, vmag_dict

def get_star_profile(star_input, hwo_fallback_data):
    """
    Building a complete star profile using a tiered data retrieval system:
    1. User-provided data
    2. SIMBAD query (using new functions)
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

    # Fill in missing data from SIMBAD using new functions
    if any(val is None for val in [profile['teff'], profile['dec'], profile['vsini'], profile['vmag']]):
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
            if profile['teff'] is None and simbad_data.get('teff'):
                profile['teff'] = simbad_data['teff']

    # making sure we have an HD ID for HWO lookup
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

    # sanity checks
    required_keys = ['hd_id', 'teff', 'dec', 'vsini', 'vmag']
    missing_keys = [key for key in required_keys if profile.get(key) is None]
    if missing_keys:
        print(f"Incomplete profile for {star_name} (HD {profile['hd_id']}). Missing: {', '.join(missing_keys)}")
        return None
    print(f"Successfully built profile for {star_name} (HD {profile['hd_id']}).")
    return profile




# SCHEDULING FUNCTIONS 
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
        # Optional Plot
        plt.figure(figsize=(10,6))
        plt.plot(days, observable_hours, color="darkblue")
        plt.fill_between(days, 0, observable_hours, alpha=0.3, color="skyblue")
        plt.xlabel("Day of Year")
        plt.ylabel("Observable Hours After Sundown")
        plt.title(f"Observable Hours of {target.name} from {observatory} in {year}")
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
    """
    np.random.seed(seed)
    random.seed(seed)

    observation_timestamps = {}

    for star in stars:
        n_obs = n_observations_per_star.get(star, 0)
        if n_obs <= 0:
            continue

        # Determine the observable days for the star 
        star_dec = dec_dict.get(star)
        observable_days_of_year = None
        if star_dec is not None:
            try:
                observable_days_of_year = observable_window(
                    star_name=f"HD {star}",
                    observatory="Kitt Peak",
                    airmass_limit=1.5,
                    min_hours=2.0,
                    year=2025,
                    plot=False
                )
            except Exception as e:
                print(f"ERROR: Could not calculate observable window for HD {star}: {e}")
                print("Stopping execution due to observable window calculation error.")
                return None

        if observable_days_of_year is None or len(observable_days_of_year) == 0:
            print(f"ERROR: No observable window found for HD {star}.")
            print("Stopping execution - no observable days available for the given target(s).")
            return None

        # Schedule observations within the observable window 
        if cadence_days is not None:
            obs_days_raw = list(range(start_day, end_day, cadence_days))
            obs_days = [day for day in obs_days_raw if day in observable_days_of_year]
            obs_days = obs_days[:n_obs] if len(obs_days) >= n_obs else obs_days 
            
        elif even_spread:
            if n_obs > 1:
                indices = np.round(np.linspace(0, len(observable_days_of_year) - 1, num=n_obs)).astype(int)
                obs_days = list(observable_days_of_year[indices])
            elif n_obs == 1:
                obs_days = [observable_days_of_year[len(observable_days_of_year) // 2]]
            else:
                obs_days = []
        else:
            obs_days = sorted(np.random.choice(observable_days_of_year, size=n_obs, replace=False)) 
            print("Fallback mode: Using random days from 'Observable Days'")

        observation_timestamps[star] = obs_days

    return observation_timestamps




# GENERAL SCHEDULING FUNCTIONS 
def calculate_exposures_and_uniform_visits(stars_to_process, teff_dict, vmag_dict, vsini_dict,
                                          exposure_time_calculator,
                                          target_nights_for_uniformity=20, hours_per_night=10):
    """
    Generic function to calculate exposure times and determine max uniform observation visits.
    Can be used with any instrument's exposure_time_calculator function.
    """
    exposure_times = {}
    total_time_per_cycle_seconds = 0
    print("\nCalculating observation times for selected stars:")

    valid_selected_stars = []

    for hd in stars_to_process:
        teff = teff_dict.get(hd)
        vmag = vmag_dict.get(hd)
        vsini = vsini_dict.get(hd)

        if teff is None or vmag is None or vsini is None:
            print(f"Warning: Missing data for HD {hd} (Teff/Vmag/vsini). Skipping exposure calculation.")
            continue

        # Use the provided exposure time calculator (instrument-specific)
        exptime = exposure_time_calculator(teff, vmag, vsini)

        if not np.isnan(exptime):
            total_obs_duration = exptime
            exposure_times[hd] = total_obs_duration
            total_time_per_cycle_seconds += total_obs_duration
            valid_selected_stars.append(hd)
            print(f"HD {hd}: T_eff={teff:.0f}K, Vmag={vmag:.2f}, vsini={vsini:.1f} km/s -> {total_obs_duration:.2f} s")
        else:
            print(f"Warning: Could not calculate observation time for HD {hd}. Skipping.")

    print(f"\nTotal time for one observation cycle: {total_time_per_cycle_seconds / 3600:.2f} hours")

    total_available_time_for_uniformity = target_nights_for_uniformity * hours_per_night * 3600
    max_uniform_visits_calculated = 0
    if total_time_per_cycle_seconds > 0:
        max_uniform_visits_calculated = int(total_available_time_for_uniformity // total_time_per_cycle_seconds)

    print(f"Maximum uniform observation visits in {target_nights_for_uniformity} nights: {max_uniform_visits_calculated}")

    return valid_selected_stars, exposure_times, max_uniform_visits_calculated

def read_monthly_weather_stats(weather_file):
    """
    Reads monthly clear night fractions from a file.
    Generic function reusable for any observatory.
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
    Generic function reusable for any observatory.
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
                               observatory_location='kitt peak',
                               start_year=2025, num_years=1, clear_night_threshold=0.6,
                               target_obs_per_star=None,
                               total_sim_nights_limit=None,
                               cadence_days=None,
                               even_spread_flag=False):
    """
    Generates a simulated observing schedule attempting to achieve a target number of observations per star.
    Generic function reusable for any instrument/observatory.
    """
    location = Observer.at_site(observatory_location)
    monthly_weather_fractions = read_monthly_weather_stats(weather_file)
    potential_observing_dates = select_observing_nights_with_weather(monthly_weather_fractions, start_year, num_years)

    # Determine target observation days based on cadence/spread logic
    stars_for_cadence = {hd: target_obs_per_star for hd in selected_stars}
    
    desired_obs_days = generate_observation_timestamps(
        selected_stars,
        dec_dict,
        stars_for_cadence,
        start_day=0,
        end_day=365 * num_years,
        cadence_days=cadence_days,
        even_spread=even_spread_flag
    )
    
    star_observations = {hd: [] for hd in selected_stars}
    observations_count = {hd: 0 for hd in selected_stars}

    print("\nGenerating visibility windows and observations for each target on potential observing nights:")
    
    processed_nights_count = 0
    
    for obs_date in potential_observing_dates:
        if total_sim_nights_limit is not None and processed_nights_count >= total_sim_nights_limit:
            print(f"\nReached total_sim_nights_limit of {total_sim_nights_limit} nights")
            break

        if target_obs_per_star is not None and all(count >= target_obs_per_star for count in observations_count.values()):
            print("\nAll stars have reached their target number of observations")
            break

        processed_nights_count += 1
        
        current_day_of_year = obs_date.datetime.timetuple().tm_yday

        try:
            night_start = location.twilight_evening_nautical(obs_date, which='next')
            night_end = location.twilight_morning_nautical(night_start, which='next')
            if night_end < night_start:
                night_end += 1 * u.day

            actual_night_end = night_start + TimeDelta(10 * u.hour)
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
            is_desired_night = current_day_of_year in desired_obs_days.get(hd, [])
            
            if is_desired_night and (target_obs_per_star is None or observations_count[hd] < target_obs_per_star):
                target = FixedTarget.from_name(f"HD {hd}")

                # Check visibility
                test_times_in_window = np.linspace(night_start_mjd, night_end_mjd, 5)
                is_visible_this_night = False
                for t_mjd in test_times_in_window:
                    time_point = Time(t_mjd, format='mjd')
                    if location.target_is_up(time_point, target) and \
                                   location.altaz(time_point, target).secz < 1.5:
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

    print("\n Summary of Simulated Observations:")
    total_simulated_observations = 0
    for hd, obs_list in star_observations.items():
        print(f"HD {hd}: {len(obs_list)} observations")
        total_simulated_observations += len(obs_list)

    print(f"\nTotal simulated observations across all selected stars: {total_simulated_observations}")
    print(f"Total potential observing nights: {len(potential_observing_dates)}")
    print(f"Actual nights processed in simulation: {processed_nights_count}")

    return star_observations




# VISUALIZATION FUNCTIONS
def plot_observations_over_year(simulated_observations_data):
    """
    Plots all generated observation timestamps as a scatter plot.
    Generic function reusable for any instrument.
    """
    if not simulated_observations_data:
        print("No observation data to plot.")
        return

    def get_hd_number(hd_str):
        match = re.match(r'(\d+)', hd_str)
        if match:
            return int(match.group(1))
        return float('-inf')
    
    star_counts = {hd: len(times) for hd, times in simulated_observations_data.items()}
    sorted_stars = sorted(star_counts.keys(), key=lambda s: get_hd_number(s), reverse=False)
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
    plt.ylabel("Star (sorted by HD number)")
    plt.title("Generated Observation Timestamps Over the Year")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()




# NEID INSTRUMENT MODULE


def apply_neid_vsini_condition(vsini_value):
    """Apply NEID condition: minimum vsini of 1.0 km/s"""
    if vsini_value is not None and vsini_value < 1.0:
        return 1.0
    return vsini_value

def vsini_scaling(vsini=2.0):
    """Scale the RV precision based to account for rotational broadening effects"""
    precision_ratio = 0.000103 * vsini**4. - 0.004042 * vsini**3 + 0.048354 * vsini ** 2. - 0.014283 * vsini + 0.868
    return precision_ratio

def NEID_exptime_RV(teff, vmag, rv_precision, seeing=0.8, vsini=2.0, use_order=False, order=0):
    """NEID-specific exposure time calculation"""
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

    teff_index=InterpolatedUnivariateSpline(teff_grid,np.arange(len(teff_grid), dtype=np.double))(teff)
    vmag_index=InterpolatedUnivariateSpline(vmag_grid,np.arange(len(vmag_grid), dtype=np.double))(vmag)
    seeing_index=InterpolatedUnivariateSpline(seeing_grid,np.arange(len(seeing_grid), dtype=np.double))(seeing)

    j=0
    eta=1e10
    while eta>rv_precision:
        exptime=2*(j+6)
        if exptime>np.max(exptime_grid):
            return np.nan
        exptime_index=InterpolatedUnivariateSpline(logexp, np.arange(len(exptime_grid),dtype=np.double))(np.log10(exptime))
        rvprec_interpolator=RegularGridInterpolator((np.arange(len(seeing_grid)),np.arange(len(exptime_grid)),np.arange(len(vmag_grid)),np.arange(len(teff_grid))),rvprec_grid)
        inputs=[seeing_index, exptime_index, vmag_index, teff_index]
        eta=rvprec_interpolator(inputs)[0]*vsini_scaling(vsini)
        j+=1

    return exptime

def neid_exposure_time_calculator(teff, vmag, vsini):
    """Wrapper function for NEID exposure time calculation with instrument-specific conditions"""
    # Apply NEID-specific conditions
    vsini = apply_neid_vsini_condition(vsini)
    return NEID_exptime_RV(teff, vmag, 0.5, seeing=1.0, vsini=vsini)




# MAIN EXECUTION (running with neid for now)


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

        for star_input in user_stars_input:
            profile = get_star_profile(star_input, hwo_fallback_data)
            
            if profile:
                hd_id = profile['hd_id']
                selected_stars_list.append(hd_id)
                final_teff_dict[hd_id] = profile['teff']
                final_vmag_dict[hd_id] = profile['vmag']
                final_vsini_dict[hd_id] = profile['vsini']
                final_dec_dict[hd_id] = profile['dec']

        if selected_stars_list:
            selected_stars, exposure_times, max_uniform_visits_calculated = \
                calculate_exposures_and_uniform_visits(
                    selected_stars_list,
                    final_teff_dict,
                    final_vmag_dict,
                    final_vsini_dict,
                    exposure_time_calculator=neid_exposure_time_calculator,  # Pass instrument-specific calculator
                    target_nights_for_uniformity=20,
                    hours_per_night=10
                )

            # Check if we got valid stars back
            if not selected_stars:
                print("ERROR: No valid stars with exposure time calculations.")
                exit(1)

            simulated_observations_data = generate_observing_schedule(
                selected_stars,
                exposure_times,
                final_dec_dict,
                weather_file,
                observatory_location='kitt peak',  
                start_year=2025,
                num_years=1,
                clear_night_threshold=0.6,
                target_obs_per_star=max_uniform_visits_calculated,
                cadence_days=cadence_days,
                even_spread_flag=even_spread_flag
            )

            # Check if scheduling was successful
            if simulated_observations_data is None:
                print("ERROR: Observation scheduling failed due to observable window issues.")
                exit(1)

            try:
                with open(output_timestamps_file, 'w', newline='') as outfile:
                    outfile.write("Observation Timestamp\n")
                    for hd_number, obs_times in simulated_observations_data.items():
                        for t in obs_times:
                            outfile.write(f"{t.iso}\n")
                print(f"Observation timestamps saved to '{output_timestamps_file}'")

            except IOError as e:
                print(f"Error writing to file {output_timestamps_file}: {e}")

            print("\n Plotting observation timestamps over the year: ")
            plot_observations_over_year(simulated_observations_data)

        else:
            print("\nExecution complete. No stars from the user input could be processed.")
            exit(1)
        
        print("\nExecution complete.")