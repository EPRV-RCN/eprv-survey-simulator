# simpler pipeline (can only select 1 instrument at a time for now)

# IMPORTS
import csv
import calendar
import random
import re
import os
import warnings
import pytz
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from datetime import datetime, timedelta

from astropy import units as u
from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord, EarthLocation
from astroplan import Observer, FixedTarget
from scipy.interpolate import InterpolatedUnivariateSpline, RegularGridInterpolator

from astroquery.simbad import Simbad
from astroquery.simbad.core import Simbad as SimbadCore

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Add the stellar_gp subdirectory to path
stellar_gp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stellar_gp")
sys.path.insert(0, stellar_gp_path)

from stellar_gp.argo_model import GPModel, GPData
from stellar_gp.argo_model import GranulationKernel,OscillationKernel,QPKernel,PerKernel,M52Kernel,M32Kernel,WNKernel, SEKernel,M52PDKernel
from stellar_gp.argo_model import generate_ts, get_stellar_kernels
from stellar_gp.stellar_scalings import get_stellar_hypers, calc_Pg


# USER INPUT / CONFIGURATION MODULE

selected_instrument = "KPF"  # input: "NEID" or "KPF"

# input stars with optional parameters
user_stars_input = [
    {'name': 'HD 166'},
]

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Set all paths relative to the repository root
hwo_file = os.path.join(script_dir, "hwo_star_list_for_neid.xlsx")
output_timestamps_file = os.path.join(script_dir, "output_timestamps.csv")

# scheduling parameters (if neither is selected: random selection)
cadence_days = None      # input: None or number of days // will rule over even_spread_flag
even_spread_flag = True  # input: True or False

# time parameters
start_year = 2025  # input: starting yr for scheduling
num_years = 1  # input: no. of yrs to schedule
target_nights_for_uniformity = 20  # input: target nights for uniformity calculation
hours_per_night = 10  # input: available observing hrs/night

# Stellar variability configuration
stellar_kernel_type = "matern52"   # options: "matern52", "qp", "se", "granulation"
plot_duration_hours = 10
time_step_seconds = 500



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
    simbad.add_votable_fields("allfluxes")
    result = simbad.query_object(name)
    if result is None:
        return None

    def extract_number(x):
        if x is None:
            return None
        s = str(x).strip()
        if s == '' or s == '--':
            return None
        m = re.search(r'[-+]?\d+(\.\d+)?', s)
        if m:
            try:
                return float(m.group(0))
            except Exception:
                return None
        return None

    for col in result.colnames:
        cl = col.lower()
        tokens = re.split(r'[^a-z0-9]+', cl)
        if ('v' in tokens or cl.endswith('_v') or cl.endswith('.v')) and ('flux' in cl or 'mag' in cl or 'allfluxes' in cl):
            val = result[col][0]
            num = extract_number(val)
            if num is not None:
                return num

    for col in result.colnames:
        cl = col.lower()
        if '_v' in cl or cl.endswith('v'):
            val = result[col][0]
            num = extract_number(val)
            if num is not None:
                return num

    for col in result.colnames:
        for i in range(len(result)):
            val = result[col][i]
            num = extract_number(val)
            if num is not None and  -5.0 < num < 40.0:
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

    possible_cols = [c for c in result.colnames if "mesrot.vsini" in c.lower()]
    if not possible_cols:
        return None
    vsini_col = possible_cols[0]
    bib_col = "mesrot.bibcode" if "mesrot.bibcode" in result.colnames else None
    if bib_col is None:
        return None

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
        name_match = re.search(r'HD\s*(\d+[A-Za-z]*)', star_name, re.IGNORECASE)
        data['hd_id'] = name_match.group(1).strip() if name_match else star_name.replace(' ', '_')

        ra, dec = get_ra_dec(star_name)
        if dec is not None:
            data['dec'] = dec

        vmag = get_vmag(star_name)
        if vmag is not None:
            data['vmag'] = vmag

        vsini = get_vsini(star_name)
        if vsini is not None:
            data['vsini'] = vsini

        teff = get_teff(star_name)
        if teff is not None:
            data['teff'] = teff

        return data

    except Exception as e:
        print(f"An error occurred during astroquery for {star_name}: {e}")
        return data

def load_hwo_candidates(filepath):
    """
    Loads star data from the HWO candidates Excel file including vsini, declination, and Vmag.
    Returns a tuple containing:
                           - hd_set (set): A set of HD star identifiers.
                           - teff_dict (dict): Dictionary mapping HD to effective temperature (Teff) values.
                           - vsini_dict (dict): Dictionary mapping HD to vsini values.
                           - dec_dict (dict): Dictionary mapping HD to declination values.
                           - vmag_dict (dict): Dictionary mapping HD to Vmag values from HWO.
    """
    hd_set, teff_dict, vsini_dict, dec_dict, vmag_dict = set(), {}, {}, {}, {}

    try:
        df = pd.read_excel(filepath)

        for index, row in df.iterrows():
            raw_hd = str(row[4]).strip() if pd.notna(row[4]) else ""
            dec_str = str(row[11]).strip() if pd.notna(row[11]) else ""
            hwo_vmag_str = str(row[15]).strip() if pd.notna(row[15]) else ""
            teff_str = str(row[25]).strip() if pd.notna(row[25]) else ""
            vsini_str = str(row[59]).strip() if pd.notna(row[59]) else ""

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
                if vsini_str and vsini_str != "0" and vsini_str != "nan":
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
        print(f"An unexpected error occurred loading HWO candidates data: {e}")
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

    profile = {
        'name': star_name,
        'hd_id': None,
        'teff': star_input.get('teff'),
        'dec': star_input.get('dec'),
        'vsini': star_input.get('vsini'),
        'vmag': star_input.get('vmag')
    }

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

    if profile['hd_id'] is None:
        name_match = re.search(r'HD\s*(\d+[A-Za-z]*)', star_name, re.IGNORECASE)
        if name_match:
            profile['hd_id'] = name_match.group(1).strip()
        else:
            print(f"Could not determine HD identifier for '{star_name}' from input or SIMBAD. Cannot use HWO fallback.")

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

    required_keys = ['hd_id', 'teff', 'dec', 'vsini', 'vmag']
    missing_keys = [key for key in required_keys if profile.get(key) is None]
    if missing_keys:
        print(f"Incomplete profile for {star_name} (HD {profile['hd_id']}). Missing: {', '.join(missing_keys)}")
        return None

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
    location = EarthLocation.of_site(observatory)
    observer = Observer(location=location, name=observatory)

    if star_name is not None:
        target = FixedTarget.from_name(star_name)
    elif ra is not None and dec is not None:
        target = FixedTarget(coord=SkyCoord(ra*u.deg, dec*u.deg), name="CustomTarget")
    else:
        raise ValueError("Please provide either star_name or (ra, dec).")

    altitude_limit = np.degrees(np.arcsin(1/airmass_limit))

    days = np.arange(1, 367)
    observable_hours = []

    for day in days:
        time_midnight = Time(f"{year}-01-01 00:00:00") + (day-1)*u.day
        midnight = observer.midnight(time_midnight, which="nearest")

        try:
            evening_twilight = observer.twilight_evening_astronomical(midnight, which="nearest")
            morning_twilight = observer.twilight_morning_astronomical(midnight, which="next")
        except:
            observable_hours.append(0.0)
            continue

        n_points = 200
        delta_t = np.linspace(0, (morning_twilight - evening_twilight).to(u.hour).value, n_points)*u.hour
        times = evening_twilight + delta_t

        altitudes = observer.altaz(times, target).alt.deg

        mask = altitudes >= altitude_limit
        hours = np.sum(mask) * (delta_t[1]-delta_t[0]).to(u.hour).value
        observable_hours.append(hours)

    observable_hours = np.array(observable_hours)

    if plot==True:
        plt.figure(figsize=(10,6))
        plt.plot(days, observable_hours, color="darkblue")
        plt.fill_between(days, 0, observable_hours, alpha=0.3, color="skyblue")
        plt.xlabel("Day of Year")
        plt.ylabel("Observable Hours After Sundown")
        plt.title(f"Observable Hours of {target.name} from {observatory} in {year}")
        plt.grid(alpha=0.4)
        plt.show()

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
    seed=42,
    observatory_name="Kitt Peak",
    year=2025
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

        star_dec = dec_dict.get(star)
        observable_days_of_year = None
        if star_dec is not None:
            try:
                observable_days_of_year = observable_window(
                    star_name=f"HD {star}",
                    observatory=observatory_name,
                    airmass_limit=1.5,
                    min_hours=2.0,
                    year=year,
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
    location = Observer.at_site(observatory_location)
    monthly_weather_fractions = read_monthly_weather_stats(weather_file)
    potential_observing_dates = select_observing_nights_with_weather(monthly_weather_fractions, start_year, num_years)

    stars_for_cadence = {hd: target_obs_per_star for hd in selected_stars}

    desired_obs_days = generate_observation_timestamps(
        selected_stars,
        dec_dict,
        stars_for_cadence,
        start_day=0,
        end_day=365 * num_years,
        cadence_days=cadence_days,
        even_spread=even_spread_flag,
        observatory_name=location.name if hasattr(location, "name") and location.name else "Kitt Peak",
        year=start_year
    )

    star_observations = {hd: [] for hd in selected_stars}
    observations_count = {hd: 0 for hd in selected_stars}

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
    
    

def plot_stellar_rv(times_dt, rv_ts, star_name, ra, dec, observatory_location):
    if times_dt is None or rv_ts is None:
        return

    import matplotlib.dates as mdates
    import pytz

    # Observatory mapping: location (for barycentric correction) + civil timezone (for labeling)
    obs_key = str(observatory_location).lower()

    if obs_key in ["keck", "maunakea", "mauna kea"]:
        local_tz = pytz.timezone("Pacific/Honolulu")  # HST
        location = EarthLocation.of_site("Keck Observatory")
    elif obs_key in ["kitt peak", "kpno"]:
        local_tz = pytz.timezone("US/Arizona")  # MST, no DST
        location = EarthLocation.of_site("Kitt Peak")
    else:
        # Fallback: keep labels in UTC if site is unknown to this mapping
        local_tz = pytz.utc
        location = EarthLocation.of_site(observatory_location)

    # Build UTC times from the pandas DatetimeIndex
    t_utc = Time(times_dt.to_pydatetime(), scale="utc")

    # Convert to local civil time for plotting
    t_local = t_utc.to_datetime(timezone=local_tz)

    fig, ax = plt.subplots(figsize=(12, 5))

    # Primary axis: local time
    ax.plot(t_local, rv_ts, color="purple")
    
    # Move primary x-axis to the top
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    
    ax.set_xlabel(f"Local Time ({local_tz.zone})")
    ax.set_ylabel("RV (m/s)")
    ax.set_title(f"Synthetic Stellar RV Signal for {star_name} ({stellar_kernel_type})", pad=22)
    ax.grid(True)
    
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%#m-%#d %#H:%M", tz=local_tz)
    )

    # Secondary x-axis BELOW: BJD_TDB labels at the same tick positions
    # Use fewer ticks for BJD to avoid overlap
    tick_locs = ax.get_xticks()[::2]  # take every other tick
    
    tick_dt_local = mdates.num2date(tick_locs, tz=local_tz)
    tick_dt_utc = [dt.astimezone(pytz.utc) for dt in tick_dt_local]
    
    t_ticks = Time(tick_dt_utc, scale="utc")
    target = SkyCoord(ra * u.deg, dec * u.deg)
    
    bjd_ticks = (
        t_ticks.tdb +
        t_ticks.light_travel_time(target, location=location)
    ).jd
    
    ax_bjd = ax.secondary_xaxis("bottom")
    ax_bjd.spines["bottom"].set_position(("outward", 18))
    
    ax_bjd.set_xticks(tick_locs)
    ax_bjd.set_xticklabels([f"{bjd:.2f}" for bjd in bjd_ticks])
    ax_bjd.set_xlabel("Barycentric Julian Date (BJD)")

    plt.tight_layout()
    plt.show()







# NEID INSTRUMENT MODULE

def apply_neid_vsini_condition(vsini_value):
    if vsini_value is not None and vsini_value < 1.0:
        return 1.0
    return vsini_value

def vsini_scaling(vsini=2.0):
    precision_ratio = 0.000103 * vsini**4. - 0.004042 * vsini**3 + 0.048354 * vsini ** 2. - 0.014283 * vsini + 0.868
    return precision_ratio

def NEID_exptime_RV(teff, vmag, rv_precision, seeing=0.8, vsini=2.0, use_order=False, order=0):
    neid_etc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "etc_neid")

    exptime_grid = fits.open(os.path.join(neid_etc_path,'photon_grid_exptime.fits'))[0].data
    teff_grid = fits.open(os.path.join(neid_etc_path,'photon_grid_teff.fits'))[0].data
    vmag_grid = fits.open(os.path.join(neid_etc_path,'photon_grid_vmag.fits'))[0].data
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
        order_grid = fits.open(os.path.join(neid_etc_path,'order_wvl_centers.fits'))[0].data[0]
        order_loc=np.where(order_grid==order)[0][0]
        rvprec_grid=[]
        for s in seeing_grid:
            rvprec_grid_order = fits.open(os.path.join(neid_etc_path,'dv_uncertainty_master_order_seeing','dv_uncertainty_master_order_'+str(s)+'.fits'))[0].data
            grid_s=rvprec_grid_order[order_loc]
            rvprec_grid.append(grid_s)
    else:
        rvprec_grid=[]
        for s in seeing_grid:
            grid_s=fits.open(os.path.join(neid_etc_path,'dv_uncertainty_master_seeing','dv_uncertainty_master_'+str(s)+'.fits'))[0].data
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
    vsini = apply_neid_vsini_condition(vsini)
    return NEID_exptime_RV(teff, vmag, 0.5, seeing=1.0, vsini=vsini)




# KPF INSTRUMENT MODULE

def kpf_exposure_time_calculator(teff, vmag, vsini):
    return kpf_etc_rv(teff, vmag, 0.5)

def kpf_etc_rv(teff, vmag, sigma_rv):
    kpf_etc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "etc_kpf")

    teff_grid_file = os.path.join(kpf_etc_path, 'photon_grid_teff.fits')
    vmag_grid_file = os.path.join(kpf_etc_path, 'photon_grid_vmag.fits')
    exp_time_grid_file = os.path.join(kpf_etc_path, 'photon_grid_exptime.fits')

    sigma_rv_total_file = os.path.join(kpf_etc_path, 'dv_uncertainty_master.fits')

    sigma_rv_grid = fits.getdata(sigma_rv_total_file)
    teff_grid = fits.getdata(teff_grid_file)
    vmag_grid = fits.getdata(vmag_grid_file)
    exptime_grid = fits.getdata(exp_time_grid_file)
    logexp = np.log10(exptime_grid)

    flag_bound = True
    if teff < np.min(teff_grid) or teff > np.max(teff_grid):
        print("Temperature out of bounds (%d K to %d K)" % (np.amin(teff_grid), np.amax(teff_grid)))
        flag_bound = False
    if vmag < np.min(vmag_grid) or vmag > np.max(vmag_grid):
        print("Magnitude out of bounds (V = %d to V = %d)" % (np.amin(vmag_grid), np.amax(vmag_grid)))
        flag_bound = False
    if not flag_bound:
        return np.nan

    teff_index_spline = InterpolatedUnivariateSpline(teff_grid,
                                                     np.arange(len(teff_grid),
                                                               dtype=np.double))
    teff_location = teff_index_spline(teff)

    vmag_index_spline = InterpolatedUnivariateSpline(vmag_grid,
                                                     np.arange(len(vmag_grid),
                                                               dtype=np.double))
    vmag_location = vmag_index_spline(vmag)

    ind = 2
    maxout = 1e10

    while maxout > sigma_rv:
        trial_exp = min(exptime_grid) + ind

        exptime_index = InterpolatedUnivariateSpline(logexp,
                                                     np.arange(len(exptime_grid),
                                                               dtype=np.double))(np.log10(trial_exp))

        sigma_rv_interpolator = RegularGridInterpolator((np.arange(len(exptime_grid)),
                                                         np.arange(len(vmag_grid)),
                                                         np.arange(len(teff_grid))),
                                                        sigma_rv_grid)

        inputs = [exptime_index, vmag_location, teff_location]
        maxout = sigma_rv_interpolator(inputs)[0]
        ind += 1

    exptime = trial_exp
    return exptime




# STELLAR GP KERNELS

def build_stellar_kernel(kernel_type, teff=None):
    if kernel_type == "granulation":
        # Built in stellar kernels expect time in seconds in GPData (per example notebook)
        # We use solar logg=4.43 and the available Teff if provided; otherwise solar defaults
        logg_sun = 4.43
        if teff is None:
            kosc, kgran1, kgran2 = get_stellar_kernels()
        else:
            kosc, kgran1, kgran2 = get_stellar_kernels(logg_sun, teff)
        return [kosc, kgran1, kgran2]

    if kernel_type == "matern52":
        return M52PDKernel(0.5, 0.5, 0.05)
    if kernel_type == "qp":
        return QPKernel(1.0, 10.0, 0.5, 0.5)
    if kernel_type == "se":
        return SEKernel(1.0, 0.5)

    raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    
def sample_stellar_rv_at_observations(obs_times, ra, dec, observatory_location):
    import pytz

    # Observatory setup
    obs_key = observatory_location.lower()
    if obs_key in ["keck", "maunakea", "mauna kea"]:
        local_tz = pytz.timezone("Pacific/Honolulu")
        location = EarthLocation.of_site("Keck Observatory")
    elif obs_key in ["kitt peak", "kpno"]:
        local_tz = pytz.timezone("US/Arizona")
        location = EarthLocation.of_site("Kitt Peak")
    else:
        local_tz = pytz.utc
        location = EarthLocation.of_site(observatory_location)

    # Convert observation times to Astropy Time
    t_utc = Time([t.to_datetime() for t in obs_times], scale="utc")

    # Reference epoch
    t0 = t_utc[0]

    # GP time in days
    t_days = (t_utc - t0).to(u.day).value
    duration_days = np.full_like(t_days, 500.0 / 86400.0)

    rv = np.zeros_like(t_days)
    rv_err = np.full_like(t_days, 0.5)

    kernels = build_stellar_kernel(stellar_kernel_type)
    if isinstance(kernels, list):
        model = GPModel(kernels=kernels)
    else:
        model = GPModel(kernels=[kernels])


    data = GPData(
        time=t_days,
        duration=duration_days,
        rv=rv,
        rv_err=rv_err
    )

    rv_samples = generate_ts(model, data, ignore_errs=True)

    # Compute BJD_TDB
    target = SkyCoord(ra * u.deg, dec * u.deg)
    bjd = (t_utc.tdb + t_utc.light_travel_time(target, location=location)).jd

    # Local times
    local_times = t_utc.to_datetime(timezone=local_tz)

    return bjd, local_times, rv_samples, rv_err


def generate_stellar_rv_timeseries(timestamps, teff=None, obs_duration_sec=500.0):
    if len(timestamps) == 0:
        return None, None

    t0 = timestamps[0].to_datetime()
    n_points = int(plot_duration_hours * 3600 / time_step_seconds)

    times_dt = pd.date_range(
        start=t0,
        periods=n_points,
        freq=f"{time_step_seconds}s"
    )

    times_sec_rel = (times_dt - t0).total_seconds().to_numpy()

    # Exposure duration (seconds)
    duration_sec = np.full_like(times_sec_rel, float(obs_duration_sec), dtype=float)

    # Base arrays
    rv = np.zeros_like(times_sec_rel, dtype=float)
    rv_err = np.full_like(times_sec_rel, 0.5, dtype=float)

    kernels = build_stellar_kernel(stellar_kernel_type, teff=teff)

    # Time unit handling consistent with the example notebook:
    # - get_stellar_kernels() (osc + gran) expects time in seconds in GPData
    # - non-SHO examples often use days; keep those consistent
    if stellar_kernel_type == "granulation":
        time_for_gp = times_sec_rel
        duration_for_gp = duration_sec
        data = GPData(time=time_for_gp, duration=duration_for_gp, rv=rv, rv_err=rv_err)
        stellar_model = GPModel(kernels=kernels)
        rv_ts = generate_ts(stellar_model, data, ignore_errs=True)
        return times_dt, rv_ts

    # Non-SHO path: use days consistently
    time_days = times_sec_rel / (24.0 * 3600.0)
    duration_days = duration_sec / (24.0 * 3600.0)
    data = GPData(time=time_days, duration=duration_days, rv=rv, rv_err=rv_err)

    stellar_model = GPModel(kernels=[kernels])
    rv_ts = generate_ts(stellar_model, data, ignore_errs=True)
    return times_dt, rv_ts




# MAIN EXECUTION

if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    if selected_instrument == "NEID":
        exposure_time_calculator = neid_exposure_time_calculator
        observatory_location = 'kitt peak'
        weather_file = os.path.join(script_dir, "KPNO.txt")
    elif selected_instrument == "KPF":
        exposure_time_calculator = kpf_exposure_time_calculator
        observatory_location = 'keck'
        weather_file = os.path.join(script_dir, "Maunakea.txt")
    else:
        raise ValueError("Unknown instrument")

    hwo_fallback_data = load_hwo_candidates(hwo_file)

    selected_stars_list = []
    final_teff_dict = {}
    final_vmag_dict = {}
    final_vsini_dict = {}
    final_dec_dict = {}

    for star_input in user_stars_input:
        profile = get_star_profile(star_input, hwo_fallback_data)
        if profile:
            hd = profile['hd_id']
            selected_stars_list.append(hd)
            final_teff_dict[hd] = profile['teff']
            final_vmag_dict[hd] = profile['vmag']
            final_vsini_dict[hd] = profile['vsini']
            final_dec_dict[hd] = profile['dec']

    selected_stars, exposure_times, max_uniform_visits_calculated = \
        calculate_exposures_and_uniform_visits(
            selected_stars_list,
            final_teff_dict,
            final_vmag_dict,
            final_vsini_dict,
            exposure_time_calculator,
            target_nights_for_uniformity,
            hours_per_night
        )

    simulated_observations_data = generate_observing_schedule(
        selected_stars,
        exposure_times,
        final_dec_dict,
        weather_file,
        observatory_location=observatory_location,
        start_year=start_year,
        num_years=num_years,
        clear_night_threshold=0.6,
        target_obs_per_star=max_uniform_visits_calculated,
        cadence_days=cadence_days,
        even_spread_flag=even_spread_flag
    )

    with open(output_timestamps_file, "w") as f:
        f.write("BJD_TDB,LOCAL_TIME,RV,RV_ERR\n")
    
        for hd, obs_times in simulated_observations_data.items():
            ra, dec = get_ra_dec(f"HD {hd}")
    
            bjd, local_times, rv_vals, rv_errs = sample_stellar_rv_at_observations(
                obs_times,
                ra,
                dec,
                observatory_location
            )
    
            for i in range(len(bjd)): 
                rv_val = float(rv_vals[i])
                rv_err = float(rv_errs[i])
            
                f.write(
                    f"{bjd[i]:.8f},"
                    f"{local_times[i].strftime('%Y-%m-%d %H:%M:%S.%f')},"
                    f"{rv_val:.3f},"
                    f"{rv_err:.3f}\n"
                )


                
    # Plot observation cadence over the year
    plot_observations_over_year(simulated_observations_data)
    
    # Generate and plot stellar RVs
    for hd, obs_times in simulated_observations_data.items():
        ra, dec = get_ra_dec(f"HD {hd}")
    
        times_dt, rv_ts = generate_stellar_rv_timeseries(obs_times)
    
        plot_stellar_rv(
            times_dt,
            rv_ts,
            f"HD {hd}",
            ra,
            dec,
            observatory_location
        )
    
    print("Execution complete.")
