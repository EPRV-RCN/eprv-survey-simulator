# imports 
import csv
import calendar
import random
import re
import os

import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord, EarthLocation
from astroplan import Observer, FixedTarget
from scipy.interpolate import InterpolatedUnivariateSpline, RegularGridInterpolator

from stellar_gp.argo_model import GPModel, GPData # import the functions for the general model and data setup structure
from stellar_gp.argo_model import GranulationKernel,OscillationKernel,QPKernel,PerKernel,M52Kernel,M32Kernel,WNKernel, SEKernel,M52PDKernel # import the kernel functions
from stellar_gp.argo_model import covariance_matrix,generate_ts, get_stellar_kernels # import general functions
from stellar_gp.stellar_scalings import get_stellar_hypers, calc_Pg  # import the scaling relations

global path_to_grid
path_to_grid = r"C:\Users\shire\Downloads\neid etc"

# Data Loading/Filtering 

def parse_hd_for_sort(hd_str):
    """
    Parses an HD identifier string into a tuple(numeric_part, alphabetic_part) for correct sorting.
    """
    match = re.match(r'(\d+)\s*([A-Za-z]*)', hd_str)
    if match:
        num_part = int(match.group(1))
        alpha_part = match.group(2).strip()
        return (num_part, alpha_part)
    
    print(f"could not parse HD string for sorting: '{hd_str}'. Placing at start of sort.")
    return (float('-inf'), hd_str) # Sorts unparsable entries to the beginning


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
    
    # IGNORE: debugging counters 
    row_counter = 0
    skipped_row_count = 0

    try:
        with open(filepath, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            header = next(reader, None) 
            if header:
                row_counter += 1 

            #print(f"--- Debugging load_hwo_candidates ---")
            #print(f"Header: {header}")

            for row_num_in_file, row in enumerate(reader, start=2): 
                row_counter += 1 
                
                if len(row) < expected_min_cols:
                    # print(f"DEBUG SKIP: Row {row_num_in_file}: Not enough columns ({len(row)} < {expected_min_cols}). Raw HD (if available): '{row[4].strip() if len(row) > 4 else 'N/A'}'")
                    skipped_row_count += 1
                    continue # Skip this row

                raw_hd = row[4].strip()       # Column E
                dec_str = row[11].strip()      # Column L
                hwo_vmag_str = row[15].strip() # Column P
                teff_str = row[25].strip()     # Column Z
                vsini_str = row[59].strip()    # Column BH

                hd_identifier = ""
                if raw_hd.upper().startswith("HD "):
                    hd_identifier = raw_hd[3:].strip()
                else:
                    # print(f"DEBUG SKIP: Row {row_num_in_file}: HD identifier '{raw_hd}' does not start with 'HD '.")
                    skipped_row_count += 1
                    continue 

                if not hd_identifier: # If stripping "HD " left an empty string (e.g., "HD ")
                    # print(f"DEBUG SKIP: Row {row_num_in_file}: HD identifier is empty after stripping 'HD ' from original '{raw_hd}'.")
                    skipped_row_count += 1
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
                            vsini_val = 1.0 #NEID condition
                        vsini_dict[hd_identifier] = round(vsini_val, 1)
                    else:
                        vsini_dict[hd_identifier] = None
                except ValueError:
                    vsini_dict[hd_identifier] = None

                try:
                    dec_dict[hd_identifier] = float(dec_str)
                except ValueError:
                    dec_dict[hd_identifier] = None

    except FileNotFoundError:
        print(f"Error: HWO candidates file not found at {filepath}")
    except Exception as e:
        print(f"An unexpected error occurred loading HWO candidates data: {e}. Last processed row (approx): {row_counter}")
    
    #print(f"--- Debugging Summary ---")
    #print(f"Total rows read from CSV file (including header): {row_counter}")
    #print(f"Total rows explicitly skipped by 'continue' statements: {skipped_row_count}")
    print(f"Final hd_set size: {len(hd_set)}")
    #print(f"--- End Debugging Summary ---")
    
    return hd_set, teff_dict, vsini_dict, dec_dict, vmag_dict


def filter_hwo_stars(hwo_path):
    """
    Criteria: Teff < 6250 K, vsini < 5 km/s, and Dec >= 0 degrees.
    """
    hwo_hd, teff_dict, vsini_dict, dec_dict, hwo_vmag_dict = load_hwo_candidates(hwo_path)

    print(f"\nTotal stars in HWO list: {len(hwo_hd)}\n")

    print("Stars with Teff < 6250 K, vsini < 5 km/s, and Dec ≥ 0 degrees:\n")
    selected_stars = []
    for hd in sorted(list(hwo_hd), key=parse_hd_for_sort):
    
        teff = teff_dict.get(hd)
        vsini = vsini_dict.get(hd)
        dec = dec_dict.get(hd)
        vmag_hwo = hwo_vmag_dict.get(hd)

        if (teff is not None and teff < 6250 and
            vsini is not None and vsini < 5 and
            dec is not None and dec >= 0):

            vmag_display = f"{vmag_hwo:.2f}" if vmag_hwo is not None else "N/A"
            print(f"{hd}: Teff = {teff} K, vsini = {vsini} km/s, Dec = {dec}°, Vmag (HWO) = {vmag_display}")
            selected_stars.append(hd)

    print(f"\nNumber of stars that satisfy all 3 conditions: {len(selected_stars)}")
    return selected_stars, teff_dict, vsini_dict, dec_dict, hwo_vmag_dict

#NEID ETC Functions

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
    teff:          Effective Temperature (K)
    vmag:          V-band magnitude
    rv_precision:  Desired Radial Velocity Precision (m/s)
    seeing:        Atmospheric seeing (arcsec)
    vsini:         Projected stellar rotational velocity (km/s)

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
            print("\nMaximum Exposure Time Exceeded (t>3600s).\n")
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


def calculate_total_time_and_observations(hwo_path, target_nights_for_uniformity=20, hours_per_night=10):
    """
    Calculates exposure times for selected stars and determines max uniform observation cycles.

    Args:
        hwo_path (str): Path to HWO candidates CSV.
        target_nights_for_uniformity (int): The number of nights to consider for the theoretical
                                             'maximum uniform observation cycles' calculation.
        hours_per_night (int): The assumed observing hours per night for the theoretical calculation.

    Returns:
        tuple: selected_stars, teff_dict, hwo_vmag_dict, vsini_dict, exposure_times, max_uniform_cycles_calculated
    """
    selected_stars, teff_dict, vsini_dict, _, hwo_vmag_dict = filter_hwo_stars(hwo_path)

    exposure_times = {}     
    total_time_per_cycle_seconds = 0    

    print("\nCalculating observation times for selected stars (accounting for vsini):")
    for hd in selected_stars:
        teff = teff_dict.get(hd)
        if teff is None:
            print(f"Warning: Teff for HD {hd} not found. Skipping.")
            continue

        vmag = hwo_vmag_dict.get(hd)
        if vmag is None:
            print(f"Warning: Vmag for HD {hd} not found in HWO data. Skipping.")
            continue
        
        # Get the actual vsini for the star, defaulting to 1.0 km/s if not found
        vsini = vsini_dict.get(hd, 1.0)
        
        # fixed seeing of 1.0 to the new ETC function
        exptime = NEID_exptime_RV(teff, vmag, 0.5, seeing=1.0, vsini=vsini)

        if not np.isnan(exptime):
            total_obs_duration = exptime #+ 120 overhead  
            exposure_times[hd] = total_obs_duration
            total_time_per_cycle_seconds += total_obs_duration
            print(f"HD {hd}: {total_obs_duration:.2f} s")
        else:
            print(f"Warning: Could not calculate observation time for HD {hd} (Teff={teff}, Vmag={vmag}, vsini={vsini}). Skipping.")

    print(f"\nTotal time for one observation cycle (one of each selected star): {total_time_per_cycle_seconds / 3600:.2f} hours")

    # theoretical max uniform observations based on the target_nights_for_uniformity
    total_available_time_for_uniformity = target_nights_for_uniformity * hours_per_night * 3600 # in seconds

    max_uniform_cycles_calculated = 0
    if total_time_per_cycle_seconds > 0:
        max_uniform_cycles_calculated = int(total_available_time_for_uniformity // total_time_per_cycle_seconds)

    print(f"Maximum uniform observation cycles of all selected stars in {target_nights_for_uniformity} nights ({hours_per_night} hours/night): {max_uniform_cycles_calculated}")

    return selected_stars, teff_dict, hwo_vmag_dict, vsini_dict, exposure_times, max_uniform_cycles_calculated

# Weather + Night Selection Functions 

def read_monthly_weather_stats(weather_file):
    """
    Reads monthly clear night fractions from a file.
    Assumes the file contains 12 lines, each with a fraction for a month (Jan-Dec).
    """
    try:
        with open(weather_file, 'r') as f:
            monthly_fractions = [float(line.strip()) for line in f if line.strip()]
        if len(monthly_fractions) != 12:
            raise ValueError("Weather file must contain exactly 12 monthly clear night fractions.")
        return monthly_fractions
    except FileNotFoundError:
        print(f"Error: Weather file not found at {weather_file}")
        return []
    except Exception as e:
        print(f"Error reading weather stats: {e}")
        return []

def select_observing_nights_with_weather(monthly_weather_fractions, start_year, num_years=1):
    """
    Selects potential observing nights over a period of years based on monthly clear night fractions.
    For each day, it probabilistically determines if it's a clear night based on the monthly fraction.
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

# Observing Schedule Generation

def generate_observing_schedule(selected_stars, exposure_times, weather_file,
                                 start_year=2025, num_years=1, clear_night_threshold=0.6,
                                 target_obs_per_star=None,
                                 total_sim_nights_limit=None):
    """
    Generates a simulated observing schedule attempting to achieve a target number of observations per star.

    Args:
        selected_stars (list): List of HD star identifiers.
        exposure_times (dict): Dictionary mapping HD to total observation duration (exptime + overhead).
        weather_file (str): Path to the weather data file.
        start_year (int): Starting year for simulation.
        num_years (int): Number of years to simulate.
        clear_night_threshold (float): Probability threshold for a night to be clear.
        target_obs_per_star (int, optional): The target number of observations for EACH star.
                                             If None, it will observe as many as possible.
        total_sim_nights_limit (int, optional): If set, the simulation will stop after this many
                                                  potential observing nights, even if target_obs_per_star
                                                  hasn't been met.

    Returns: 
        dict: A dictionary where keys are HD star numbers and values are lists of
              Astropy Time objects representing their observation times.
    """
    location = Observer.at_site('kitt peak')
    monthly_weather_fractions = read_monthly_weather_stats(weather_file)
    potential_observing_dates = select_observing_nights_with_weather(
        monthly_weather_fractions, start_year, num_years, clear_night_threshold
    )

    star_observations = {hd: [] for hd in selected_stars}
    observations_count = {hd: 0 for hd in selected_stars}

    print("\nGenerating visibility windows and observations for each target on potential observing nights:")
    potential_observing_dates.sort()

    processed_nights_count = 0

    for obs_date in potential_observing_dates:
        if total_sim_nights_limit is not None and processed_nights_count >= total_sim_nights_limit:
            print(f"\nReached total_sim_nights_limit of {total_sim_nights_limit} nights")
            break

        if target_obs_per_star is not None and all(count >= target_obs_per_star for count in observations_count.values()):
            print("\nAll stars have reached their target number of observations")
            break

        processed_nights_count += 1

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
            if target_obs_per_star is None or observations_count[hd] < target_obs_per_star:
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

    print("\n Summary of Simulated Observations ---")
    total_simulated_observations = 0
    for hd, obs_list in star_observations.items():
        print(f"HD {hd}: {len(obs_list)} observations")
        total_simulated_observations += len(obs_list)

    print(f"\nTotal simulated observations across all selected stars: {total_simulated_observations}")
    print(f"Total potential observing nights (based on weather threshold): {len(potential_observing_dates)}")
    print(f"Actual nights processed in simulation (up to limit): {processed_nights_count}")

    return star_observations


# Main Execution 

if __name__ == "__main__":
    hwo_file = r"C:/Users/shire/Downloads/DI_STARS_EXEP_2025.06.20_11.46.50.csv"
    weather_file = r"C:\Users\shire\Downloads\KPNO.txt"
    output_timestamps_file = r"C:\Users\shire\Downloads\sim timestamps.txt"

    print("calculating theoretical uniform observations ---")
    selected_stars, teff_dict, hwo_vmag_dict, vsini_dict_for_calc, exposure_times, max_uniform_cycles_calculated = \
        calculate_total_time_and_observations(hwo_file,
                                             target_nights_for_uniformity=20,
                                             hours_per_night=10)

    print("\n generating simulated observing schedule ---")
    simulated_observations_data = generate_observing_schedule(
        selected_stars,
        exposure_times,
        weather_file,
        start_year=2025,
        num_years=1,    
        clear_night_threshold=0.6,
        target_obs_per_star=max_uniform_cycles_calculated,    
    )

    print("\n writing Observation Timestamps to file ---")
    try:
        with open(output_timestamps_file, 'w', newline='') as outfile:
            outfile.write("Star,Observation Timestamp\n")
            for hd_number, obs_times in simulated_observations_data.items():
                for t in obs_times:
                    outfile.write(f"HD {hd_number},{t.iso}\n")
        print(f"Observation timestamps saved to '{output_timestamps_file}'")

    except IOError as e:
        print(f"Error writing to file {output_timestamps_file}: {e}")

    print("\n execution complete.")