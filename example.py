#!/usr/bin/env python3

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from simulator import (
    get_star_profile, 
    load_hwo_candidates,
    calculate_exposures_and_uniform_visits,
    generate_observing_schedule,
    plot_observations_over_year,
    neid_exposure_time_calculator,
    kpf_exposure_time_calculator
)

# USER CONFIGURATION
star_name = "HD 166"
instrument = "NEID"  # Choose either "NEID" or "KPF"
cadence_days = 7
target_nights = 20

def schedule_star(star_name, instrument, cadence_days, target_nights):
    if instrument == "NEID":
        exposure_calculator = neid_exposure_time_calculator
        weather_file = os.path.join(script_dir, "KPNO.txt")
        observatory = "kitt peak"
        # Set the ETC path for NEID
        os.environ['instrument_etc_path'] = os.path.join(script_dir, "etc_neid")
    elif instrument == "KPF":
        exposure_calculator = kpf_exposure_time_calculator
        weather_file = os.path.join(script_dir, "Maunakea.txt")
        observatory = "keck"
        # Set the ETC path for KPF
        os.environ['instrument_etc_path'] = os.path.join(script_dir, "etc_kpf")
    else:
        print("Error: Instrument must be 'NEID' or 'KPF'")
        return None

    hwo_file = os.path.join(script_dir, "hwo_star_list_for_neid.xlsx")
    hwo_data = load_hwo_candidates(hwo_file)
    
    profile = get_star_profile({'name': star_name}, hwo_data)
    if not profile:
        return None

    stars_list = [profile['hd_id']]
    teff_dict = {profile['hd_id']: profile['teff']}
    vmag_dict = {profile['hd_id']: profile['vmag']}
    vsini_dict = {profile['hd_id']: profile['vsini']}
    dec_dict = {profile['hd_id']: profile['dec']}
    
    selected_stars, exposure_times, max_visits = calculate_exposures_and_uniform_visits(
        stars_list, teff_dict, vmag_dict, vsini_dict, exposure_calculator, target_nights, 10
    )
    
    if not selected_stars:
        return None

    schedule = generate_observing_schedule(
        selected_stars, exposure_times, dec_dict, weather_file,
        observatory_location=observatory, start_year=2025, num_years=1,
        target_obs_per_star=max_visits, cadence_days=cadence_days
    )
    
    return schedule

if __name__ == "__main__":
    schedule = schedule_star(star_name, instrument, cadence_days, target_nights)
    if schedule:
        plot_observations_over_year(schedule)