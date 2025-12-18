"""
Tutorial script
"""

from simulator import *


# User settings
EXAMPLE_STAR = "HD 166"        # Change to any star name
INSTRUMENT = "NEID"            # Choose "NEID" or "KPF"
CADENCE_DAYS = 7               # Days between observations (None = even spread)
TARGET_NIGHTS = 20             # No. of nights to distribute observations over / Total no. of available nights 
START_YEAR = 2025              # Starting year for scheduling


# 1. Getting star data 
"""
Functions to get star's data from SIMBAD using astroquery. 
Any missing parameters are filled in from HWO data compilations.
Builds a profile with Teff, Vmag, vsini, RA, and Dec values.
"""
ra, dec = get_ra_dec(EXAMPLE_STAR)
print(f"\n1. Stellar parameters for {EXAMPLE_STAR}:")
hwo_file = os.path.join(script_dir, "hwo_star_list_for_neid.xlsx")
hd_set, teff_dict, vsini_dict, dec_dict, vmag_dict = load_hwo_candidates(hwo_file)
hwo_data = (hd_set, teff_dict, vsini_dict, dec_dict, vmag_dict)

profile = get_star_profile({'name': EXAMPLE_STAR}, hwo_data)
if profile:
    print(f"   Teff={profile['teff']:.0f}K, Vmag={profile['vmag']:.3f}, "
          f"vsini={profile['vsini']:.3f} km/s, Dec={profile['dec']:.3f}°, RA={ra:.3f}°")



# 2. Calculate observable days
"""
Calculates days when the target is observable from the observatory.
Uses airmass constraint (< 1.5) and minimum observable time (> 2 hours).
Returns days of the year when the target meets these criteria.
"""
print(f"\n2. Observable window for {EXAMPLE_STAR} from {INSTRUMENT}:")
observable_days = observable_window(
    star_name=EXAMPLE_STAR, 
    year=START_YEAR, 
    plot=False
)
print(f"   {len(observable_days)} observable days in {START_YEAR}")



# 3. Instrument exposure time calculation
"""
Calculates the exposure time needed by the selected instrument to reach 0.5 m/s RV precision.
The calculation uses the star's Teff, Vmag, and vsini along with instrument-specific ETC grids.
Overhead time is NOT included. 
"""
print(f"\n3. Exposure time calculation for {INSTRUMENT}:")
if profile:
    if INSTRUMENT == "NEID":
        exposure_time = neid_exposure_time_calculator(
            profile['teff'], 
            profile['vmag'], 
            profile['vsini']
        )
    elif INSTRUMENT == "KPF":
        exposure_time = kpf_exposure_time_calculator(
            profile['teff'], 
            profile['vmag'], 
            profile['vsini']
        )
    
    print(f"   Required exposure time: {exposure_time:.1f} seconds ({exposure_time/60:.1f} minutes)")



# 4. Determine how many observations can be made and select days
"""
Calculates maximum number of observations possible given the exposure time and available observing nights.
Then selects specific days for observations based on cadence or even spread strategy.
"""
if profile and observable_days is not None:
    # Calculate maximum observations based on exposure time
    hours_per_night = 10
    available_time_seconds = TARGET_NIGHTS * hours_per_night * 3600
    max_observations = int(available_time_seconds // exposure_time)
    
    print(f"\n4. Maximum observations in {TARGET_NIGHTS} nights: {max_observations}")
    
    # Prepare data for generate_observation_timestamps
    stars_list = [profile['hd_id']]
    dec_dict_single = {profile['hd_id']: profile['dec']}
    n_obs_dict = {profile['hd_id']: max_observations}
    
    selected_days = generate_observation_timestamps(
        stars=stars_list,
        dec_dict=dec_dict_single,
        n_observations_per_star=n_obs_dict,
        start_day=0,
        end_day=365,
        cadence_days=CADENCE_DAYS,
        even_spread=(CADENCE_DAYS is None)
    )
    
    if selected_days:
        obs_days = selected_days[profile['hd_id']]
        print(f"   Selected {len(obs_days)} observation days based on {'cadence' if CADENCE_DAYS else 'even spread'}")
        if len(obs_days) > 0:
            print(f"   Days of year: {obs_days}")


# 5. Weather analysis for observatory location
"""
Selects potential observing nights based on monthly clear night fractions.
NEID uses Kitt Peak (KPNO.txt), KPF uses Maunakea (Maunakea.txt).
"""
print(f"\n5. Weather analysis for {INSTRUMENT} location:")
if INSTRUMENT == "NEID":
    weather_file = os.path.join(script_dir, "KPNO.txt")
    observatory_location = 'kitt peak'
elif INSTRUMENT == "KPF":
    weather_file = os.path.join(script_dir, "Maunakea.txt")
    observatory_location = 'keck'

monthly_fractions = read_monthly_weather_stats(weather_file)
potential_nights = select_observing_nights_with_weather(monthly_fractions, START_YEAR)
print(f"   Potential clear nights in {START_YEAR}: {len(potential_nights)}")



# 6. Generate observation timestamps
"""
Generates final observation timestamps by combining:
- Selected observation days (from step 5)
- Weather constraints from observatory location
- Visibility constraints (airmass, altitude, etc.)
"""
print(f"\n6. Generating timestamps:")
if profile:
    schedule = generate_observing_schedule(
        selected_stars=[profile['hd_id']],
        exposure_times={profile['hd_id']: exposure_time},
        dec_dict={profile['hd_id']: profile['dec']},
        weather_file=weather_file,
        observatory_location=observatory_location,
        start_year=START_YEAR,
        num_years=1,
        target_obs_per_star=max_observations,
        cadence_days=CADENCE_DAYS,
        even_spread_flag=(CADENCE_DAYS is None)
    )
    
    if schedule and profile['hd_id'] in schedule:
        obs_count = len(schedule[profile['hd_id']])
        print(f"   Successfully scheduled {obs_count} observations")
        #if obs_count > 0:
           # print(f"   First observation: {schedule[profile['hd_id']][0].iso}")
           # if obs_count > 1:
               # print(f"   Last observation: {schedule[profile['hd_id']][-1].iso}")
               # print(f"   Date range: {(schedule[profile['hd_id']][-1] - schedule[profile['hd_id']][0]).to('day').value:.1f} days")

# 7. Visualizing
"""
Creates a scatter plot showing all scheduled observations throughout the year.
X-axis: Day of year, Y-axis: Star HD number.
"""
print("\n7. Plotting the observation schedule.")
if profile and schedule and profile['hd_id'] in schedule and len(schedule[profile['hd_id']]) > 0:
    plot_observations_over_year(schedule)
else:
    print("No observations to plot")


print("Tutorial completed.")