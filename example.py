"""
Tutorial script
"""

from simulator import *

# User settings


EXAMPLE_STAR = "HD 166"        # Change to any star name
INSTRUMENT = "NEID"            # Choose "NEID" or "KPF"
CADENCE_DAYS = 7               # Days between observations (None = even spread)
TARGET_NIGHTS = 20             # Number of available observing nights
START_YEAR = 2025              # Starting year for scheduling



# 1. Getting stellar data


"""
Queries SIMBAD for stellar parameters.
Missing values are filled from the HWO compilation.
Builds a complete stellar profile.
"""

print(f"\n1. Stellar parameters for {EXAMPLE_STAR}:")

ra, dec = get_ra_dec(EXAMPLE_STAR)

hwo_file = os.path.join(script_dir, "hwo_star_list_for_neid.xlsx")
hd_set, teff_dict, vsini_dict, dec_dict, vmag_dict = load_hwo_candidates(hwo_file)
hwo_data = (hd_set, teff_dict, vsini_dict, dec_dict, vmag_dict)

profile = get_star_profile({'name': EXAMPLE_STAR}, hwo_data)

if profile:
    print(
        f"   Teff={profile['teff']:.0f} K, "
        f"Vmag={profile['vmag']:.3f}, "
        f"vsini={profile['vsini']:.3f} km/s, "
        f"Dec={profile['dec']:.3f}°, "
        f"RA={ra:.3f}°"
    )



# 2. Observable window


"""
Computes days of the year when the target is observable
from the observatory with:
- Airmass < 1.5
- Minimum 2 hours visibility
"""

print(f"\n2. Observable window for {EXAMPLE_STAR}:")

observable_days = observable_window(
    star_name=EXAMPLE_STAR,
    year=START_YEAR,
    plot=False
)

if observable_days is not None:
    print(f"   {len(observable_days)} observable days in {START_YEAR}")



# 3. Instrument exposure time


"""
Computes the exposure time required to reach
0.5 m/s RV precision for the selected instrument.
"""

print(f"\n3. Exposure time calculation for {INSTRUMENT}:")

if profile:
    if INSTRUMENT == "NEID":
        exposure_time = neid_exposure_time_calculator(
            profile['teff'],
            profile['vmag'],
            profile['vsini']
        )
        observatory_location = "kitt peak"

    elif INSTRUMENT == "KPF":
        exposure_time = kpf_exposure_time_calculator(
            profile['teff'],
            profile['vmag'],
            profile['vsini']
        )
        observatory_location = "keck"

    print(
        f"   Required exposure time: "
        f"{exposure_time:.1f} seconds "
        f"({exposure_time/60:.1f} minutes)"
    )



# 4. Maximum observations


"""
Determines how many observations can be made
given exposure time and available nights.
"""

if profile and observable_days is not None:
    hours_per_night = 10
    available_time_seconds = TARGET_NIGHTS * hours_per_night * 3600
    max_observations = int(available_time_seconds // exposure_time)

    print(f"\n4. Maximum observations in {TARGET_NIGHTS} nights:")
    print(f"   {max_observations} observations possible")



# 5. Weather analysis


"""
Selects potential observing nights using
monthly clear-night statistics.
"""

print(f"\n5. Weather analysis for {INSTRUMENT}:")

if INSTRUMENT == "NEID":
    weather_file = os.path.join(script_dir, "KPNO.txt")
elif INSTRUMENT == "KPF":
    weather_file = os.path.join(script_dir, "Maunakea.txt")

monthly_fractions = read_monthly_weather_stats(weather_file)
potential_nights = select_observing_nights_with_weather(
    monthly_fractions,
    START_YEAR
)

print(f"   Potential clear nights in {START_YEAR}: {len(potential_nights)}")




# 6. Generate observing schedule


"""
Generates final observation timestamps by combining:
- Visibility constraints
- Weather constraints
- Cadence or even-spread logic
"""

print(f"\n6. Generating observation schedule:")

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



# 7. Output + visualization


"""
Produces:
1) A yearly observation cadence plot
2) A synthetic stellar RV curve
3) A CSV file with:
   BJD_TDB, local time, RV, RV uncertainty
"""

print("\n7. Visualizing and generating outputs.")

if schedule and profile['hd_id'] in schedule:

    # Plot yearly observation cadence
    plot_observations_over_year(schedule)

    # Generate and plot stellar RV time series
    times_dt, rv_ts = generate_stellar_rv_timeseries(
        schedule[profile['hd_id']],
        teff=profile['teff']
    )

    plot_stellar_rv(
        times_dt,
        rv_ts,
        EXAMPLE_STAR,
        ra,
        dec,
        observatory_location
    )

    print("   RV plot generated")
    print("   Output CSV written to output_timestamps.csv")

else:
    print("   No observations available to visualize")


print("\nTutorial completed.")
