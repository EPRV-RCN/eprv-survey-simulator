"""
Tutorial script
"""

from multi_instrument_scheduler import *

warnings.filterwarnings('ignore')


# USER SETTINGS

# Stars to process. Each entry is a dict with a required 'name' field.
# You can optionally override any stellar parameter directly here.
# If a parameter is not provided, it is fetched from SIMBAD and then
# from the HWO catalogue as a fallback.
#
# Available override keys:
#   'teff'            — effective temperature in K
#   'vmag'            — V-band magnitude
#   'vsini'           — projected rotational velocity in km/s
#   'dec'             — declination in degrees
#   'chaplin_exptime' — Chaplin (2019) oscillation-averaging exposure time in seconds
#
# Examples:
#   {'name': 'HD 166'}                          # fully automatic lookup
#   {'name': 'HD 166', 'vmag': 6.07}            # override one value, fetch the rest
#   {'name': 'HD 166', 'teff': 5378, 'vmag': 6.07, 'vsini': 6.4, 'dec': 29.0}
STARS_INPUT = [
    {'name': 'HD 166'},
    {'name': 'HD 142860'},
]

INSTRUMENT           = "NEID"    # "NEID" or "KPF"
CADENCE_DAYS         = 3      # Days between observations. None = even spread.
TARGET_NIGHTS        = 40        # Nights used for the visit budget calculation.
START_YEAR           = 2025      # Starting year for scheduling.
HOURS_PER_NIGHT      = 10        # Available observing hours per night.

# Chaplin (2019) oscillation-averaging filter.
# True  → effective exptime = max(ETC_exptime, chaplin_exptime) per star.
# The Chaplin time is a stellar property, independent of instrument.
# Pre-computed values (10 cm/s threshold) are in column BI of the HWO catalogue.
# Stars not in that list fall back to ETC-only with a warning.
APPLY_CHAPLIN        = True

# Weather toggle.
# True  → only schedule on nights that pass the monthly weather model.
# False → treat every night as clear (useful for isolating cadence logic).
APPLY_WEATHER        = True

# Multi-instrument strategy, applied after scheduling is complete.
# Observations are first scheduled for the primary instrument, then
# individual visits are reassigned according to the strategy.
#   "none"             — all observations stay on the primary instrument
#   "weather_fallback" — reassign to secondary on months where the primary
#                        site clear fraction is below 50%
#   "alternate"        — alternate primary and secondary in time order
STRATEGY             = "weather_fallback"
SECONDARY_INSTRUMENT = "KPF"    # Only used when STRATEGY != "none"


# 1. STELLAR PARAMETERS

"""
For each star, parameters are resolved in order:
  1. Values provided directly in STARS_INPUT
  2. SIMBAD query
  3. HWO Excel catalogue fallback

load_hwo_candidates returns six values:
  hd_set, teff_dict, vsini_dict, dec_dict, vmag_dict, chaplin_exptime_dict
The sixth (chaplin_exptime_dict) contains pre-computed Chaplin (2019)
oscillation-averaging exposure times at the 10 cm/s threshold.
"""

print(f"\n1. Stellar parameters")

hwo_fallback_data = load_hwo_candidates(hwo_file)

profiles    = {}
ra_dec_map  = {}

for star_input in STARS_INPUT:
    name    = star_input.get('name', '')
    profile = get_star_profile(star_input, hwo_fallback_data)
    if profile is None:
        print(f"   {name}: could not build profile — skipping.")
        continue
    hd                = profile['hd_id']
    profiles[hd]      = profile
    ra, dec           = get_ra_dec(name)
    ra_dec_map[hd]    = (ra, dec)
    chaplin_str       = (f"{profile['chaplin_exptime']:.0f} s"
                         if profile.get('chaplin_exptime') is not None
                         else "not available")
    print(f"   HD {hd}: Teff={profile['teff']:.0f} K  Vmag={profile['vmag']:.2f}  "
          f"vsini={profile['vsini']:.1f} km/s  Dec={profile['dec']:.2f}°  "
          f"Chaplin={chaplin_str}")

if not profiles:
    print("   No valid star profiles. Exiting.")
    exit(1)


# 2. OBSERVABLE WINDOW

"""
For each star, finds the days of the year when it is above airmass 1.5
for at least 2 continuous hours during astronomical night, from the
primary instrument's site. This is the same constraint the scheduler uses.
"""

obs_site = INSTRUMENT_CONFIG[INSTRUMENT]["observatory"]
print(f"\n2. Observable windows from {obs_site}")

observable_days_map = {}
for hd, profile in profiles.items():
    days = observable_window(
        star_name=f"HD {hd}",
        observatory=obs_site,
        airmass_limit=1.5,
        min_hours=2.0,
        year=START_YEAR,
        plot=False,
    )
    if days is not None:
        observable_days_map[hd] = days
        print(f"   HD {hd}: {len(days)} observable days  (days {days[0]}–{days[-1]})")
    else:
        print(f"   HD {hd}: not observable from {obs_site} in {START_YEAR} — will be skipped")


# 3. EXPOSURE TIMES

"""
For each star, computes the per-visit exposure time needed to reach
0.5 m/s RV precision using the selected instrument's ETC.

If APPLY_CHAPLIN is True, the effective exposure time becomes
max(ETC_exptime, chaplin_exptime) — long enough to also average down
p-mode oscillations to 10 cm/s.

Overhead (acquisition + readout) is then added to get the full cycle time.
  NEID: 273s acquisition + 27s readout = 300s per visit  (WIYN call for proposals)
  KPF:  120s acquisition + 27s readout = 147s per visit  (placeholder)
"""

etc_fn   = INSTRUMENT_CONFIG[INSTRUMENT]["etc"]
overhead = overhead_sec(INSTRUMENT)
acq      = INSTRUMENT_CONFIG[INSTRUMENT]["acquisition_sec"]
readout  = INSTRUMENT_CONFIG[INSTRUMENT]["readout_sec"]

print(f"\n3. Exposure times ({INSTRUMENT})  —  overhead: {acq}s acq + {readout}s readout = {overhead}s/visit")

exposure_times  = {}
cycle_times     = {}
max_visits_map  = {}

for hd, profile in profiles.items():
    if hd not in observable_days_map:
        continue

    etc_time = etc_fn(profile['teff'], profile['vmag'], profile['vsini'])
    if np.isnan(etc_time):
        print(f"   HD {hd}: ETC returned NaN — skipping")
        continue

    if APPLY_CHAPLIN:
        chaplin_time = profile.get('chaplin_exptime')
        if chaplin_time is None:
            print(f"   HD {hd}: Chaplin filter ON but no value found — using ETC only")
            exptime = etc_time
            binding = "ETC"
        else:
            exptime = max(etc_time, chaplin_time)
            binding = "CHAPLIN" if chaplin_time >= etc_time else "ETC"
        print(f"   HD {hd}: ETC={etc_time:.0f}s  Chaplin={chaplin_time if chaplin_time else 'N/A'}s  "
              f"effective={exptime:.0f}s  (binding: {binding})")
    else:
        exptime = etc_time
        print(f"   HD {hd}: ETC={exptime:.0f}s ({exptime/60:.1f} min)")

    cyc                 = cycle_sec(INSTRUMENT, exptime)
    exposure_times[hd]  = exptime
    cycle_times[hd]     = cyc
    max_vis             = int(HOURS_PER_NIGHT * 3600 // cyc) if cyc > 0 else 0
    max_visits_map[hd]  = max_vis
    print(f"          cycle={cyc:.0f}s  max obs/night={max_vis}")

if not exposure_times:
    print("   No stars with valid exposure times. Exiting.")
    exit(1)


# 4. VISIT BUDGET

"""
Estimates how many observations fit in the available time budget,
using cycle time (exptime + overhead) per visit.
This number is passed to the scheduler as target_obs_per_star.
The budget is shared across all stars — total cycle time per round
is the sum of individual cycle times.
"""

total_cycle_per_round = sum(cycle_times[hd] for hd in exposure_times)
available_sec         = TARGET_NIGHTS * HOURS_PER_NIGHT * 3600
max_uniform_visits    = int(available_sec // total_cycle_per_round) \
                        if total_cycle_per_round > 0 else 0

print(f"\n4. Visit budget ({TARGET_NIGHTS} nights × {HOURS_PER_NIGHT} h/night)")
print(f"   Total cycle time per round (all stars): {total_cycle_per_round/3600:.2f} h")
print(f"   Max uniform visits per star: {max_uniform_visits}")


# 5. WEATHER

"""
Reads monthly clear-night fractions from the site weather file and
draws a random sample of usable nights for the year.
Set APPLY_WEATHER = False above to schedule on all calendar nights.
"""

weather_file = INSTRUMENT_CONFIG[INSTRUMENT]["weather_file"]
print(f"\n5. Weather ({'applied' if APPLY_WEATHER else 'disabled'})")

if APPLY_WEATHER:
    monthly_fractions = read_monthly_weather_stats(weather_file)
    potential_nights  = select_observing_nights_with_weather(monthly_fractions, START_YEAR)
    print(f"   {len(potential_nights)} usable nights sampled for {START_YEAR}")
else:
    potential_nights  = select_all_nights(START_YEAR)
    print(f"   Weather disabled — all {len(potential_nights)} calendar nights available")


# 6. OBSERVING SCHEDULE

"""
Generates observation timestamps for all stars by combining:
  - Observable window (airmass + altitude)
  - Weather model (if APPLY_WEATHER is True)
  - Cadence or even-spread spacing across observable days
  - Per-night visibility check: target_is_up + secz < 1.5
"""

print(f"\n6. Scheduling observations")

stars_to_schedule = list(exposure_times.keys())
dec_dict          = {hd: profiles[hd]['dec'] for hd in stars_to_schedule}
observatory_location = INSTRUMENT_CONFIG[INSTRUMENT]["observatory"]

schedule = generate_observing_schedule(
    selected_stars=stars_to_schedule,
    exposure_times=exposure_times,
    dec_dict=dec_dict,
    weather_file=weather_file,
    observatory_location=observatory_location,
    start_year=START_YEAR,
    num_years=1,
    target_obs_per_star=max_uniform_visits,
    cadence_days=CADENCE_DAYS,
    even_spread_flag=(CADENCE_DAYS is None),
    apply_weather_flag=APPLY_WEATHER,
    verbose=True,
)

if not schedule or all(len(v) == 0 for v in schedule.values()):
    print("   No observations scheduled. Exiting.")
    exit(1)


# 7. MULTI-INSTRUMENT STRATEGY  (optional)

"""
After scheduling, individual observations can be reassigned to a
second instrument according to the chosen strategy. The timestamps
themselves do not change — only the instrument label per visit.

  "none"             — all observations stay on the primary instrument
  "weather_fallback" — use secondary on months where primary site has
                       < 50% clear nights
  "alternate"        — interleave primary and secondary in time order

Skipped when STRATEGY = "none".
"""

instrument_assignments = None

if STRATEGY != "none":
    print(f"\n7. Multi-instrument strategy: {STRATEGY}")
    secondary_cfg = INSTRUMENT_CONFIG.get(SECONDARY_INSTRUMENT)
    if secondary_cfg is None:
        print(f"   Unknown secondary instrument '{SECONDARY_INSTRUMENT}' — skipping.")
    else:
        instrument_assignments = apply_multi_instrument_strategy(
            primary_obs=schedule,
            strategy=STRATEGY,
            primary_instrument=INSTRUMENT,
            secondary_instrument=SECONDARY_INSTRUMENT,
            primary_weather_file=INSTRUMENT_CONFIG[INSTRUMENT]["weather_file"],
            secondary_weather_file=secondary_cfg["weather_file"],
            start_year=START_YEAR,
            num_years=1,
        )
        print_strategy_summary(
            instrument_assignments,
            primary=INSTRUMENT,
            secondary=SECONDARY_INSTRUMENT,
            strategy=STRATEGY,
        )
else:
    print(f"\n7. Multi-instrument strategy: none (all observations on {INSTRUMENT})")


# 8. OUTPUT AND PLOTS

"""
For each star produces:
  - Observation timeline plot (dots coloured by instrument when strategy is active)
  - Synthetic stellar RV time series for one night
  - CSV file appended with: BJD_TDB, local time, RV, RV uncertainty
"""

print("\n8. Generating outputs")

output_file = os.path.join(script_dir, "output_timestamps.csv")
with open(output_file, "w") as f:
    f.write("HD_ID,BJD_TDB,LOCAL_TIME,RV,RV_ERR\n")

    for hd in stars_to_schedule:
        obs_times = schedule.get(hd, [])
        if not obs_times:
            print(f"   HD {hd}: no observations — skipping plots and CSV.")
            continue

        ra, dec = ra_dec_map.get(hd, (None, None))
        if ra is None or dec is None:
            print(f"   HD {hd}: RA/Dec unavailable — skipping RV outputs.")
            continue

        bjd, local_times, rv_vals, rv_errs = sample_stellar_rv_at_observations(
            obs_times, ra, dec, observatory_location
        )
        for i in range(len(bjd)):
            f.write(f"{hd},{bjd[i]:.8f},"
                    f"{local_times[i].strftime('%Y-%m-%d %H:%M:%S.%f')},"
                    f"{float(rv_vals[i]):.3f},"
                    f"{float(rv_errs[i]):.3f}\n")

        times_dt, rv_ts = generate_stellar_rv_timeseries(
            obs_times, teff=profiles[hd]['teff']
        )
        plot_stellar_rv(times_dt, rv_ts, f"HD {hd}", ra, dec, observatory_location)
        print(f"   HD {hd}: RV plot generated")

plot_observations_over_year(
    schedule,
    instrument_assignments=instrument_assignments,
    primary_instrument=INSTRUMENT,
    secondary_instrument=SECONDARY_INSTRUMENT if STRATEGY != "none" else None,
)

print(f"   Timeline plot generated")
print(f"   Timestamps written to output_timestamps.csv")

print("\nTutorial complete.")