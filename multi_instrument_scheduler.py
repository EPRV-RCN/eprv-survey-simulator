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

repo_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, repo_path)

from stellar_gp.argo_model import GPModel, GPData
from stellar_gp.argo_model import GranulationKernel,OscillationKernel,QPKernel,PerKernel,M52Kernel,M32Kernel,WNKernel, SEKernel,M52PDKernel
from stellar_gp.argo_model import generate_ts, get_stellar_kernels
from stellar_gp.stellar_scalings import get_stellar_hypers, calc_Pg


# USER INPUT / CONFIGURATION MODULE


# Primary instrument
selected_instrument = "NEID"   # "NEID" or "KPF"


# Secondary instrument for multi-instrument strategies.
# Only used when multi_instrument_strategy != "none".
selected_instrument_secondary = "KPF"   # "NEID" or "KPF" or None



multi_instrument_strategy = "weather_fallback" 
#   "none"             — single instrument, no reassignment
#   "weather_fallback" — keep primary unless its site weather is bad that
#                        night; in that case reassign to secondary
#   "alternate"        — alternate observations between primary and secondary
#                        in time order (even index → primary, odd → secondary)


# Weather toggle.
# True  — weather file drives which nights are scheduled (existing behaviour)
# False — ignore weather entirely; treat every night as clear
apply_weather = True

# Input stars with optional parameters
user_stars_input = [
    {'name': 'HD 166'},
]

# Paths
script_dir             = os.path.dirname(os.path.abspath(__file__))
hwo_file               = os.path.join(script_dir, "hwo_star_list_for_neid.xlsx")
output_timestamps_file = os.path.join(script_dir, "output_timestamps.csv")

# Scheduling parameters
cadence_days     = None   # None or integer days (overrides even_spread_flag)
even_spread_flag = True   # True = evenly spread observations across observable window

# Time parameters
start_year                   = 2025
num_years                    = 1
target_nights_for_uniformity = 20   # nights used in max-visit calculation
hours_per_night              = 10   # available observing hours per night

# Chaplin (2019) oscillation-averaging filter.
# True  → effective exptime = max(ETC_exptime, chaplin_exptime) per star.
# Chaplin time is a stellar property — same floor for all instruments.
# HWO stars have pre-computed values in column BI of hwo_star_list_for_neid.xlsx
# (10 cm/s threshold). Stars not in the HWO list fall back to ETC-only with a warning.
apply_chaplin_filter = True

# Instrument overhead constants.
# Cycle time per visit = exptime + acquisition + readout.
#   NEID: Time(sec) = V * (300 + T*N + 27*(N-1)), N=1 per visit
#         → acquisition = 300s (we split as 273s acq + 27s readout for clarity),
#           but per the formula for N=1: overhead = 300 + 27*(0) = 300s total.
#         We store as: acquisition 273s, readout 27s → total overhead 300s/visit.
NEID_ACQUISITION_SEC = 273   # acquisition time per visit (NEID, from WIYN call)
NEID_READOUT_SEC     = 27    # CCD readout per exposure (NEID, from WIYN call)
KPF_ACQUISITION_SEC  = 120   
KPF_READOUT_SEC      = 27    # PLACEHOLDER — update when KPF call is available

# Stellar variability configuration
stellar_kernel_type = "matern52"   # "matern52", "qp", "se", "granulation"
plot_duration_hours = 10
time_step_seconds   = 500


# INSTRUMENT REGISTRY

INSTRUMENT_CONFIG = {
    "NEID": {
        "etc":              None,   # set after functions are defined
        "observatory":      "kitt peak",
        "weather_file":     os.path.join(script_dir, "weather data", "KPNO.txt"),
        "rv_precision_ms":  0.5,
        "acquisition_sec":  NEID_ACQUISITION_SEC,
        "readout_sec":      NEID_READOUT_SEC,
    },
    "KPF": {
        "etc":              None,   # set after functions are defined
        "observatory":      "keck",
        "weather_file":     os.path.join(script_dir, "weather data", "Maunakea.txt"),
        "rv_precision_ms":  0.5,
        "acquisition_sec":  KPF_ACQUISITION_SEC,
        "readout_sec":      KPF_READOUT_SEC,
    },
}

def overhead_sec(instrument: str) -> int:
    """Total per-visit overhead = acquisition + readout (N=1 per visit)."""
    cfg = INSTRUMENT_CONFIG[instrument]
    return cfg["acquisition_sec"] + cfg["readout_sec"]

def cycle_sec(instrument: str, exptime: float) -> float:
    """Total time consumed per visit = exptime + overhead."""
    return exptime + overhead_sec(instrument)


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
    ra_val  = result[ra_col][0]
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
        cl     = col.lower()
        tokens = re.split(r'[^a-z0-9]+', cl)
        if ('v' in tokens or cl.endswith('_v') or cl.endswith('.v')) and \
           ('flux' in cl or 'mag' in cl or 'allfluxes' in cl):
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
            if num is not None and -5.0 < num < 40.0:
                if any(k in col.lower() for k in ('flux', 'mag', 'allfluxes', 'v', 'vmag')):
                    return num
    return None

def get_vsini(name):
    simbad = Simbad()
    simbad.add_votable_fields("mesRot")
    result = simbad.query_object(name)
    if result is None:
        return None
    possible_cols = [c for c in result.colnames if "mesrot.vsini" in c.lower()]
    if not possible_cols:
        return None
    vsini_col = possible_cols[0]
    bib_col   = "mesrot.bibcode" if "mesrot.bibcode" in result.colnames else None
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
            rows.append((int(match.group(1)), float(val)))
    if not rows:
        return None
    return max(rows, key=lambda x: x[0])[1]

def get_teff(name):
    simbad = Simbad()
    simbad.add_votable_fields("mesFe_h")
    result = simbad.query_object(name)
    if result is None:
        return None
    teff_col = bib_col = None
    for c in result.colnames:
        if "mesfe_h.teff"    in c.lower(): teff_col = c
        if "mesfe_h.bibcode" in c.lower(): bib_col  = c
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
            rows.append((int(match.group(1)), float(val)))
    if not rows:
        return None
    return max(rows, key=lambda x: x[0])[1]


# DATA PROCESSING FUNCTIONS

def query_star_data_astroquery(star_name):
    data = {'hd_id': None, 'dec': None, 'vsini': None, 'vmag': None, 'teff': None}
    try:
        name_match   = re.search(r'HD\s*(\d+[A-Za-z]*)', star_name, re.IGNORECASE)
        data['hd_id'] = name_match.group(1).strip() if name_match else star_name.replace(' ', '_')
        ra, dec = get_ra_dec(star_name)
        if dec  is not None: data['dec']   = dec
        vmag = get_vmag(star_name)
        if vmag is not None: data['vmag']  = vmag
        vsini = get_vsini(star_name)
        if vsini is not None: data['vsini'] = vsini
        teff = get_teff(star_name)
        if teff is not None: data['teff']  = teff
        return data
    except Exception as e:
        print(f"  SIMBAD query failed for {star_name}: {e}")
        return data

def load_hwo_candidates(filepath):
    """
    Loads stellar parameters from the HWO Excel catalogue.
    Returns (hd_set, teff_dict, vsini_dict, dec_dict, vmag_dict, chaplin_exptime_dict).
    Column indices: HD=4, Dec=11, Vmag=15, Teff=25, vsini=59, Chaplin exptime=60.
    """
    hd_set = set()
    teff_dict, vsini_dict, dec_dict, vmag_dict, chaplin_exptime_dict = {}, {}, {}, {}, {}
    try:
        df = pd.read_excel(filepath)
        for _, row in df.iterrows():
            raw_hd      = str(row[4]).strip()  if pd.notna(row[4])  else ""
            dec_str     = str(row[11]).strip() if pd.notna(row[11]) else ""
            vmag_str    = str(row[15]).strip() if pd.notna(row[15]) else ""
            teff_str    = str(row[25]).strip() if pd.notna(row[25]) else ""
            vsini_str   = str(row[59]).strip() if pd.notna(row[59]) else ""
            chaplin_str = str(row[60]).strip() if pd.notna(row[60]) else ""

            if not raw_hd.upper().startswith("HD "):
                continue
            hd = raw_hd[3:].strip()
            if not hd:
                continue
            hd_set.add(hd)

            def _float(s):
                try: return float(s)
                except: return None

            teff_dict[hd]    = _float(teff_str)
            vmag_dict[hd]    = round(_float(vmag_str), 2) if _float(vmag_str) is not None else None
            dec_dict[hd]     = _float(dec_str)
            chaplin_exptime_dict[hd] = _float(chaplin_str)

            vsini_val = _float(vsini_str)
            vsini_dict[hd] = round(vsini_val, 1) \
                if vsini_val is not None and vsini_str not in ("0", "nan", "") else None

    except IOError:
        print(f"  HWO file not found: {filepath}")
        return set(), {}, {}, {}, {}, {}
    except Exception as e:
        print(f"  Unexpected error loading HWO data: {e}")
        return hd_set, teff_dict, vsini_dict, dec_dict, vmag_dict, chaplin_exptime_dict

    return hd_set, teff_dict, vsini_dict, dec_dict, vmag_dict, chaplin_exptime_dict

def get_star_profile(star_input, hwo_fallback_data):
    """
    Builds a complete star profile via three-tier lookup:
      1. User-provided values
      2. SIMBAD query
      3. HWO Excel catalogue fallback
    Returns a profile dict or None if required fields cannot be filled.
    """
    hd_set, hwo_teff, hwo_vsini, hwo_dec, hwo_vmag, hwo_chaplin = hwo_fallback_data
    star_name = star_input.get('name')
    if not star_name:
        print("  Star input missing 'name'. Skipping.")
        return None

    profile = {
        'name':           star_name,
        'hd_id':          None,
        'teff':           star_input.get('teff'),
        'dec':            star_input.get('dec'),
        'vsini':          star_input.get('vsini'),
        'vmag':           star_input.get('vmag'),
        'chaplin_exptime': star_input.get('chaplin_exptime'),
    }

    if any(v is None for v in [profile['teff'], profile['dec'],
                                profile['vsini'], profile['vmag']]):
        sd = query_star_data_astroquery(star_name)
        if sd:
            if profile['hd_id']  is None and sd.get('hd_id'):  profile['hd_id']  = sd['hd_id']
            if profile['dec']    is None and sd.get('dec'):     profile['dec']    = sd['dec']
            if profile['vsini']  is None and sd.get('vsini'):   profile['vsini']  = sd['vsini']
            if profile['vmag']   is None and sd.get('vmag'):    profile['vmag']   = sd['vmag']
            if profile['teff']   is None and sd.get('teff'):    profile['teff']   = sd['teff']

    if profile['hd_id'] is None:
        m = re.search(r'HD\s*(\d+[A-Za-z]*)', star_name, re.IGNORECASE)
        profile['hd_id'] = m.group(1).strip() if m else None

    if profile['hd_id'] and profile['hd_id'] in hd_set:
        hd = profile['hd_id']
        if profile['teff']           is None: profile['teff']           = hwo_teff.get(hd)
        if profile['dec']            is None: profile['dec']            = hwo_dec.get(hd)
        if profile['vsini']          is None: profile['vsini']          = hwo_vsini.get(hd)
        if profile['vmag']           is None: profile['vmag']           = hwo_vmag.get(hd)
        if profile['chaplin_exptime'] is None: profile['chaplin_exptime'] = hwo_chaplin.get(hd)

    missing = [k for k in ['hd_id','teff','dec','vsini','vmag'] if profile.get(k) is None]
    if missing:
        print(f"  Incomplete profile for {star_name} — missing: {', '.join(missing)}")
        return None

    return profile


# SCHEDULING FUNCTIONS

def observable_window(star_name=None, ra=None, dec=None,
                      observatory="Kitt Peak", airmass_limit=1.5,
                      min_hours=2.0, year=2025, plot=False):
    """
    Returns days of the year when the target is above airmass_limit
    for at least min_hours during astronomical night.
    Returns None if no such days exist.
    """
    location = EarthLocation.of_site(observatory)
    observer = Observer(location=location, name=observatory)

    if star_name is not None:
        target = FixedTarget.from_name(star_name)
    elif ra is not None and dec is not None:
        target = FixedTarget(coord=SkyCoord(ra*u.deg, dec*u.deg), name="CustomTarget")
    else:
        raise ValueError("Provide either star_name or (ra, dec).")

    altitude_limit  = np.degrees(np.arcsin(1 / airmass_limit))
    days            = np.arange(1, 367)
    observable_hours = []

    for day in days:
        t_mid = Time(f"{year}-01-01 00:00:00") + (day - 1) * u.day
        mid   = observer.midnight(t_mid, which="nearest")
        try:
            eve  = observer.twilight_evening_astronomical(mid, which="nearest")
            morn = observer.twilight_morning_astronomical(mid, which="next")
        except Exception:
            observable_hours.append(0.0)
            continue
        dt      = np.linspace(0, (morn - eve).to(u.hour).value, 200) * u.hour
        alts    = observer.altaz(eve + dt, target).alt.deg
        hours   = np.sum(alts >= altitude_limit) * (dt[1] - dt[0]).to(u.hour).value
        observable_hours.append(hours)

    observable_hours = np.array(observable_hours)

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(days, observable_hours, color="darkblue")
        plt.fill_between(days, 0, observable_hours, alpha=0.3, color="skyblue")
        plt.xlabel("Day of Year")
        plt.ylabel("Observable Hours")
        plt.title(f"Observable Hours of {target.name} from {observatory} ({year})")
        plt.grid(alpha=0.4)
        plt.show()

    good_days = days[observable_hours >= min_hours]
    return good_days if len(good_days) > 0 else None

def generate_observation_timestamps(stars, dec_dict, n_observations_per_star,
                                     start_day=0, end_day=365,
                                     cadence_days=None, even_spread=False,
                                     seed=42, observatory_name="Kitt Peak", year=2025):
    np.random.seed(seed)
    random.seed(seed)
    observation_timestamps = {}

    for star in stars:
        n_obs = n_observations_per_star.get(star, 0)
        if n_obs <= 0:
            continue

        obs_days = None
        if dec_dict.get(star) is not None:
            try:
                obs_days = observable_window(
                    star_name=f"HD {star}", observatory=observatory_name,
                    airmass_limit=1.5, min_hours=2.0, year=year, plot=False
                )
            except Exception as e:
                print(f"  ERROR: observable window failed for HD {star}: {e}")
                return None

        if obs_days is None or len(obs_days) == 0:
            print(f"  ERROR: No observable window found for HD {star}.")
            return None

        if cadence_days is not None:
            raw  = list(range(start_day, end_day, cadence_days))
            days = [d for d in raw if d in obs_days][:n_obs]
        elif even_spread:
            if n_obs > 1:
                idx  = np.round(np.linspace(0, len(obs_days) - 1, n_obs)).astype(int)
                days = list(obs_days[idx])
            elif n_obs == 1:
                days = [obs_days[len(obs_days) // 2]]
            else:
                days = []
        else:
            days = sorted(np.random.choice(obs_days, size=n_obs, replace=False))

        observation_timestamps[star] = days

    return observation_timestamps


# GENERAL SCHEDULING FUNCTIONS

def calculate_exposures_and_uniform_visits(
        stars_to_process, teff_dict, vmag_dict, vsini_dict,
        exposure_time_calculator, instrument_name,
        target_nights_for_uniformity=20, hours_per_night=10,
        chaplin_exptime_dict=None, apply_chaplin_filter=False):
    """
    Computes per-star exposure times and the maximum uniform visit count.

    Cycle time per visit = exptime + acquisition overhead + readout
    (instrument-specific values from INSTRUMENT_CONFIG).

    Returns (valid_stars, exposure_times, max_uniform_visits).
    """
    acq      = INSTRUMENT_CONFIG[instrument_name]["acquisition_sec"]
    readout  = INSTRUMENT_CONFIG[instrument_name]["readout_sec"]
    overhead = acq + readout

    exposure_times           = {}
    total_cycle_sec          = 0
    valid_selected_stars     = []

    print(f"\n  {'Star':<12}  {'Teff':>6}  {'Vmag':>5}  {'vsini':>6}  "
          f"{'ETC (s)':>8}", end="")
    if apply_chaplin_filter:
        print(f"  {'Chaplin (s)':>11}  {'Binding':>8}", end="")
    print(f"  {'Final (s)':>9}  {'Cycle (s)':>9}")

    for hd in stars_to_process:
        teff  = teff_dict.get(hd)
        vmag  = vmag_dict.get(hd)
        vsini = vsini_dict.get(hd)

        if any(v is None for v in [teff, vmag, vsini]):
            print(f"  HD {hd}: missing Teff/Vmag/vsini — skipping")
            continue

        etc_exptime = exposure_time_calculator(teff, vmag, vsini)
        if np.isnan(etc_exptime):
            print(f"  HD {hd}: ETC returned NaN — skipping")
            continue

        if apply_chaplin_filter:
            chaplin_time = chaplin_exptime_dict.get(hd) if chaplin_exptime_dict else None
            if chaplin_time is None:
                print(f"  WARNING: Chaplin filter ON but no value for HD {hd} — using ETC only.")
                exptime  = etc_exptime
                binding  = "ETC"
            else:
                exptime  = max(etc_exptime, chaplin_time)
                binding  = "CHAPLIN" if chaplin_time >= etc_exptime else "ETC"
            chap_str = f"  {chaplin_time if chaplin_time else 'N/A':>11}  {binding:>8}"
        else:
            exptime  = etc_exptime
            chap_str = ""

        cyc  = exptime + overhead
        maxv = int(hours_per_night * 3600 // cyc) if cyc > 0 else 0

        print(f"  HD {hd:<8}  {teff:>6.0f}  {vmag:>5.2f}  {vsini:>6.1f}  "
              f"{etc_exptime:>8.0f}{chap_str}  {exptime:>9.0f}  {cyc:>9.0f}  {maxv:>9}")

        exposure_times[hd] = exptime
        total_cycle_sec   += cyc
        valid_selected_stars.append(hd)

    total_available = target_nights_for_uniformity * hours_per_night * 3600
    max_uniform     = int(total_available // total_cycle_sec) if total_cycle_sec > 0 else 0

    print(f"\n  Overhead per visit ({instrument_name}): "
          f"{acq}s acquisition + {readout}s readout = {overhead}s total")
    print(f"  Total cycle time per visit (all stars): {total_cycle_sec / 3600:.2f} h")
    print(f"  Max uniform visits across {target_nights_for_uniformity} nights: {max_uniform}")

    return valid_selected_stars, exposure_times, max_uniform

def read_monthly_weather_stats(weather_file):
    try:
        with open(weather_file, 'r') as f:
            fracs = [float(l.strip()) for l in f if l.strip()]
        if len(fracs) != 12:
            raise ValueError("Weather file must have exactly 12 monthly fractions.")
        return fracs
    except IOError:
        print(f"  Weather file not found: {weather_file}")
        return []
    except Exception as e:
        print(f"  Error reading weather file: {e}")
        return []

def select_observing_nights_with_weather(monthly_fractions, start_year, num_years=1):
    dates = []
    for yr_off in range(num_years):
        yr = start_year + yr_off
        for mi, frac in enumerate(monthly_fractions):
            for d in range(1, calendar.monthrange(yr, mi + 1)[1] + 1):
                if random.random() <= frac:
                    dates.append(Time(f"{yr}-{mi+1:02d}-{d:02d} 00:00:00",
                                      format='iso', scale='utc'))
    return sorted(dates)

def select_all_nights(start_year, num_years=1):
    """All calendar nights — used when apply_weather is False."""
    dates = []
    for yr_off in range(num_years):
        yr = start_year + yr_off
        for mi in range(12):
            for d in range(1, calendar.monthrange(yr, mi + 1)[1] + 1):
                dates.append(Time(f"{yr}-{mi+1:02d}-{d:02d} 00:00:00",
                                  format='iso', scale='utc'))
    return sorted(dates)

def generate_observing_schedule(selected_stars, exposure_times, dec_dict, weather_file,
                                observatory_location='kitt peak',
                                start_year=2025, num_years=1,
                                clear_night_threshold=0.6,
                                target_obs_per_star=None,
                                total_sim_nights_limit=None,
                                cadence_days=None,
                                even_spread_flag=False,
                                apply_weather_flag=True,
                                verbose=True):
    """
    Schedules observations for selected_stars.

    apply_weather_flag: if False, ignores weather and schedules on all nights.
    """
    location = Observer.at_site(observatory_location)

    if apply_weather_flag:
        monthly_fracs = read_monthly_weather_stats(weather_file)
        potential_dates = select_observing_nights_with_weather(
            monthly_fracs, start_year, num_years)
    else:
        potential_dates = select_all_nights(start_year, num_years)

    stars_for_cadence = {hd: target_obs_per_star for hd in selected_stars}
    desired_obs_days  = generate_observation_timestamps(
        selected_stars, dec_dict, stars_for_cadence,
        start_day=0, end_day=365 * num_years,
        cadence_days=cadence_days, even_spread=even_spread_flag,
        observatory_name=location.name if hasattr(location, "name") and location.name
                         else "Kitt Peak",
        year=start_year
    )
    if desired_obs_days is None:
        desired_obs_days = {}

    star_observations  = {hd: [] for hd in selected_stars}
    observations_count = {hd: 0   for hd in selected_stars}
    processed_nights   = 0

    for obs_date in potential_dates:
        if (total_sim_nights_limit is not None
                and processed_nights >= total_sim_nights_limit):
            if verbose:
                print(f"  Reached night limit ({total_sim_nights_limit})")
            break
        if (target_obs_per_star is not None
                and all(c >= target_obs_per_star for c in observations_count.values())):
            if verbose:
                print("  All stars reached target observation count")
            break

        processed_nights += 1
        doy = obs_date.datetime.timetuple().tm_yday

        try:
            night_start = location.twilight_evening_nautical(obs_date, which='next')
            night_end   = location.twilight_morning_nautical(night_start, which='next')
            if night_end < night_start:
                night_end += 1 * u.day
            actual_end = night_start + TimeDelta(10 * u.hour)
            if actual_end > night_end:
                actual_end = night_end
            if actual_end <= night_start:
                continue
            t0_mjd = night_start.mjd
            t1_mjd = actual_end.mjd
        except Exception:
            continue

        to_observe = []
        for hd in selected_stars:
            if doy not in desired_obs_days.get(hd, []):
                continue
            if target_obs_per_star is not None and observations_count[hd] >= target_obs_per_star:
                continue
            target = FixedTarget.from_name(f"HD {hd}")
            visible = False
            for t_mjd in np.linspace(t0_mjd, t1_mjd, 5):
                tp = Time(t_mjd, format='mjd')
                if location.target_is_up(tp, target) and \
                   location.altaz(tp, target).secz < 1.5:
                    visible = True
                    break
            if visible:
                to_observe.append(hd)

        random.shuffle(to_observe)
        for hd in to_observe:
            if target_obs_per_star is None or observations_count[hd] < target_obs_per_star:
                if exposure_times.get(hd) is None:
                    continue
                star_observations[hd].append(
                    night_start + TimeDelta(
                        random.uniform(0, t1_mjd - t0_mjd) * u.day
                    )
                )
                observations_count[hd] += 1

    if verbose:
        print(f"\n  Results:")
        total = 0
        for hd, obs_list in star_observations.items():
            print(f"    HD {hd}: {len(obs_list)} observations")
            total += len(obs_list)
        print(f"    Total: {total} observations")
        print(f"    Potential nights: {len(potential_dates)}")
        print(f"    Nights processed: {processed_nights}")

    return star_observations


# MULTI-INSTRUMENT STRATEGY LAYER

def apply_multi_instrument_strategy(
        primary_obs: dict,
        strategy: str,
        primary_instrument: str,
        secondary_instrument: str,
        primary_weather_file: str,
        secondary_weather_file: str,
        start_year: int,
        num_years: int) -> dict:
    """
    Takes {hd_id: [Time, ...]} from the primary scheduler and returns
    {hd_id: [(Time, instrument_name), ...]} with instruments assigned
    according to the chosen strategy.

    Strategies:
      "none"             — all observations stay on primary instrument
      "weather_fallback" — reassign to secondary on nights where
                           primary site weather fraction < 0.5
      "alternate"        — alternate primary/secondary in time order
    """
    if strategy == "none" or secondary_instrument is None:
        return {hd: [(t, primary_instrument) for t in times]
                for hd, times in primary_obs.items()}

    # Build a set of "bad primary nights" from monthly weather fractions
    bad_primary_nights = set()
    if strategy == "weather_fallback":
        fracs = read_monthly_weather_stats(primary_weather_file)
        if fracs:
            for yr_off in range(num_years):
                yr = start_year + yr_off
                for mi, frac in enumerate(fracs):
                    if frac < 0.5:   # month has worse than 50% clear nights → flag
                        for d in range(1, calendar.monthrange(yr, mi + 1)[1] + 1):
                            bad_primary_nights.add(
                                f"{yr}-{mi+1:02d}-{d:02d}"
                            )

    result = {}
    for hd, times in primary_obs.items():
        sorted_times = sorted(times, key=lambda t: t.mjd)
        assigned     = []
        for idx, t in enumerate(sorted_times):
            if strategy == "weather_fallback":
                date_key = t.datetime.strftime("%Y-%m-%d")
                inst     = secondary_instrument if date_key in bad_primary_nights \
                           else primary_instrument
            elif strategy == "alternate":
                inst = primary_instrument if idx % 2 == 0 else secondary_instrument
            else:
                inst = primary_instrument
            assigned.append((t, inst))
        result[hd] = assigned

    return result

def print_strategy_summary(assigned_obs: dict, primary: str, secondary: str,
                            strategy: str):
    """Print a clean summary of instrument assignments after strategy is applied."""
    if strategy == "none":
        return

    all_pairs = [(t, inst) for pairs in assigned_obs.values() for t, inst in pairs]
    total     = len(all_pairs)
    n_primary = sum(1 for _, inst in all_pairs if inst == primary)
    n_second  = sum(1 for _, inst in all_pairs if inst == secondary)

    print(f"\n  Multi-instrument strategy: {strategy}")
    print(f"    {primary:<8} {n_primary:>5} obs  ({100*n_primary/total:.0f}%)")
    if secondary:
        print(f"    {secondary:<8} {n_second:>5} obs  ({100*n_second/total:.0f}%)")


# VISUALIZATION FUNCTIONS

def plot_observations_over_year(simulated_observations_data, instrument_assignments=None,
                                 primary_instrument=None, secondary_instrument=None):
    """
    Scatter plot of observation timestamps over the year.
    If instrument_assignments is provided, dots are coloured by instrument.
    """
    if not simulated_observations_data:
        print("  No observation data to plot.")
        return

    def get_hd_number(s):
        m = re.match(r'(\d+)', s)
        return int(m.group(1)) if m else float('-inf')

    sorted_stars = sorted(simulated_observations_data.keys(),
                          key=get_hd_number)
    star_to_y    = {s: i for i, s in enumerate(sorted_stars)}

    inst_colours = {
        primary_instrument:   "#3A86FF",
        secondary_instrument: "#FF6B35",
    }

    fig, ax = plt.subplots(figsize=(13, max(4, 2 * len(sorted_stars))))

    for star in sorted_stars:
        times = simulated_observations_data[star]
        if instrument_assignments and star in instrument_assignments:
            for t, inst in instrument_assignments[star]:
                ax.scatter(t.datetime.timetuple().tm_yday, star_to_y[star],
                           s=18, alpha=0.6, linewidths=0,
                           color=inst_colours.get(inst, '#aaa'))
        else:
            for t in times:
                ax.scatter(t.datetime.timetuple().tm_yday, star_to_y[star],
                           s=18, alpha=0.6, linewidths=0, color="#3A86FF")

    ax.set_yticks(range(len(sorted_stars)))
    ax.set_yticklabels([f"HD {s}" for s in sorted_stars])
    ax.set_xlabel("Day of Year")
    ax.set_xlim(1, 366)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', color='#e8e8e8', linewidth=0.6)

    if instrument_assignments and primary_instrument and secondary_instrument:
        import matplotlib.patches as mpatches
        patches = [mpatches.Patch(color=inst_colours[primary_instrument],
                                   label=primary_instrument),
                   mpatches.Patch(color=inst_colours[secondary_instrument],
                                   label=secondary_instrument)]
        ax.legend(handles=patches, fontsize=9, frameon=False)

    ax.set_title("Observation Schedule", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_stellar_rv(times_dt, rv_ts, star_name, ra, dec, observatory_location):
    if times_dt is None or rv_ts is None:
        return

    import matplotlib.dates as mdates

    obs_key = str(observatory_location).lower()
    if obs_key in ["keck", "maunakea", "mauna kea"]:
        local_tz = pytz.timezone("Pacific/Honolulu")
        location = EarthLocation.of_site("Keck Observatory")
    elif obs_key in ["kitt peak", "kpno"]:
        local_tz = pytz.timezone("US/Arizona")
        location = EarthLocation.of_site("Kitt Peak")
    else:
        local_tz = pytz.utc
        location = EarthLocation.of_site(observatory_location)

    t_utc   = Time(times_dt.to_pydatetime(), scale="utc")
    t_local = t_utc.to_datetime(timezone=local_tz)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t_local, rv_ts, color="purple")
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.set_xlabel(f"Local Time ({local_tz.zone})")
    ax.set_ylabel("RV (m/s)")
    ax.set_title(f"Synthetic Stellar RV — {star_name}  ({stellar_kernel_type})", pad=22)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%#m-%#d %#H:%M", tz=local_tz))

    tick_locs    = ax.get_xticks()[::2]
    tick_dt_utc  = [mdates.num2date(tl, tz=local_tz).astimezone(pytz.utc)
                    for tl in tick_locs]
    t_ticks      = Time(tick_dt_utc, scale="utc")
    target_coord = SkyCoord(ra * u.deg, dec * u.deg)
    bjd_ticks    = (t_ticks.tdb + t_ticks.light_travel_time(target_coord,
                                                               location=location)).jd

    ax_bjd = ax.secondary_xaxis("bottom")
    ax_bjd.spines["bottom"].set_position(("outward", 18))
    ax_bjd.set_xticks(tick_locs)
    ax_bjd.set_xticklabels([f"{bjd:.2f}" for bjd in bjd_ticks])
    ax_bjd.set_xlabel("BJD_TDB")

    plt.tight_layout()
    plt.show()


# NEID INSTRUMENT MODULE

def apply_neid_vsini_condition(vsini_value):
    return max(vsini_value, 1.0) if vsini_value is not None and vsini_value < 1.0 else vsini_value

def vsini_scaling(vsini=2.0):
    return (0.000103 * vsini**4 - 0.004042 * vsini**3
            + 0.048354 * vsini**2 - 0.014283 * vsini + 0.868)

def NEID_exptime_RV(teff, vmag, rv_precision, seeing=0.8, vsini=2.0,
                    use_order=False, order=0):
    neid_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "etc_neid")
    exptime_grid = fits.open(os.path.join(neid_path,'photon_grid_exptime.fits'))[0].data
    teff_grid    = fits.open(os.path.join(neid_path,'photon_grid_teff.fits'))[0].data
    vmag_grid    = fits.open(os.path.join(neid_path,'photon_grid_vmag.fits'))[0].data
    seeing_grid  = np.array([0.3,0.5,0.7,0.8,0.9,1.1,1.3,1.5,1.7,1.9])
    logexp       = np.log10(exptime_grid)

    if not (np.min(teff_grid) <= teff <= np.max(teff_grid)):
        print(f"  NEID: Teff {teff:.0f} out of bounds"); return np.nan
    if not (np.min(vmag_grid) <= vmag <= np.max(vmag_grid)):
        print(f"  NEID: Vmag {vmag:.2f} out of bounds"); return np.nan
    if not (0.3 <= seeing <= 1.9):
        print(f"  NEID: seeing {seeing} out of bounds"); return np.nan

    if use_order:
        order_grid = fits.open(os.path.join(neid_path,'order_wvl_centers.fits'))[0].data[0]
        ol = np.where(order_grid == order)[0][0]
        rvprec_grid = np.array([
            fits.open(os.path.join(neid_path,'dv_uncertainty_master_order_seeing',
                      f'dv_uncertainty_master_order_{s}.fits'))[0].data[ol]
            for s in seeing_grid])
    else:
        rvprec_grid = np.array([
            fits.open(os.path.join(neid_path,'dv_uncertainty_master_seeing',
                      f'dv_uncertainty_master_{s}.fits'))[0].data
            for s in seeing_grid])

    ti = InterpolatedUnivariateSpline(teff_grid, np.arange(len(teff_grid), dtype=float))(teff)
    vi = InterpolatedUnivariateSpline(vmag_grid, np.arange(len(vmag_grid), dtype=float))(vmag)
    si = InterpolatedUnivariateSpline(seeing_grid, np.arange(len(seeing_grid), dtype=float))(seeing)

    j = 0
    eta = 1e10
    while eta > rv_precision:
        exptime = 2 * (j + 6)
        if exptime > np.max(exptime_grid):
            return np.nan
        ei  = InterpolatedUnivariateSpline(logexp, np.arange(len(exptime_grid), dtype=float))(np.log10(exptime))
        interp = RegularGridInterpolator(
            (np.arange(len(seeing_grid)), np.arange(len(exptime_grid)),
             np.arange(len(vmag_grid)),  np.arange(len(teff_grid))),
            rvprec_grid)
        eta = interp([si, ei, vi, ti])[0] * vsini_scaling(vsini)
        j  += 1
    return exptime

def neid_exposure_time_calculator(teff, vmag, vsini):
    vsini = apply_neid_vsini_condition(vsini)
    return NEID_exptime_RV(teff, vmag, 0.5, seeing=1.0, vsini=vsini)


# KPF INSTRUMENT MODULE

def kpf_exposure_time_calculator(teff, vmag, vsini):
    return kpf_etc_rv(teff, vmag, 0.5)

def kpf_etc_rv(teff, vmag, sigma_rv):
    kpf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "etc_kpf")
    sigma_rv_grid = fits.getdata(os.path.join(kpf_path, 'dv_uncertainty_master.fits'))
    teff_grid     = fits.getdata(os.path.join(kpf_path, 'photon_grid_teff.fits'))
    vmag_grid     = fits.getdata(os.path.join(kpf_path, 'photon_grid_vmag.fits'))
    exptime_grid  = fits.getdata(os.path.join(kpf_path, 'photon_grid_exptime.fits'))
    logexp        = np.log10(exptime_grid)

    if not (np.min(teff_grid) <= teff <= np.max(teff_grid)):
        print(f"  KPF: Teff {teff:.0f} out of bounds"); return np.nan
    if not (np.min(vmag_grid) <= vmag <= np.max(vmag_grid)):
        print(f"  KPF: Vmag {vmag:.2f} out of bounds"); return np.nan

    ti = InterpolatedUnivariateSpline(teff_grid, np.arange(len(teff_grid), dtype=float))(teff)
    vi = InterpolatedUnivariateSpline(vmag_grid, np.arange(len(vmag_grid), dtype=float))(vmag)

    ind = 2
    maxout = 1e10
    while maxout > sigma_rv:
        trial = min(exptime_grid) + ind
        ei    = InterpolatedUnivariateSpline(logexp, np.arange(len(exptime_grid), dtype=float))(np.log10(trial))
        interp = RegularGridInterpolator(
            (np.arange(len(exptime_grid)), np.arange(len(vmag_grid)), np.arange(len(teff_grid))),
            sigma_rv_grid)
        maxout = interp([ei, vi, ti])[0]
        ind   += 1
    return trial


# Register ETCs now that the functions are defined
INSTRUMENT_CONFIG["NEID"]["etc"] = neid_exposure_time_calculator
INSTRUMENT_CONFIG["KPF"]["etc"]  = kpf_exposure_time_calculator


# STELLAR GP KERNELS

def build_stellar_kernel(kernel_type, teff=None):
    if kernel_type == "granulation":
        logg_sun = 4.43
        args     = (logg_sun, teff) if teff is not None else ()
        kosc, kgran1, kgran2 = get_stellar_kernels(*args)
        return [kosc, kgran1, kgran2]
    if kernel_type == "matern52": return M52PDKernel(0.5, 0.5, 0.05)
    if kernel_type == "qp":       return QPKernel(1.0, 10.0, 0.5, 0.5)
    if kernel_type == "se":       return SEKernel(1.0, 0.5)
    raise ValueError(f"Unknown kernel type: {kernel_type}")

def sample_stellar_rv_at_observations(obs_times, ra, dec, observatory_location):
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

    t_utc      = Time([t.to_datetime() for t in obs_times], scale="utc")
    t0         = t_utc[0]
    t_days     = (t_utc - t0).to(u.day).value
    dur_days   = np.full_like(t_days, 500.0 / 86400.0)
    rv         = np.zeros_like(t_days)
    rv_err     = np.full_like(t_days, 0.5)

    kernels = build_stellar_kernel(stellar_kernel_type)
    model   = GPModel(kernels=kernels if isinstance(kernels, list) else [kernels])
    data    = GPData(time=t_days, duration=dur_days, rv=rv, rv_err=rv_err)
    rv_samp = generate_ts(model, data, ignore_errs=True)

    target     = SkyCoord(ra * u.deg, dec * u.deg)
    bjd        = (t_utc.tdb + t_utc.light_travel_time(target, location=location)).jd
    local_times = t_utc.to_datetime(timezone=local_tz)
    return bjd, local_times, rv_samp, rv_err

def generate_stellar_rv_timeseries(timestamps, teff=None, obs_duration_sec=500.0):
    if len(timestamps) == 0:
        return None, None
    t0      = timestamps[0].to_datetime()
    n_pts   = int(plot_duration_hours * 3600 / time_step_seconds)
    times_dt = pd.date_range(start=t0, periods=n_pts, freq=f"{time_step_seconds}s")
    t_sec   = (times_dt - t0).total_seconds().to_numpy()
    dur_sec = np.full_like(t_sec, float(obs_duration_sec))
    rv      = np.zeros_like(t_sec)
    rv_err  = np.full_like(t_sec, 0.5)
    kernels = build_stellar_kernel(stellar_kernel_type, teff=teff)

    if stellar_kernel_type == "granulation":
        data  = GPData(time=t_sec, duration=dur_sec, rv=rv, rv_err=rv_err)
        model = GPModel(kernels=kernels)
    else:
        t_days   = t_sec / 86400.0
        dur_days = dur_sec / 86400.0
        data     = GPData(time=t_days, duration=dur_days, rv=rv, rv_err=rv_err)
        model    = GPModel(kernels=[kernels])

    rv_ts = generate_ts(model, data, ignore_errs=True)
    return times_dt, rv_ts


# MAIN EXECUTION

if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    # Resolve primary instrument config
    if selected_instrument not in INSTRUMENT_CONFIG:
        raise ValueError(f"Unknown instrument: {selected_instrument}")
    primary_cfg = INSTRUMENT_CONFIG[selected_instrument]
    exposure_time_calculator = primary_cfg["etc"]
    observatory_location     = primary_cfg["observatory"]
    weather_file             = primary_cfg["weather_file"]

    # Resolve secondary instrument config (if used)
    secondary_cfg = None
    if multi_instrument_strategy != "none" and selected_instrument_secondary:
        if selected_instrument_secondary not in INSTRUMENT_CONFIG:
            raise ValueError(f"Unknown secondary instrument: {selected_instrument_secondary}")
        secondary_cfg = INSTRUMENT_CONFIG[selected_instrument_secondary]

    # Load HWO catalogue
    hwo_fallback_data = load_hwo_candidates(hwo_file)

    # Build star profiles
    selected_stars_list = []
    final_teff_dict     = {}
    final_vmag_dict     = {}
    final_vsini_dict    = {}
    final_dec_dict      = {}
    final_chaplin_dict  = {}

    for star_input in user_stars_input:
        profile = get_star_profile(star_input, hwo_fallback_data)
        if profile:
            hd = profile['hd_id']
            selected_stars_list.append(hd)
            final_teff_dict[hd]    = profile['teff']
            final_vmag_dict[hd]    = profile['vmag']
            final_vsini_dict[hd]   = profile['vsini']
            final_dec_dict[hd]     = profile['dec']
            final_chaplin_dict[hd] = profile.get('chaplin_exptime')

    # Exposure times and visit budget
    weather_label = "on" if apply_weather else "off (all nights treated as clear)"
    chaplin_label = "on" if apply_chaplin_filter else "off"

    print(f"\nInstrument:            {selected_instrument}")
    print(f"Observatory:           {observatory_location}")
    print(f"Weather:               {weather_label}")
    print(f"Chaplin filter:        {chaplin_label}")
    print(f"Multi-inst strategy:   {multi_instrument_strategy}")
    if multi_instrument_strategy != "none" and selected_instrument_secondary:
        print(f"Secondary instrument:  {selected_instrument_secondary}")

    print(f"\nExposure times and nightly cadence:")
    selected_stars, exposure_times, max_uniform_visits = \
        calculate_exposures_and_uniform_visits(
            selected_stars_list,
            final_teff_dict, final_vmag_dict, final_vsini_dict,
            exposure_time_calculator,
            instrument_name=selected_instrument,
            target_nights_for_uniformity=target_nights_for_uniformity,
            hours_per_night=hours_per_night,
            chaplin_exptime_dict=final_chaplin_dict,
            apply_chaplin_filter=apply_chaplin_filter,
        )

    # Schedule observations
    print(f"\nScheduling observations...")
    simulated_observations_data = generate_observing_schedule(
        selected_stars,
        exposure_times,
        final_dec_dict,
        weather_file,
        observatory_location=observatory_location,
        start_year=start_year,
        num_years=num_years,
        clear_night_threshold=0.6,
        target_obs_per_star=max_uniform_visits,
        cadence_days=cadence_days,
        even_spread_flag=even_spread_flag,
        apply_weather_flag=apply_weather,
        verbose=True,
    )

    # Apply multi-instrument strategy (post-scheduling instrument assignment)
    instrument_assignments = None
    if multi_instrument_strategy != "none" and selected_instrument_secondary:
        print(f"\nApplying multi-instrument strategy: {multi_instrument_strategy}")
        instrument_assignments = apply_multi_instrument_strategy(
            primary_obs=simulated_observations_data,
            strategy=multi_instrument_strategy,
            primary_instrument=selected_instrument,
            secondary_instrument=selected_instrument_secondary,
            primary_weather_file=primary_cfg["weather_file"],
            secondary_weather_file=secondary_cfg["weather_file"],
            start_year=start_year,
            num_years=num_years,
        )
        print_strategy_summary(
            instrument_assignments,
            primary=selected_instrument,
            secondary=selected_instrument_secondary,
            strategy=multi_instrument_strategy,
        )

    # Write output timestamps
    with open(output_timestamps_file, "w") as f:
        f.write("BJD_TDB,LOCAL_TIME,RV,RV_ERR\n")
        for hd, obs_times in simulated_observations_data.items():
            ra, dec = get_ra_dec(f"HD {hd}")
            bjd, local_times, rv_vals, rv_errs = sample_stellar_rv_at_observations(
                obs_times, ra, dec, observatory_location
            )
            for i in range(len(bjd)):
                f.write(f"{bjd[i]:.8f},"
                        f"{local_times[i].strftime('%Y-%m-%d %H:%M:%S.%f')},"
                        f"{float(rv_vals[i]):.3f},"
                        f"{float(rv_errs[i]):.3f}\n")

    # Plots
    plot_observations_over_year(
        simulated_observations_data,
        instrument_assignments=instrument_assignments,
        primary_instrument=selected_instrument,
        secondary_instrument=selected_instrument_secondary,
    )

    for hd, obs_times in simulated_observations_data.items():
        ra, dec   = get_ra_dec(f"HD {hd}")
        times_dt, rv_ts = generate_stellar_rv_timeseries(obs_times)
        plot_stellar_rv(times_dt, rv_ts, f"HD {hd}", ra, dec, observatory_location)

    print("\nDone.")