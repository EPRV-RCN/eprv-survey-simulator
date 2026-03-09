"""
multi_instrument_scheduler.py

(Imports all existing functions from the main simulator)
"""

from simulator import *   # get_star_profile, calculate_exposures_and_uniform_visits,
                           # read_monthly_weather_stats, generate_observing_schedule,
                           # observable_window, EarthLocation, Observer, FixedTarget,
                           # SkyCoord, Time, u, np, plt, os, calendar, datetime,
                           # random, warnings, script_dir, hwo_file, load_hwo_candidates

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker


# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Edit to change stars or add instruments.


MULTI_INSTRUMENT_STARS = [
    {'name': 'HD 217987'},   # M2V,  Dec -36
    {'name': 'HD 166'},      # K0V,  Dec +29
    {'name': 'HD 142860'},   # G2V,  Dec +27
]


INSTRUMENTS: Dict[str, dict] = {
    "NEID": {
        "etc":          neid_exposure_time_calculator,   # from simulator.*
        "site":         "kitt peak",
        "weather_file": os.path.join(script_dir, "KPNO.txt"),
        "rv_precision": 0.5,   # m/s
    },
    "KPF": {
        "etc":          kpf_exposure_time_calculator,    # from simulator.*
        "site":         "keck",
        "weather_file": os.path.join(script_dir, "Maunakea.txt"),
        "rv_precision": 0.5,
    },
}

# Scheduling constants
START_YEAR        = 2025
NUM_YEARS         = 1
HOURS_PER_NIGHT   = 10
OVERHEAD_SEC      = 120      # per-observation overhead: readout + slew
AIRMASS_LIMIT     = 1.5
N_ALT_POINTS      = 100      # altitude grid resolution for obs-hours calculation


# ── DATA CLASSES ──────────────────────────────────────────────────────────────

class WeatherStatus(Enum):
    CLEAR   = "clear"
    PARTIAL = "partial"
    CLOSED  = "closed"


@dataclass
class ScheduledObservation:
    hd_id:        str
    instrument:   str
    obs_time:     Time
    exposure_sec: float
    strategy:     str


@dataclass
class ScheduleResult:
    strategy_name: str
    label:         str                              # short label for plots
    observations:  List[ScheduledObservation] = field(default_factory=list)

    def count_by_star(self) -> Dict[str, int]:
        c: Dict[str, int] = {}
        for o in self.observations:
            c[o.hd_id] = c.get(o.hd_id, 0) + 1
        return c

    def count_by_instrument(self) -> Dict[str, int]:
        c: Dict[str, int] = {}
        for o in self.observations:
            c[o.instrument] = c.get(o.instrument, 0) + 1
        return c

    def obs_doys(self, hd: str) -> List[int]:
        return sorted(o.obs_time.datetime.timetuple().tm_yday
                      for o in self.observations if o.hd_id == hd)


# ── STEP 1: STAR PROFILES & EXPOSURE TIMES ────────────────────────────────────
# Calls get_star_profile() and calculate_exposures_and_uniform_visits()


def build_profiles(stars_input: List[dict], hwo_data) -> Tuple[dict, dict]:
    """
    Returns
        profiles      {hd_id: profile_dict}   — from simulator's get_star_profile
        inst_exptimes {inst: {hd_id: exptime_sec}}
    """
    profiles: dict = {}
    teff_map, vmag_map, vsini_map = {}, {}, {}

    for s in stars_input:
        p = get_star_profile(s, hwo_data)   # simulator function
        if p is None:
            continue
        hd = p['hd_id']
        # Store RA for the FixedTarget cache (needed by astroplan)
        if p.get('ra') is None:
            ra, _ = get_ra_dec(s['name'])   # simulator function
            p['ra'] = ra
        profiles[hd]  = p
        teff_map[hd]  = p['teff']
        vmag_map[hd]  = p['vmag']
        vsini_map[hd] = p['vsini']

    inst_exptimes: Dict[str, dict] = {}
    print("\n" + "="*65)
    print(f"{'INSTRUMENT':10}  {'STAR':12}  {'Exptime (s)':>11}  {'Cycle (s)':>9}  {'Max visits/night':>17}")
    print("-"*65)

    for inst, cfg in INSTRUMENTS.items():
        _, exptimes, _ = calculate_exposures_and_uniform_visits(   # simulator function
            list(profiles.keys()), teff_map, vmag_map, vsini_map,
            cfg["etc"],
            target_nights_for_uniformity=20,
            hours_per_night=HOURS_PER_NIGHT,
        )
        inst_exptimes[inst] = exptimes
        for hd in sorted(profiles.keys()):
            exp   = exptimes.get(hd, 0)
            cycle = exp + OVERHEAD_SEC
            maxv  = int(HOURS_PER_NIGHT * 3600 // cycle) if cycle > 0 else 0
            print(f"  {inst:8}  HD {hd:8}  {exp:>11.0f}  {cycle:>9.0f}  {maxv:>17}")
        print()

    return profiles, inst_exptimes


# ── STEP 2: OBSERVABLE HOURS CACHE ────────────────────────────────────────────
# Wraps astroplan directly (same logic as observable_window in the simulator)
# but caches every star × site × doy result so strategies run fast.

_OBS_CACHE: Dict[Tuple, float] = {}   # (hd, site, year, doy) -> hours


def _compute_obs_hours(hd: str, site: str, year: int, doy: int) -> float:
    """Compute observable hours for one star on one night at one site."""
    alt_limit = np.degrees(np.arcsin(1.0 / AIRMASS_LIMIT))
    try:
        loc      = EarthLocation.of_site(site)
        observer = Observer(location=loc, name=site)
        target   = _TARGET_CACHE[hd]   # populated before this is called
        t_mid    = Time(f"{year}-01-01 00:00:00") + (doy - 1) * u.day
        mid      = observer.midnight(t_mid, which="nearest")
        eve      = observer.twilight_evening_astronomical(mid, which="nearest")
        morn     = observer.twilight_morning_astronomical(mid, which="next")
        dt       = np.linspace(0, (morn - eve).to(u.hour).value, N_ALT_POINTS) * u.hour
        alts     = observer.altaz(eve + dt, target).alt.deg
        return float(np.sum(alts >= alt_limit) * (dt[1] - dt[0]).to(u.hour).value)
    except Exception:
        return 0.0


def build_obs_cache(stars: List[str], year: int) -> None:
    """Pre-compute observable hours for all star × site × doy combinations."""
    sites = {inst: INSTRUMENTS[inst]["site"] for inst in INSTRUMENTS}
    # Pre-build one Observer per site to avoid repeated EarthLocation calls
    observers = {site: Observer(location=EarthLocation.of_site(site), name=site)
                 for site in set(sites.values())}
    alt_limit = np.degrees(np.arcsin(1.0 / AIRMASS_LIMIT))
    total = len(stars) * len(sites) * 365
    done  = 0

    print(f"  Pre-computing observable hours ({total} evaluations)...")
    for hd in stars:
        target = _TARGET_CACHE[hd]
        for inst, site in sites.items():
            obs = observers[site]
            for doy in range(1, 366):
                key = (hd, site, year, doy)
                if key in _OBS_CACHE:
                    done += 1
                    continue
                try:
                    t_mid = Time(f"{year}-01-01 00:00:00") + (doy - 1) * u.day
                    mid   = obs.midnight(t_mid, which="nearest")
                    eve   = obs.twilight_evening_astronomical(mid, which="nearest")
                    morn  = obs.twilight_morning_astronomical(mid, which="next")
                    dt    = np.linspace(0, (morn - eve).to(u.hour).value,
                                        N_ALT_POINTS) * u.hour
                    alts  = obs.altaz(eve + dt, target).alt.deg
                    hours = float(np.sum(alts >= alt_limit)
                                  * (dt[1] - dt[0]).to(u.hour).value)
                    _OBS_CACHE[key] = hours
                except Exception:
                    _OBS_CACHE[key] = 0.0
                done += 1
            print(f"    {100*done/total:>5.1f}%  HD {hd}  {site}", end='\r')
    print(f"    100.0%  done.{' '*40}")


def obs_hours(hd: str, inst: str, year: int, doy: int) -> float:
    site = INSTRUMENTS[inst]["site"]
    return _OBS_CACHE.get((hd, site, year, doy), 0.0)


# ── STEP 3: TARGET NIGHTLY CADENCE ────────────────────────────────────────────
# target_cadence = floor(usable_time / (exptime + OVERHEAD_SEC))
# usable_time    = min(observable_hours × weather_fraction, HOURS_PER_NIGHT) × 3600

def weather_fraction(ws: WeatherStatus) -> float:
    return {WeatherStatus.CLEAR: 1.0, WeatherStatus.PARTIAL: 0.5, WeatherStatus.CLOSED: 0.0}[ws]


def target_cadence(hd: str, inst: str, exp_sec: float,
                   doy: int, year: int, wfrac: float = 1.0) -> int:
    """
    Maximum visits achievable for star HD {hd} on night doy with instrument inst.
    Accounts for: airmass/altitude (via obs_hours cache), weather fraction,
    and per-observation overhead.
    """
    usable_sec = min(obs_hours(hd, inst, year, doy) * wfrac,
                     HOURS_PER_NIGHT) * 3600.0
    cycle = exp_sec + OVERHEAD_SEC
    return int(usable_sec // cycle) if cycle > 0 and usable_sec >= cycle else 0


def print_cadence_report(profiles: dict, inst_exptimes: dict,
                          sample_days: List[int] = [80, 172, 264]) -> None:
    """
    Print target nightly cadence for representative days of year.
    Days 80/172/264 ≈ spring equinox, summer solstice, fall equinox.
    """
    stars = list(profiles.keys())
    print("\n" + "="*70)
    print("TARGET NIGHTLY CADENCE  (clear night, airmass ≤ 1.5, +120s overhead)")
    print("="*70)
    day_labels = "   ".join(f"DoY {d}" for d in sample_days)
    print(f"  {'Instrument':8}  {'Star':12}  {'Exptime':>8}  {day_labels}")
    print("  " + "-"*60)
    for inst in INSTRUMENTS:
        for hd in stars:
            exp      = inst_exptimes[inst].get(hd, 0)
            cadences = [target_cadence(hd, inst, exp, d, START_YEAR)
                        for d in sample_days]
            vals = "     ".join(f"{c:>5}" for c in cadences)
            print(f"  {inst:8}  HD {hd:8}  {exp:>8.0f}s  {vals}")
        print()


# ── STEP 4: WEATHER (uses simulator's read_monthly_weather_stats) ─────────────

def sample_weather(inst: str, year: int, num_years: int) -> Dict[str, WeatherStatus]:
    """
    Uses read_monthly_weather_stats() from the simulator.
    Returns {date_str: WeatherStatus} for every night.
    """
    fracs = read_monthly_weather_stats(INSTRUMENTS[inst]["weather_file"])  # simulator fn
    if not fracs:
        return {}
    result: Dict[str, WeatherStatus] = {}
    for yr_off in range(num_years):
        yr = year + yr_off
        for mi, frac in enumerate(fracs):
            for d in range(1, calendar.monthrange(yr, mi + 1)[1] + 1):
                r  = random.random()
                ws = (WeatherStatus.CLEAR   if r <= frac else
                      WeatherStatus.PARTIAL if r <= frac + 0.15 else
                      WeatherStatus.CLOSED)
                result[f"{yr}-{mi+1:02d}-{d:02d}"] = ws
    return result


def all_weather(year: int, num_years: int) -> Dict[str, Dict[str, WeatherStatus]]:
    return {inst: sample_weather(inst, year, num_years) for inst in INSTRUMENTS}


# ── STEP 5: FixedTarget cache (same pattern as before) ────────────────────────

_TARGET_CACHE: Dict[str, object] = {}


def build_target_cache(profiles: dict) -> None:
    for hd, p in profiles.items():
        ra, dec = p.get('ra'), p.get('dec')
        if ra is not None and dec is not None:
            _TARGET_CACHE[hd] = FixedTarget(
                coord=SkyCoord(ra * u.deg, dec * u.deg, frame='icrs'),
                name=f"HD {hd}"
            )
        else:
            print(f"  [cache] one-time SIMBAD lookup for HD {hd}...")
            _TARGET_CACHE[hd] = FixedTarget.from_name(f"HD {hd}")


# ── STEP 6: NIGHTLY SCHEDULING ENGINE ─────────────────────────────────────────
# Shared engine used by all strategies.
# Each night: determine open instruments, compute target cadence per star,
# schedule that many visits.

def _nights_sorted(weather_all: dict) -> List[str]:
    return sorted({dk for wmap in weather_all.values() for dk in wmap})


def _schedule_nights(
    stars:         List[str],
    inst_exptimes: dict,
    weather_all:   dict,
    strategy_name: str,
    label:         str,
    inst_picker,   # callable(night_idx, date_key, star, open_insts) -> inst or None
) -> ScheduleResult:
    """
    Core night loop. inst_picker decides which instrument to use for each
    star on each night — all strategy logic lives there, not here.
    """
    result = ScheduleResult(strategy_name=strategy_name, label=label)

    for night_idx, date_key in enumerate(_nights_sorted(weather_all)):
        try:
            dt       = datetime.strptime(date_key, "%Y-%m-%d")
            doy      = dt.timetuple().tm_yday
            obs_year = dt.year
        except ValueError:
            continue

        open_insts = [inst for inst in INSTRUMENTS
                      if weather_all[inst].get(date_key, WeatherStatus.CLOSED)
                      != WeatherStatus.CLOSED]
        if not open_insts:
            continue

        for hd in stars:
            inst = inst_picker(night_idx, date_key, hd, open_insts)
            if inst is None:
                continue

            exp   = inst_exptimes[inst].get(hd, 0)
            wfrac = weather_fraction(weather_all[inst].get(date_key, WeatherStatus.CLOSED))
            n     = target_cadence(hd, inst, exp, doy, obs_year, wfrac)
            if n <= 0:
                continue

            usable_sec = min(obs_hours(hd, inst, obs_year, doy) * wfrac,
                             HOURS_PER_NIGHT) * 3600.0
            t_base = Time(f"{date_key} 22:00:00", format='iso', scale='utc')
            for v in range(n):
                offset = (v + 0.5) * (usable_sec / n)
                result.observations.append(ScheduledObservation(
                    hd_id=hd, instrument=inst,
                    obs_time=t_base + offset * u.second,
                    exposure_sec=exp,
                    strategy=strategy_name,
                ))

    return result


# ── STRATEGIES ────────────────────────────────────────────────────────────────

def strategy_primary_fallback(
    stars: List[str], inst_exptimes: dict, weather_all: dict,
    primary: str = "NEID", fallback: str = "KPF",
) -> ScheduleResult:
    """
    S1 — PRIMARY / FALLBACK
    Every night: try NEID first. Use KPF only if NEID is closed.
    """
    def picker(night_idx, date_key, hd, open_insts):
        if primary in open_insts:
            return primary
        if fallback in open_insts:
            return fallback
        return None

    return _schedule_nights(
        stars, inst_exptimes, weather_all,
        strategy_name=f"S1  Primary ({primary}) / Fallback ({fallback})",
        label="S1: Primary/Fallback",
        inst_picker=picker,
    )


def strategy_alternation(
    stars: List[str], inst_exptimes: dict, weather_all: dict,
) -> ScheduleResult:
    """
    S2 — STRICT ALTERNATION
    Instruments alternate by night index: even nights → NEID, odd → KPF.
    Falls back to the other if the assigned instrument is closed.
    """
    inst_list = list(INSTRUMENTS.keys())

    def picker(night_idx, date_key, hd, open_insts):
        preferred = inst_list[night_idx % len(inst_list)]
        other     = inst_list[(night_idx + 1) % len(inst_list)]
        if preferred in open_insts:
            return preferred
        if other in open_insts:
            return other
        return None

    return _schedule_nights(
        stars, inst_exptimes, weather_all,
        strategy_name="S2  Strict Alternation (NEID ↔ KPF)",
        label="S2: Alternation",
        inst_picker=picker,
    )


def strategy_parallel_independent(
    stars: List[str], inst_exptimes: dict, weather_all: dict,
) -> ScheduleResult:
    """
    S4 — PARALLEL INDEPENDENT [control]
    Both instruments run simultaneously on every night they're open.
    No coordination. Same star can be observed by both in one night.
    This is the ceiling on total observations.
    """
    result = ScheduleResult(
        strategy_name="S4  Parallel Independent [control]",
        label="S4: Parallel (control)",
    )

    for date_key in _nights_sorted(weather_all):
        try:
            dt       = datetime.strptime(date_key, "%Y-%m-%d")
            doy      = dt.timetuple().tm_yday
            obs_year = dt.year
        except ValueError:
            continue

        for inst in INSTRUMENTS:
            ws    = weather_all[inst].get(date_key, WeatherStatus.CLOSED)
            wfrac = weather_fraction(ws)
            if wfrac == 0.0:
                continue

            for hd in stars:
                exp = inst_exptimes[inst].get(hd, 0)
                n   = target_cadence(hd, inst, exp, doy, obs_year, wfrac)
                if n <= 0:
                    continue

                usable_sec = min(obs_hours(hd, inst, obs_year, doy) * wfrac,
                                 HOURS_PER_NIGHT) * 3600.0
                t_base = Time(f"{date_key} 22:00:00", format='iso', scale='utc')
                for v in range(n):
                    offset = (v + 0.5) * (usable_sec / n)
                    result.observations.append(ScheduledObservation(
                        hd_id=hd, instrument=inst,
                        obs_time=t_base + offset * u.second,
                        exposure_sec=exp,
                        strategy=result.strategy_name,
                    ))

    return result


# ── PLOTS ─────────────────────────────────────────────────────────────────────


INST_CLR  = {"NEID": "#3A86FF", "KPF": "#FF6B35"}
STRAT_CLR = {"S1": "#3A86FF", "S2": "#FF6B35", "S4": "#2EC4B6"}
MONTH_TICKS = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366]
MONTH_LABELS = ['Jan','Feb','Mar','Apr','May','Jun',
                'Jul','Aug','Sep','Oct','Nov','Dec','']


def _style_ax(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', color='#e0e0e0', linewidth=0.6, zorder=0)


def plot_weather(weather_all: Dict[str, Dict[str, WeatherStatus]],
                 year: int = START_YEAR) -> None:
    """Clean weather calendar — one row per instrument."""
    clr = {WeatherStatus.CLEAR: "#3A86FF",
           WeatherStatus.PARTIAL: "#FFB347",
           WeatherStatus.CLOSED: "#E8E8E8"}
    n   = len(weather_all)
    fig, axes = plt.subplots(n, 1, figsize=(15, 2.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (inst, wmap) in zip(axes, weather_all.items()):
        days    = []
        colours = []
        for m in range(1, 13):
            for d in range(1, calendar.monthrange(year, m)[1] + 1):
                dk   = f"{year}-{m:02d}-{d:02d}"
                yday = datetime(year, m, d).timetuple().tm_yday
                days.append(yday)
                colours.append(clr[wmap.get(dk, WeatherStatus.CLOSED)])
        ax.bar(days, [1]*len(days), color=colours, width=1.0,
               edgecolor='none', zorder=2)
        n_use = sum(1 for ws in wmap.values() if ws != WeatherStatus.CLOSED)
        ax.set_ylabel(inst, fontsize=10, fontweight='bold',
                      rotation=0, ha='right', va='center', labelpad=38)
        ax.set_yticks([])
        ax.set_xlim(1, 366)
        ax.text(0.985, 0.5, f"{n_use} usable nights",
                transform=ax.transAxes, ha='right', va='center',
                fontsize=8.5, color='#555')
        ax.spines[['top','right','left','bottom']].set_visible(False)

    axes[-1].set_xticks(MONTH_TICKS)
    axes[-1].set_xticklabels(MONTH_LABELS, fontsize=9)

    patches = [mpatches.Patch(color=clr[ws], label=ws.value)
               for ws in [WeatherStatus.CLEAR, WeatherStatus.PARTIAL, WeatherStatus.CLOSED]]
    axes[-1].legend(handles=patches, ncol=3, loc='lower center',
                    bbox_to_anchor=(0.5, -0.55), fontsize=9, frameon=False)

    fig.suptitle(f"Simulated Weather  ({year})", fontsize=13,
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.show()


def plot_timelines(results: List[ScheduleResult], profiles: dict) -> None:
    """
    Grid: rows = strategies, cols = stars.
    Each dot = one observation, coloured by instrument.
    """
    stars    = sorted(profiles.keys())
    n_s      = len(results)
    n_st     = len(stars)
    inst_list = list(INSTRUMENTS.keys())

    fig, axes = plt.subplots(n_s, n_st, figsize=(5.5 * n_st, 2.4 * n_s),
                              sharex=True, squeeze=False)
    fig.subplots_adjust(hspace=0.08, wspace=0.05)

    for row, result in enumerate(results):
        for col, hd in enumerate(stars):
            ax = axes[row][col]
            for obs in result.observations:
                if obs.hd_id != hd:
                    continue
                doy = obs.obs_time.datetime.timetuple().tm_yday
                ax.scatter(doy, 0,
                           color=INST_CLR.get(obs.instrument, '#aaa'),
                           s=6, alpha=0.45, linewidths=0,
                           transform=ax.get_xaxis_transform())

            n_obs = result.count_by_star().get(hd, 0)
            ax.text(0.97, 0.82, f"n={n_obs}",
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=8, color='#444')

            ax.set_ylim(-0.5, 0.5)
            ax.set_yticks([])
            ax.set_xlim(1, 366)
            ax.set_xticks(MONTH_TICKS[:-1])
            ax.set_xticklabels(MONTH_LABELS[:-1], fontsize=7.5)
            _style_ax(ax)

            if row == 0:
                ax.set_title(f"HD {hd}", fontsize=11, fontweight='bold', pad=6)
            if col == 0:
                ax.set_ylabel(result.label, fontsize=8.5,
                               rotation=0, ha='right', va='center', labelpad=90)

    patches = [mpatches.Patch(color=INST_CLR[inst], label=inst)
               for inst in inst_list if inst in INST_CLR]
    fig.legend(handles=patches, loc='lower center', ncol=len(inst_list),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Observation Timelines — Strategy × Star",
                 fontsize=13, fontweight='bold', y=1.01)
    plt.show()


def plot_monthly_bars(results: List[ScheduleResult], profiles: dict) -> None:
    """Monthly observation counts per strategy, one panel per star."""
    stars  = sorted(profiles.keys())
    months = range(1, 13)
    x      = np.arange(12)
    w      = 0.25
    clrs   = [STRAT_CLR.get(r.label.split(":")[0].strip(), '#888')
              for r in results]

    fig, axes = plt.subplots(1, len(stars), figsize=(5.5 * len(stars), 4),
                              squeeze=False)
    for col, hd in enumerate(stars):
        ax = axes[0][col]
        for si, result in enumerate(results):
            monthly = [0] * 12
            for obs in result.observations:
                if obs.hd_id == hd:
                    monthly[obs.obs_time.datetime.month - 1] += 1
            offset = (si - 1) * w
            ax.bar(x + offset, monthly, w, label=result.label,
                   color=clrs[si], alpha=0.88, edgecolor='white', linewidth=0.5)

        ax.set_title(f"HD {hd}", fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'],
                            fontsize=8.5)
        if col == 0:
            ax.set_ylabel("Observations / month", fontsize=9)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        _style_ax(ax)

    handles = [mpatches.Patch(color=clrs[i], label=r.label)
               for i, r in enumerate(results)]
    fig.legend(handles=handles, loc='lower center', ncol=len(results),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Monthly Observation Counts by Strategy",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_cumulative(results: List[ScheduleResult], profiles: dict) -> None:
    """Cumulative observations over the year — one panel per star."""
    stars = sorted(profiles.keys())
    clrs  = [STRAT_CLR.get(r.label.split(":")[0].strip(), '#888')
             for r in results]

    fig, axes = plt.subplots(1, len(stars), figsize=(5.5 * len(stars), 4),
                              squeeze=False)
    for col, hd in enumerate(stars):
        ax = axes[0][col]
        for si, result in enumerate(results):
            doys = result.obs_doys(hd)
            if not doys:
                continue
            ax.step(doys, range(1, len(doys) + 1), where='post',
                    color=clrs[si], linewidth=1.8, label=result.label)

        ax.set_title(f"HD {hd}", fontsize=11, fontweight='bold')
        ax.set_xlim(1, 366)
        ax.set_xticks(MONTH_TICKS[:-1])
        ax.set_xticklabels(MONTH_LABELS[:-1], fontsize=8)
        if col == 0:
            ax.set_ylabel("Cumulative observations", fontsize=9)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        _style_ax(ax)
        ax.legend(fontsize=7.5, frameon=False)

    fig.suptitle("Cumulative Observations Over the Year",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_instrument_split(results: List[ScheduleResult]) -> None:
    """Stacked bar: NEID vs KPF contribution per strategy."""
    inst_list = list(INSTRUMENTS.keys())
    labels = [r.label for r in results]
    x      = np.arange(len(results))
    bottom = np.zeros(len(results))

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for inst in inst_list:
        counts = [r.count_by_instrument().get(inst, 0) for r in results]
        bars   = ax.bar(x, counts, bottom=bottom, label=inst,
                        color=INST_CLR.get(inst, '#aaa'),
                        alpha=0.88, edgecolor='white', linewidth=0.6)
        for i, (b, c) in enumerate(zip(bottom, counts)):
            if c > 0:
                ax.text(i, b + c / 2, str(c), ha='center', va='center',
                        fontsize=8.5, color='white', fontweight='bold')
        bottom = bottom + np.array(counts, dtype=float)

    # Total labels on top
    for i, r in enumerate(results):
        tot = len(r.observations)
        ax.text(i, bottom[i] + bottom.max() * 0.01, str(tot),
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("Total observations (all stars)", fontsize=10)
    ax.set_title("Instrument Contribution by Strategy",
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, frameon=False)
    _style_ax(ax)
    plt.tight_layout()
    plt.show()


def print_summary(results: List[ScheduleResult], stars: List[str]) -> None:
    inst_list = list(INSTRUMENTS.keys())
    w = 38
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    hdr = f"{'Strategy':<{w}} {'Total':>6}"
    for inst in inst_list:
        hdr += f"  {inst:>10}"
    for hd in sorted(stars):
        hdr += f"  {'HD '+hd:>10}"
    print(hdr)
    print("-"*80)
    for r in results:
        bi  = r.count_by_instrument()
        bs  = r.count_by_star()
        tot = len(r.observations)
        line = f"{r.strategy_name:<{w}} {tot:>6}"
        for inst in inst_list:
            n   = bi.get(inst, 0)
            pct = 100 * n / tot if tot else 0
            line += f"  {n:>4}({pct:>2.0f}%)"
        for hd in sorted(stars):
            line += f"  {bs.get(hd, 0):>10}"
        print(line)
    print("="*80)
    print("Instrument columns: count(% of total).  Star columns: total obs count.")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    random.seed(99)
    np.random.seed(99)


    print("MULTI-INSTRUMENT SCHEDULING:")


    # 1. Load HWO catalogue (uses simulator's load_hwo_candidates)
    hwo_data = load_hwo_candidates(hwo_file)

    # 2. Star profiles and exposure times
    print("\nBuilding star profiles...")
    profiles, inst_exptimes = build_profiles(MULTI_INSTRUMENT_STARS, hwo_data)
    if not profiles:
        print("No valid star profiles found. Exiting.")
        exit(1)
    stars = list(profiles.keys())

    # 3. FixedTarget cache (one coordinate lookup per star, reused everywhere)
    print("\nBuilding coordinate cache...")
    build_target_cache(profiles)
    print(f"  Cached: {[f'HD {hd}' for hd in _TARGET_CACHE]}")

    # 4. Observable hours cache (all stars × sites × days pre-computed once)
    print("\nPre-computing observable hours...")
    build_obs_cache(stars, START_YEAR)

    # 5. Target nightly cadence report
    print_cadence_report(profiles, inst_exptimes)

    # 6. Sample weather (uses simulator's read_monthly_weather_stats)
    print("\nSampling weather...")
    weather = all_weather(START_YEAR, NUM_YEARS)
    for inst, wmap in weather.items():
        n_cl = sum(1 for ws in wmap.values() if ws == WeatherStatus.CLEAR)
        n_pa = sum(1 for ws in wmap.values() if ws == WeatherStatus.PARTIAL)
        n_co = sum(1 for ws in wmap.values() if ws == WeatherStatus.CLOSED)
        print(f"  {inst}: {n_cl} clear  {n_pa} partial  {n_co} closed")
    plot_weather(weather)

    # 7. Run strategies
    print("RUNNING STRATEGIES")


    print("\n[S1] Primary(NEID) / Fallback(KPF)...")
    r1 = strategy_primary_fallback(stars, inst_exptimes, weather)
    print(f"     {len(r1.observations)} total observations")

    print("\n[S2] Strict Alternation...")
    r2 = strategy_alternation(stars, inst_exptimes, weather)
    print(f"     {len(r2.observations)} total observations")

    print("\n[S4] Parallel Independent (control)...")
    r4 = strategy_parallel_independent(stars, inst_exptimes, weather)
    print(f"     {len(r4.observations)} total observations")

    results = [r1, r2, r4]

    # 8. Summary and plots
    print_summary(results, stars)

    print("\nGenerating plots...")
    plot_timelines(results, profiles)
    plot_monthly_bars(results, profiles)
    plot_cumulative(results, profiles)
    plot_instrument_split(results)

    print("\nDone.")