#!/usr/bin/env python3
"""
BritzPlot — PALM Model Validation Against Britz Field Observations
====================================================================

Config-driven analysis package comparing PALM LES simulation output with
field observations from the Britz experimental forest site (Brandenburg,
Germany).  Validates soil moisture, soil temperature, leaf temperature,
air temperature, sap flow, and dendrometer observations against the
britz_snap_60m_feddes PALM simulation.

Usage:
    python britz_plot.py --config config.yml

Author: JoshuaB-L
"""

import argparse
import glob as globmod
import sys
from pathlib import Path

import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import mpl_toolkits.axisartist.floating_axes as FA
import mpl_toolkits.axisartist.grid_finder as GF
import netCDF4 as nc
import numpy as np
import pandas as pd
import pyproj
import yaml
import warnings
warnings.filterwarnings("ignore")

from scipy.stats import pearsonr
from scipy.signal import correlate, correlation_lags


# =============================================================================
# Configuration
# =============================================================================

def load_config(config_path):
    """Load and validate YAML configuration.

    Parameters
    ----------
    config_path : str or Path
        Path to the YAML config file.

    Returns
    -------
    cfg : dict
        Parsed configuration with all sections.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    for section in ("paths", "soil", "leaf_temp", "plot", "analysis_toggles"):
        if section not in cfg:
            sys.exit(f"ERROR: missing '{section}' section in config.")

    # Ensure met_tower section defaults
    if "met_tower" not in cfg:
        cfg["met_tower"] = {"enabled": False}

    Path(cfg["paths"]["output_dir"]).mkdir(parents=True, exist_ok=True)
    return cfg


# =============================================================================
# Matplotlib setup
# =============================================================================

def setup_matplotlib(plot_cfg):
    """Configure matplotlib rcParams for publication-quality output.

    Parameters
    ----------
    plot_cfg : dict
        Plot settings from config.yml (font sizes, DPI, format, etc.).
    """
    plt.rcParams.update({
        "font.family": plot_cfg.get("font_family", "serif"),
        "font.serif": ["Times New Roman", "DejaVu Serif", "Bitstream Vera Serif",
                        "Computer Modern Roman", "serif"],
        "font.size": plot_cfg.get("font_size", 9),
        "axes.titlesize": plot_cfg.get("title_size", 10),
        "axes.labelsize": plot_cfg.get("label_size", 9),
        "xtick.labelsize": plot_cfg.get("tick_size", 8),
        "ytick.labelsize": plot_cfg.get("tick_size", 8),
        "legend.fontsize": plot_cfg.get("legend_size", 7.5),
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#CCCCCC",
        "legend.fancybox": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "axes.grid": False,
        "figure.dpi": plot_cfg.get("dpi", 300),
        "savefig.dpi": plot_cfg.get("dpi", 300),
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "mathtext.default": "regular",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def mm_to_inches(mm):
    """Convert millimetres to inches for matplotlib figure sizing."""
    return mm / 25.4


# =============================================================================
# Output helpers
# =============================================================================

def _outpath(plot_dir, name, fmt, plot_cfg):
    """Build output path, prepending file_prefix from plot_cfg if set."""
    pfx = plot_cfg.get("file_prefix", "")
    return Path(plot_dir) / f"{pfx}{name}.{fmt}"


def _save_figure(fig, plot_dir, name, plot_cfg):
    """Save figure in all configured output formats, then close.

    Uses ``plot_cfg["output_formats"]`` (list) if present, otherwise falls
    back to the single ``plot_cfg["format"]``.
    """
    formats = plot_cfg.get("output_formats", [plot_cfg["format"]])
    for fmt in formats:
        out = _outpath(plot_dir, name, fmt, plot_cfg)
        fig.savefig(out, format=fmt)
        print(f"  Saved: {out}")
    plt.close(fig)


# =============================================================================
# Tol colorblind-safe palette
# =============================================================================

TOL_BLUE = "#4477AA"
TOL_CYAN = "#66CCEE"
TOL_GREEN = "#228833"
TOL_YELLOW = "#CCBB44"
TOL_RED = "#EE6677"
TOL_PURPLE = "#AA3377"
TOL_GREY = "#BBBBBB"


# =============================================================================
# PALM data loading and restart-series merging
# =============================================================================

def load_palm_restart_series(output_dir, job_name, file_type, domain,
                             variables=None):
    """Load and merge PALM NetCDF restart-series files into continuous arrays.

    Globs for ``{job_name}_{file_type}_{domain}.{NNN}.nc`` files, sorts by
    restart number, and concatenates time and data variables along the time
    axis.

    Parameters
    ----------
    output_dir : str or Path
        Directory containing PALM OUTPUT files.
    job_name : str
        PALM job name (e.g. ``"britz_snap_60m_feddes"``).
    file_type : str
        File type string (e.g. ``"av_3d"``).
    domain : str
        Domain suffix (e.g. ``"N02"``).
    variables : list of str or None
        If given, only load these data variable names (plus time and
        coordinates).  When *None*, load all data variables.

    Returns
    -------
    data : dict
        Merged arrays keyed by variable name, including ``"time"`` and all
        coordinate and requested data variables found in the files.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        sys.exit(f"ERROR: PALM output directory not found: {output_dir}")

    pattern = str(output_dir / f"{job_name}_{file_type}_{domain}.*.nc")
    files = sorted(globmod.glob(pattern))
    if not files:
        sys.exit(f"ERROR: no files matching {pattern}")

    # Identify coordinate (non-time) and data variables from first file
    ds0 = nc.Dataset(files[0], "r")
    coord_vars = set()
    all_data_vars = set()
    for vname, var in ds0.variables.items():
        if "time" not in var.dimensions and vname != "time":
            coord_vars.add(vname)
        elif vname != "time":
            all_data_vars.add(vname)
    ds0.close()

    # Filter to requested variables only
    if variables is not None:
        data_vars = set(variables) & all_data_vars
    else:
        data_vars = all_data_vars

    time_chunks = []
    data_chunks = {v: [] for v in data_vars}

    for fpath in files:
        ds = nc.Dataset(fpath, "r")
        time_chunks.append(ds.variables["time"][:])
        for v in data_vars:
            if v in ds.variables:
                data_chunks[v].append(ds.variables[v][:])
        ds.close()

    data = {"time": np.concatenate(time_chunks)}

    # Load coordinate variables from first file
    ds0 = nc.Dataset(files[0], "r")
    for v in coord_vars:
        data[v] = ds0.variables[v][:]
    ds0.close()

    for v in data_vars:
        if data_chunks[v]:
            data[v] = np.concatenate(data_chunks[v], axis=0)

    n_files = len(files)
    n_times = data["time"].shape[0]
    t0 = data["time"][0]
    t1 = data["time"][-1]
    print(f"  Loaded {n_files} files, {n_times} total timesteps, "
          f"time range: {t0:.1f} to {t1:.1f} s")

    return data


def extract_palm_soil_at_location(data, lat, lon, origin_E, origin_N, dx, dy):
    """Extract soil variables at the grid cell nearest to a lat/lon location.

    Converts observation lat/lon to UTM 33N (EPSG:25833), computes the offset
    from the PALM domain origin, and finds the nearest grid cell using the x/y
    coordinate arrays.

    Parameters
    ----------
    data : dict
        Merged PALM data from :func:`load_palm_restart_series`.  Must contain
        ``"x"`` and ``"y"`` coordinate arrays.
    lat, lon : float
        Observation station latitude and longitude (WGS 84).
    origin_E, origin_N : float
        PALM domain origin easting and northing (UTM 33N, metres).
    dx, dy : float
        Grid spacing in x and y directions (metres).

    Returns
    -------
    result : dict
        ``"m_soil"`` and ``"t_soil"`` arrays of shape ``(n_times, n_soil_levels)``,
        plus ``"ix"``, ``"iy"`` grid indices.
    """
    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", "EPSG:25833", always_xy=True
    )
    easting, northing = transformer.transform(lon, lat)

    # Offset from domain origin in metres (PALM x/y are relative to origin)
    dx_m = easting - origin_E
    dy_m = northing - origin_N

    x = np.asarray(data["x"])
    y = np.asarray(data["y"])
    ix = int(np.argmin(np.abs(x - dx_m)))
    iy = int(np.argmin(np.abs(y - dy_m)))

    print(f"  Grid cell (iy={iy}, ix={ix}) for station at "
          f"lat={lat}, lon={lon}; easting={easting:.1f}, northing={northing:.1f}")

    result = {"ix": ix, "iy": iy}

    for var in ("m_soil", "t_soil"):
        if var in data:
            arr = data[var][:, :, iy, ix]
            arr = np.ma.masked_values(arr, -999999.0)
            result[var] = arr

    return result


def align_time_axes(palm_time_seconds, obs_time_datetime, reference_time):
    """Align PALM time axis (seconds since reference) with observation datetimes.

    Rounds PALM time values to the nearest 900 s step, converts to
    DatetimeIndex, resamples observations to 15-minute bins, and returns
    the aligned (inner join) pair.

    Parameters
    ----------
    palm_time_seconds : array_like
        PALM time values in seconds since ``reference_time``.
    obs_time_datetime : pandas.DatetimeIndex or array_like
        Observation timestamps.
    reference_time : str
        Reference datetime string (e.g. ``"2024-09-04 00:00:00"``).

    Returns
    -------
    palm_aligned : pandas.Series
        PALM times as DatetimeIndex after alignment.
    obs_aligned : pandas.Series or pandas.DataFrame
        Observation data resampled and aligned to PALM times.
    common_times : pandas.DatetimeIndex
        The common timestamps.
    """
    ref = pd.to_datetime(reference_time)

    # Round PALM times to nearest 900s step
    palm_rounded = np.round(np.asarray(palm_time_seconds) / 900.0) * 900.0
    palm_dt = ref + pd.to_timedelta(palm_rounded, unit="s")
    palm_idx = pd.DatetimeIndex(palm_dt)

    # Build observation series/frame with datetime index
    if not isinstance(obs_time_datetime, pd.DatetimeIndex):
        obs_time_datetime = pd.DatetimeIndex(obs_time_datetime)

    # Find common timestamps via inner join
    common_times = palm_idx.intersection(obs_time_datetime)
    common_times = common_times.sort_values()

    return palm_idx, common_times


# =============================================================================
# Met tower observation loading
# =============================================================================

def load_tower_csv(csv_path, time_start, time_end):
    """Load a single met tower observation CSV file.

    Tower CSVs use semicolon-delimited format with 'DATE (DD.MM.YYYY)' and
    'TIME (UTC) (HH:MM:SS)' columns.  Returns a pd.Series with
    DatetimeIndex containing the first numeric data column, filtered to
    the requested time window.

    Parameters
    ----------
    csv_path : str or Path
        Path to the tower CSV file.
    time_start, time_end : str
        Simulation time window boundaries (parseable by pd.Timestamp).

    Returns
    -------
    pd.Series or None
        Time-filtered observation series, or None if loading fails.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"  [WARN] Tower CSV not found: {csv_path}")
        return None

    t0 = pd.Timestamp(time_start)
    t1 = pd.Timestamp(time_end)

    with open(csv_path, "r", encoding="utf-8", errors="ignore") as fh:
        raw_lines = fh.readlines()

    # Find the header row starting with 'DATE'
    header_idx = None
    for i, line in enumerate(raw_lines):
        if line.strip().startswith("DATE"):
            header_idx = i
            break
    if header_idx is None:
        print(f"  [WARN] No DATE header found in {csv_path}")
        return None

    header_parts = [c.strip() for c in raw_lines[header_idx].strip().split(";")]
    col_map = {name: idx for idx, name in enumerate(header_parts)}

    date_col = "DATE (DD.MM.YYYY)"
    time_col = "TIME (UTC) (HH:MM:SS)"
    if date_col not in col_map or time_col not in col_map:
        print(f"  [WARN] Required datetime columns missing in {csv_path}")
        return None

    date_idx = col_map[date_col]
    time_idx_col = col_map[time_col]

    # Identify the first numeric data column (not DATE/TIME)
    data_idx = None
    for name, idx in col_map.items():
        if idx not in (date_idx, time_idx_col):
            data_idx = idx
            break
    if data_idx is None:
        print(f"  [WARN] No data column found in {csv_path}")
        return None

    invalid_vals = {"-9999.00", "-999.00", "-99.00", "", "NaN", "nan"}
    timestamps, values = [], []
    max_idx = max(date_idx, time_idx_col, data_idx)

    for line in raw_lines[header_idx + 1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(";")
        if len(parts) <= max_idx:
            continue
        data_str = parts[data_idx].strip()
        if data_str in invalid_vals:
            continue
        try:
            dt = pd.to_datetime(
                f"{parts[date_idx].strip()} {parts[time_idx_col].strip()}",
                format="%d.%m.%Y %H:%M:%S",
            )
            val = float(data_str)
        except (ValueError, TypeError):
            continue
        if not np.isfinite(val):
            continue
        timestamps.append(dt)
        values.append(val)

    if not timestamps:
        print(f"  [WARN] No valid data in {csv_path}")
        return None

    s = pd.Series(values, index=pd.DatetimeIndex(timestamps))
    s = s[s.index.notna()]
    s = s[(s.index >= t0) & (s.index <= t1)]
    if s.empty:
        print(f"  [WARN] No data in time window for {csv_path}")
        return None
    return s


# =============================================================================
# ICON-D2 dynamic driver loading (netCDF4 only)
# =============================================================================

def _dd_height_to_z_index(target_height):
    """Map observation height (m) to dynamic driver z-index (5 m grid)."""
    if target_height <= 3.0:
        return 0
    elif target_height <= 9.0:
        return 1
    elif target_height <= 12.0:
        return 1
    else:
        return max(0, min(int((target_height - 2.5) / 5.0), 79))


def _average_boundary_forcing_nc(ds, variable_base, z_index, t_idx):
    """Average ls_forcing_{left,right,south,north}_{var} at given z and t.

    Parameters
    ----------
    ds : netCDF4.Dataset
        Open dynamic-driver file.
    variable_base : str
        Base variable name (e.g. 'pt', 'qv', 'u', 'v', 'w').
    z_index : int
        Vertical grid index.
    t_idx : int
        Time index.

    Returns
    -------
    float
        Mean across the four walls, or np.nan if unavailable.
    """
    vals = []
    for wall in ("left", "right", "south", "north"):
        vname = f"ls_forcing_{wall}_{variable_base}"
        if vname not in ds.variables:
            continue
        var = ds.variables[vname]
        zdim = "zw" if variable_base == "w" else "z"
        dim_names = var.dimensions
        if zdim not in dim_names:
            continue
        zi = list(dim_names).index(zdim)
        if z_index >= var.shape[zi]:
            continue
        # Build index slices
        slices = []
        for d in dim_names:
            if d == "time":
                slices.append(t_idx)
            elif d == zdim:
                slices.append(z_index)
            else:
                slices.append(slice(None))
        chunk = var[tuple(slices)]
        mean_val = float(np.nanmean(chunk))
        if np.isfinite(mean_val):
            vals.append(mean_val)
    return np.mean(vals) if vals else np.nan


def load_icon_d2_boundary_mean(driver_path, variable_name, target_height,
                                time_start, time_end):
    """Load ICON-D2 boundary-mean profile from a PALM dynamic driver.

    Reads ``ls_forcing_{left,right,south,north}_{var}`` variables, averages
    across the four walls at the z-level closest to *target_height*, and
    returns a time series.  Uses netCDF4 only (no xarray).

    Parameters
    ----------
    driver_path : str or Path
        Path to the dynamic driver NetCDF file.
    variable_name : str
        One of 'air_temperature', 'relative_humidity', 'wind_speed',
        'wind_direction'.
    target_height : float
        Observation height in metres for z-index mapping.
    time_start, time_end : str
        Time window boundaries.

    Returns
    -------
    pd.Series or None
        Boundary-mean time series, or None on failure.
    """
    driver_path = Path(driver_path)
    if not driver_path.exists():
        print(f"  [WARN] Dynamic driver not found: {driver_path}")
        return None

    t0 = pd.Timestamp(time_start)
    t1 = pd.Timestamp(time_end)
    z_idx = _dd_height_to_z_index(target_height)

    try:
        ds = nc.Dataset(str(driver_path), "r")
    except Exception as e:
        print(f"  [WARN] Cannot open dynamic driver: {e}")
        return None

    try:
        n_times = ds.dimensions["time"].size
        # Build time axis: try 'time' variable, fall back to hourly from t0
        if "time" in ds.variables:
            raw_t = ds.variables["time"][:]
            times = t0 + pd.to_timedelta(raw_t, unit="s")
        else:
            times = pd.date_range(start=t0, periods=n_times, freq="H")

        values = []
        for ti in range(n_times):
            if variable_name == "air_temperature":
                v = _average_boundary_forcing_nc(ds, "pt", z_idx, ti)
                if np.isfinite(v) and v > 200:
                    v -= 273.15
            elif variable_name == "relative_humidity":
                qv = _average_boundary_forcing_nc(ds, "qv", z_idx, ti)
                pt = _average_boundary_forcing_nc(ds, "pt", z_idx, ti)
                if np.isfinite(qv) and np.isfinite(pt):
                    tc = pt - 273.15 if pt > 200 else pt
                    p_hpa = 1013.25 * (1 - 0.0065 * target_height / 288.15) ** 5.255
                    es = 6.112 * np.exp(17.67 * tc / (tc + 243.5))
                    ws = 0.622 * es / (p_hpa - es)
                    v = 100.0 * qv / ws if ws > 0 else np.nan
                else:
                    v = np.nan
            elif variable_name == "wind_speed":
                u = _average_boundary_forcing_nc(ds, "u", z_idx, ti)
                vv = _average_boundary_forcing_nc(ds, "v", z_idx, ti)
                v = np.sqrt(u**2 + vv**2) if np.isfinite(u) and np.isfinite(vv) else np.nan
            elif variable_name == "wind_direction":
                u = _average_boundary_forcing_nc(ds, "u", z_idx, ti)
                vv = _average_boundary_forcing_nc(ds, "v", z_idx, ti)
                if np.isfinite(u) and np.isfinite(vv):
                    v = (270.0 - np.degrees(np.arctan2(vv, u))) % 360.0
                else:
                    v = np.nan
            else:
                v = np.nan
            values.append(v)

        ds.close()
    except Exception as e:
        ds.close()
        print(f"  [WARN] Error reading dynamic driver: {e}")
        return None

    s = pd.Series(values, index=times).dropna()
    s = s[(s.index >= t0) & (s.index <= t1)]
    return s if not s.empty else None


# =============================================================================
# Met tower grid cell and PALM extraction
# =============================================================================

def calculate_met_tower_grid_cell(tower_lat, tower_lon, data, origin_E,
                                  origin_N):
    """Calculate the PALM grid cell for the met tower location.

    Converts tower lat/lon to UTM 33N (EPSG:25833), computes offset from the
    PALM domain origin, and finds the nearest (iy, ix) in the grid coordinate
    arrays.

    Parameters
    ----------
    tower_lat, tower_lon : float
        Met tower WGS 84 coordinates.
    data : dict
        Merged PALM data containing ``"x"`` and ``"y"`` coordinate arrays,
        plus ``"zu_3d"`` for the vertical grid.
    origin_E, origin_N : float
        PALM domain origin in UTM 33N (metres).

    Returns
    -------
    ix, iy : int
        Grid indices in x and y directions.
    """
    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", "EPSG:25833", always_xy=True
    )
    easting, northing = transformer.transform(tower_lon, tower_lat)

    dx_m = easting - origin_E
    dy_m = northing - origin_N

    x = np.asarray(data["x"])
    y = np.asarray(data["y"])
    ix = int(np.argmin(np.abs(x - dx_m)))
    iy = int(np.argmin(np.abs(y - dy_m)))

    zu = np.asarray(data["zu_3d"]) if "zu_3d" in data else None
    z_info = f", z-levels: 0..{zu[-1]:.0f} m ({len(zu)} levels)" if zu is not None else ""
    print(f"  Met tower at lat={tower_lat}, lon={tower_lon} "
          f"-> grid cell (iy={iy}, ix={ix}){z_info}")

    # Verify within domain (not at boundary)
    if ix <= 0 or ix >= len(x) - 1 or iy <= 0 or iy >= len(y) - 1:
        print(f"  [WARN] Met tower grid cell is at domain boundary!")

    return ix, iy


def _find_z_index(zu, target_height):
    """Find nearest zu_3d index for a target height in metres."""
    return int(np.argmin(np.abs(np.asarray(zu) - target_height)))


def extract_palm_at_tower(data, ix, iy, height_to_palm_z, variables,
                          reference_time):
    """Extract PALM variables at the met tower grid cell for multiple heights.

    Parameters
    ----------
    data : dict
        Merged PALM data from :func:`load_palm_restart_series`.
    ix, iy : int
        Grid cell indices for the met tower location.
    height_to_palm_z : dict
        Mapping from tower observation height (m) to PALM z-coordinate (m),
        e.g. ``{2: 19.5, 3: 20.5, 8: 25.5, 10: 27.5}``.
    variables : list of str
        PALM variable names to extract (e.g. ``["ta", "rh", "wspeed"]``).
    reference_time : str
        Reference datetime for converting PALM time to DatetimeIndex.

    Returns
    -------
    result : dict
        ``{(variable, tower_height): pd.Series}`` with DatetimeIndex.
        Also includes ``"palm_times"`` key with the DatetimeIndex.
    """
    zu = np.asarray(data["zu_3d"])
    ref = pd.to_datetime(reference_time)

    # Build PALM DatetimeIndex
    palm_t = np.round(np.asarray(data["time"]) / 900.0) * 900.0
    palm_dt = ref + pd.to_timedelta(palm_t, unit="s")
    palm_idx = pd.DatetimeIndex(palm_dt)

    result = {"palm_times": palm_idx}

    for var in variables:
        if var not in data:
            print(f"  [WARN] Variable '{var}' not found in PALM data")
            continue

        arr = data[var]  # shape: (time, z, y, x)

        for tower_h, palm_z in height_to_palm_z.items():
            tower_h = int(tower_h) if isinstance(tower_h, str) else tower_h
            zi = _find_z_index(zu, palm_z)

            ts = arr[:, zi, iy, ix]
            ts = np.ma.filled(ts, fill_value=np.nan)
            ts = np.where(np.isclose(ts, -999999.0), np.nan, ts)
            series = pd.Series(ts.astype(float), index=palm_idx, name=f"{var}_{tower_h}m")
            result[(var, tower_h)] = series

        print(f"  Extracted '{var}' at {len(height_to_palm_z)} heights, "
              f"{len(palm_idx)} timesteps")

    return result


# =============================================================================
# Soil observation loading and depth-matching
# =============================================================================

SPECIES_LABELS = {
    "9801_02": "Beech (Ly2)",
    "9801_04": "Oak (Ly4)",
    "9801_08": "Douglas fir (Ly8)",
}


def load_soil_observations(csv_path, stations, time_start, time_end):
    """Load soil moisture observations from CSV, filter, average replicates.

    Parameters
    ----------
    csv_path : str or Path
        Path to ``wc_britz_all_stations_2024.csv``.
    stations : list of str
        Station IDs to load (e.g. ``["9801_02", "9801_04", "9801_08"]``).
    time_start, time_end : str
        ISO datetime strings bounding the time window.

    Returns
    -------
    obs : dict
        ``{station_id: pandas.DataFrame}`` with DatetimeIndex and one column
        per depth (in cm).  Values are volumetric water content in m3/m3.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        sys.exit(f"ERROR: soil observation CSV not found: {csv_path}")

    print(f"  Loading soil observations from {csv_path.name} ...")
    df = pd.read_csv(csv_path, parse_dates=["UTC_time"])

    # Filter by station and time window early
    t_start = pd.to_datetime(time_start)
    t_end = pd.to_datetime(time_end)
    df = df[df["station"].isin(stations)]
    df = df[(df["UTC_time"] >= t_start) & (df["UTC_time"] <= t_end)]
    print(f"    After filtering: {len(df)} rows for {len(stations)} stations")

    obs = {}
    for station in stations:
        sdf = df[df["station"] == station]

        # Average replicates per (time, depth)
        avg = (
            sdf.groupby(["UTC_time", "depth_cm"])["water_content_pct"]
            .mean()
            .reset_index()
        )

        # Convert vol% to m3/m3
        avg["water_content_pct"] = avg["water_content_pct"] / 100.0

        # Pivot to (time, depth) DataFrame
        pivot = avg.pivot(
            index="UTC_time", columns="depth_cm", values="water_content_pct"
        )
        pivot.index = pd.DatetimeIndex(pivot.index)
        pivot.index.name = "time"

        obs[station] = pivot
        n_times = len(pivot)
        n_depths = len(pivot.columns)
        label = SPECIES_LABELS.get(station, station)
        print(f"    {label}: {n_times} times x {n_depths} depths")

    return obs


def match_obs_to_palm_depths(obs_depths_cm, dz_soil):
    """Map observation depths to nearest PALM soil layer centres.

    Parameters
    ----------
    obs_depths_cm : array_like of float
        Observation depth values in centimetres (e.g. ``[10, 20, ..., 460]``).
    dz_soil : list of float
        PALM soil layer thicknesses in metres
        (e.g. ``[0.10, 0.10, 0.10, 0.10, 0.10, 0.50, 1.00, 1.00, 1.00, 0.60]``).

    Returns
    -------
    mapping : dict
        ``{obs_depth_cm: palm_layer_index}`` for the nearest layer centre.
    """
    dz = np.asarray(dz_soil)
    bottoms_m = np.cumsum(dz)
    centres_m = bottoms_m - dz / 2.0
    centres_cm = centres_m * 100.0
    bottoms_cm = bottoms_m * 100.0
    thickness_cm = dz * 100.0

    mapping = {}
    print("  Depth matching: obs_depth -> PALM layer (centre, offset)")

    for obs_d in obs_depths_cm:
        distances = np.abs(centres_cm - obs_d)
        idx = int(np.argmin(distances))
        offset = obs_d - centres_cm[idx]
        mapping[obs_d] = idx

        pct = abs(offset) / thickness_cm[idx] * 100.0
        warn = " *** WARNING: offset > 20% of layer thickness" if pct > 20 else ""
        print(f"    {obs_d:6.0f} cm -> layer {idx:2d} "
              f"(centre {centres_cm[idx]:6.1f} cm, offset {offset:+.1f} cm, "
              f"{pct:.0f}%){warn}")

    return mapping


# =============================================================================
# Leaf temperature observation loading and tree_id masking
# =============================================================================

def load_toa5_data(dat_path, sensor_metadata_csv, sensors="all_sensors",
                   exclude_sensors=None, exclude_after=None,
                   time_start=None, time_end=None):
    """Parse Campbell Scientific TOA5 .dat file and extract leaf/air columns.

    Parameters
    ----------
    dat_path : str or Path
        Path to the TOA5 .dat file (4-row header).
    sensor_metadata_csv : str or Path
        Path to oak_sensor_metadata.csv with leaf_col/air_col mappings.
    sensors : str or list
        ``"all_sensors"`` for all, or list of sensor_id ints.
    exclude_sensors : list of int or None
        Sensor IDs to exclude entirely.
    exclude_after : str or None
        ISO datetime; exclude data after this time for excluded sensors.
        (Not used for fully excluded sensors — they are dropped entirely.)
    time_start, time_end : str or None
        ISO datetimes to filter the output time window.

    Returns
    -------
    df : pandas.DataFrame
        DatetimeIndex, columns like Leaf1, Air1, Leaf2, Air2, ...
    metadata : pandas.DataFrame
        Sensor metadata table.
    """
    dat_path = Path(dat_path)
    if not dat_path.exists():
        sys.exit(f"ERROR: TOA5 file not found: {dat_path}")

    # Read column names from row 1 (second header row)
    col_names = pd.read_csv(dat_path, skiprows=1, nrows=0).columns.tolist()
    print(f"  TOA5 columns: {len(col_names)} columns")

    # Read data starting from row 4 (skip 4 header rows)
    df = pd.read_csv(
        dat_path, skiprows=4, header=None, names=col_names,
        na_values=["NAN", '"NAN"'], parse_dates=["TIMESTAMP"],
        index_col="TIMESTAMP", low_memory=False,
    )
    print(f"  TOA5 raw: {df.shape[0]} rows, {df.shape[1]} columns")

    # Load sensor metadata
    meta = pd.read_csv(sensor_metadata_csv)

    # Determine which sensors to include
    if exclude_sensors is None:
        exclude_sensors = []
    if sensors == "all_sensors":
        sensor_ids = [s for s in meta["sensor_id"].tolist()
                      if s not in exclude_sensors]
    else:
        sensor_ids = [s for s in sensors if s not in exclude_sensors]

    # Extract Leaf and Air columns for selected sensors
    keep_cols = []
    for sid in sensor_ids:
        row = meta[meta["sensor_id"] == sid]
        if row.empty:
            continue
        leaf_col = row.iloc[0]["leaf_col"]
        air_col = row.iloc[0]["air_col"]
        if leaf_col in df.columns:
            keep_cols.append(leaf_col)
        if air_col in df.columns:
            keep_cols.append(air_col)

    df = df[keep_cols]

    # Filter to simulation time window
    if time_start is not None:
        df = df[df.index >= pd.to_datetime(time_start)]
    if time_end is not None:
        df = df[df.index <= pd.to_datetime(time_end)]

    print(f"  TOA5 filtered: {df.shape[0]} rows, sensors: "
          f"{sorted(sensor_ids)}, excluded: {sorted(exclude_sensors)}")

    return df, meta


def build_tree_id_mask(static_path, palm_tree_id):
    """Build 3D boolean mask for a specific tree_id from the static driver.

    Parameters
    ----------
    static_path : str or Path
        Path to the PALM static driver NetCDF file.
    palm_tree_id : int
        The tree_id value to match.

    Returns
    -------
    mask : numpy.ndarray
        Boolean array of shape (zlad, y, x).
    voxels : list of tuple
        List of (z, y, x) coordinates where mask is True.
    """
    ds = nc.Dataset(str(static_path), "r")
    tree_id = ds.variables["tree_id"][:]
    ds.close()

    mask = np.asarray(tree_id == palm_tree_id)
    n_voxels = int(np.sum(mask))

    if n_voxels == 0:
        unique_ids = np.unique(tree_id[tree_id > 0])
        raise ValueError(
            f"tree_id {palm_tree_id} not found in static driver. "
            f"Available IDs: {unique_ids[:20].tolist()} "
            f"(total: {len(unique_ids)})"
        )

    zz, yy, xx = np.where(mask)
    voxels = list(zip(zz.tolist(), yy.tolist(), xx.tolist()))

    print(f"  tree_id {palm_tree_id}: {n_voxels} voxels, "
          f"z=[{zz.min()}-{zz.max()}], "
          f"y=[{yy.min()}-{yy.max()}], "
          f"x=[{xx.min()}-{xx.max()}]")

    return mask, voxels

#TODO - Need to find out if the mask is split into 3 vertical layers for North/South half of the tree, to match the 3 leaf/air sensor pairs.
def extract_palm_ta_at_tree(palm_3d_data, tree_mask, reference_time):
    """Extract PALM air temperature at tree crown from 3D output.

    PALM's time-averaged 3D output masks ``ta`` inside the plant canopy.
    For each (y, x) column in the tree footprint, this function finds
    the first valid z-level above the canopy top and extracts ``ta``
    there.  The per-column values are averaged spatially per timestep.

    Parameters
    ----------
    palm_3d_data : dict
        Merged PALM 3D data from :func:`load_palm_restart_series`.
        Must contain ``"ta"`` and ``"time"``.
    tree_mask : numpy.ndarray
        Boolean mask of shape (n_zlad, y, x) from :func:`build_tree_id_mask`.
    reference_time : str
        Reference datetime for converting PALM seconds to timestamps.

    Returns
    -------
    ta_series : pandas.Series
        Spatially averaged air temperature at tree crown, DatetimeIndex.
        Units are as-is from PALM (Celsius if units="degree_").
    """
    ta = palm_3d_data["ta"]  # shape: (n_times, n_zu, y, x)
    time_s = palm_3d_data["time"]
    n_zu = ta.shape[1]

    # Determine (y, x) footprint and max z per column from tree mask
    zz, yy, xx = np.where(tree_mask)
    yx_pairs = {}
    for z, y, x in zip(zz, yy, xx):
        key = (int(y), int(x))
        if key not in yx_pairs or z > yx_pairs[key]:
            yx_pairs[key] = int(z)

    # For each (y,x) column, find the first valid z in ta (first timestep)
    ta_first = ta[0]  # (n_zu, y, x)
    extract_indices = []  # (z, y, x) tuples with valid ta
    for (y, x), max_z in yx_pairs.items():
        for z in range(max_z, n_zu):
            val = ta_first[z, y, x]
            if not (np.ma.is_masked(val) if hasattr(val, "mask")
                    else (np.isnan(val) or val < -9e5)):
                extract_indices.append((z, y, x))
                break

    n_columns = len(yx_pairs)
    n_valid = len(extract_indices)
    if n_valid == 0:
        print(f"  WARNING: no valid ta values found above tree crown")
        ref = pd.to_datetime(reference_time)
        palm_rounded = np.round(np.asarray(time_s) / 900.0) * 900.0
        palm_dt = ref + pd.to_timedelta(palm_rounded, unit="s")
        return pd.Series(np.nan, index=pd.DatetimeIndex(palm_dt),
                         name="ta_tree")

    # Extract ta at those locations for all timesteps
    ta_values = []
    for t in range(ta.shape[0]):
        vals = []
        for z, y, x in extract_indices:
            val = ta[t, z, y, x]
            if hasattr(val, "mask") and np.ma.is_masked(val):
                continue
            fval = float(val)
            if fval > -9e5:
                vals.append(fval)
        ta_values.append(np.mean(vals) if vals else np.nan)

    # Build datetime index
    ref = pd.to_datetime(reference_time)
    palm_rounded = np.round(np.asarray(time_s) / 900.0) * 900.0
    palm_dt = ref + pd.to_timedelta(palm_rounded, unit="s")

    ta_series = pd.Series(ta_values, index=pd.DatetimeIndex(palm_dt),
                          name="ta_tree")

    z_levels = [z for z, _, _ in extract_indices]
    print(f"  PALM ta: {len(ta_values)} timesteps from {n_valid}/{n_columns} "
          f"columns, z-levels {min(z_levels)}-{max(z_levels)}, "
          f"range [{np.nanmin(ta_values):.2f}, {np.nanmax(ta_values):.2f}] "
          f"(Celsius — units='degree_', no conversion needed)")

    return ta_series


def split_tree_mask_north_south(tree_mask):
    """Split tree_id mask into North and South halves by median y-index.

    Parameters
    ----------
    tree_mask : numpy.ndarray
        Boolean mask of shape (zlad, y, x).

    Returns
    -------
    north_mask : numpy.ndarray
        Boolean mask for voxels in the north half (higher y-index).
    south_mask : numpy.ndarray
        Boolean mask for voxels in the south half (lower y-index).
    """
    zz, yy, xx = np.where(tree_mask)
    median_y = np.median(yy)

    north_mask = np.zeros_like(tree_mask, dtype=bool)
    south_mask = np.zeros_like(tree_mask, dtype=bool)

    for z, y, x in zip(zz, yy, xx):
        if y >= median_y:
            north_mask[z, y, x] = True
        else:
            south_mask[z, y, x] = True

    n_north = int(np.sum(north_mask))
    n_south = int(np.sum(south_mask))
    print(f"  North/South split: median_y={median_y:.1f}, "
          f"north={n_north} voxels, south={n_south} voxels")

    return north_mask, south_mask


# =============================================================================
# Statistical metrics
# =============================================================================

def compute_statistics(obs, sim):
    """Compute validation statistics for paired observation/simulation arrays.

    Implements RMSE, MBE, Pearson R, modified KGE (Kling et al. 2012), and
    Nash-Sutcliffe efficiency with pairwise NaN removal.

    Parameters
    ----------
    obs, sim : array_like
        Observed and simulated values (same length).

    Returns
    -------
    stats : dict
        Keys: ``rmse``, ``mbe``, ``r``, ``kge``, ``nse``, ``n_valid``,
        ``r_beta``, ``r_gamma``.
    """
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)

    # Pairwise NaN removal
    mask = ~(np.isnan(obs) | np.isnan(sim))
    obs = obs[mask]
    sim = sim[mask]

    nan_result = {
        "rmse": np.nan, "mbe": np.nan, "r": np.nan,
        "kge": np.nan, "nse": np.nan, "n_valid": len(obs),
        "r_beta": np.nan, "r_gamma": np.nan,
    }

    if len(obs) < 3:
        return nan_result

    rmse = np.sqrt(np.mean((sim - obs) ** 2))
    mbe = np.mean(sim - obs)
    r = np.corrcoef(obs, sim)[0, 1]

    # Modified KGE (Kling et al. 2012)
    mean_obs = np.mean(obs)
    mean_sim = np.mean(sim)

    if abs(mean_obs) < 1e-10:
        beta = np.nan
        gamma = np.nan
        kge = np.nan
    else:
        beta = mean_sim / mean_obs
        if abs(mean_sim) < 1e-10:
            gamma = np.nan
            kge = np.nan
        else:
            gamma = (np.std(sim) / mean_sim) / (np.std(obs) / mean_obs)
            kge = 1.0 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)

    # Nash-Sutcliffe efficiency
    ss_res = np.sum((obs - sim) ** 2)
    ss_tot = np.sum((obs - mean_obs) ** 2)
    nse = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "rmse": rmse,
        "mbe": mbe,
        "r": r,
        "kge": kge,
        "nse": nse,
        "n_valid": len(obs),
        "r_beta": beta,
        "r_gamma": gamma,
    }


# =============================================================================
# Soil comparison plots
# =============================================================================

def plot_soil_moisture_timeseries(obs_data, palm_data, station_cfg, depth_mapping,
                                  palm_time_dt, plot_cfg, outdir):
    """Plot soil moisture timeseries: obs vs PALM, one subplot per depth.

    Parameters
    ----------
    obs_data : dict
        ``{station_id: DataFrame}`` with DatetimeIndex and depth columns.
    palm_data : dict
        ``{station_id: {"m_soil": array(n_times, n_layers), ...}}``.
    station_cfg : dict
        Station config with labels and coordinates.
    depth_mapping : dict
        ``{obs_depth_cm: palm_layer_index}``.
    palm_time_dt : pandas.DatetimeIndex
        PALM times as datetimes.
    plot_cfg : dict
        Plot settings from config.
    outdir : str
        Output directory.

    Returns
    -------
    all_stats : list of dict
        Per-station, per-depth statistics for CSV export.
    """
    dz_soil = None  # will be derived from depth_mapping
    all_stats = []
    obs_depths = sorted(depth_mapping.keys())
    dz = np.asarray(list(depth_mapping.values()))

    fig_w = mm_to_inches(plot_cfg["figure_width_mm"])
    lw = plot_cfg.get("line_width", 1.5)

    for station_id, scfg in station_cfg.items():
        label = scfg["label"]
        obs_df = obs_data.get(station_id)
        palm_st = palm_data.get(station_id)
        if obs_df is None or palm_st is None:
            continue

        n_depths = len(obs_depths)
        fig_h = fig_w * 0.35 * max(n_depths, 2)
        fig, axes = plt.subplots(n_depths, 1, figsize=(fig_w, fig_h),
                                  sharex=True, squeeze=False)

        for i, depth_cm in enumerate(obs_depths):
            ax = axes[i, 0]
            palm_idx = depth_mapping[depth_cm]

            # Observation timeseries at this depth
            if depth_cm in obs_df.columns:
                obs_ts = obs_df[depth_cm].dropna()
            else:
                obs_ts = pd.Series(dtype=float)

            # PALM timeseries at matched layer
            palm_m_soil = palm_st["m_soil"][:, palm_idx]
            palm_series = pd.Series(
                np.ma.filled(palm_m_soil, np.nan), index=palm_time_dt
            )

            # Find common times
            common = obs_ts.index.intersection(palm_series.index).sort_values()

            if len(common) > 0:
                obs_aligned = obs_ts.reindex(common)
                palm_aligned = palm_series.reindex(common)
                stats = compute_statistics(obs_aligned.values, palm_aligned.values)
            else:
                stats = {"rmse": np.nan, "mbe": np.nan, "r": np.nan,
                         "kge": np.nan, "nse": np.nan, "n_valid": 0}

            # Compute PALM layer centre for stats export
            dz_arr = list(depth_mapping.values())
            # We need the actual dz_soil to compute centres
            # Use depth_mapping keys/values to reconstruct
            palm_centre_cm = _palm_layer_centre_cm(palm_idx, plot_cfg)

            all_stats.append({
                "station": station_id,
                "depth_cm": depth_cm,
                "palm_layer_index": palm_idx,
                "palm_layer_center_cm": palm_centre_cm,
                "offset_cm": depth_cm - palm_centre_cm,
                "rmse": stats["rmse"],
                "mbe": stats["mbe"],
                "r": stats["r"],
                "kge": stats["kge"],
                "nse": stats["nse"],
                "n_valid": stats["n_valid"],
            })

            # Plot
            ax.plot(obs_ts.index, obs_ts.values, color=TOL_BLUE, lw=lw,
                    label="Obs", zorder=3)
            ax.plot(palm_series.index, palm_series.values, color=TOL_RED,
                    lw=lw, ls="--", label="PALM", zorder=2)

            ax.set_ylabel(r"$\theta$ [m$^3$ m$^{-3}$]")
            depth_title = f"{depth_cm} cm"
            if not np.isnan(stats["rmse"]):
                depth_title += (f"  |  RMSE={stats['rmse']:.4f}  "
                                f"R={stats['r']:.3f}  "
                                f"KGE={stats['kge']:.3f}")
            ax.set_title(depth_title, fontsize=plot_cfg.get("tick_size", 8),
                         loc="left")

            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))

        axes[0, 0].legend(loc="upper right", ncol=2, frameon=True,
                           fancybox=False, edgecolor="#CCCCCC", framealpha=0.9)
        axes[-1, 0].set_xlabel("Time (UTC)")
        fig.suptitle(f"Soil Moisture: Obs vs PALM \u2014 {label}",
                      fontsize=plot_cfg.get("title_size", 10))
        fig.tight_layout()
        _save_figure(fig, outdir, f"soil_moisture_ts_{station_id}", plot_cfg)

    return all_stats


def _palm_layer_centre_cm(layer_idx, plot_cfg):
    """Return the PALM layer centre in cm for a given layer index.

    Reads dz_soil from plot_cfg['_dz_soil'] (injected at dispatch time).
    """
    dz_soil = plot_cfg.get("_dz_soil")
    if dz_soil is None:
        return np.nan
    dz = np.asarray(dz_soil)
    bottoms = np.cumsum(dz)
    centres = bottoms - dz / 2.0
    if layer_idx < len(centres):
        return centres[layer_idx] * 100.0
    return np.nan


def plot_soil_temperature_timeseries(palm_data, station_cfg, dz_soil,
                                      palm_time_dt, plot_cfg, outdir):
    """Plot PALM soil temperature evolution (no obs available).

    One line per soil depth; colour gradient from warm (shallow) to cool (deep).
    Converts PALM Kelvin to Celsius.

    Parameters
    ----------
    palm_data : dict
        ``{station_id: {"t_soil": array(n_times, n_layers), ...}}``.
    station_cfg : dict
        Station config with labels.
    dz_soil : list of float
        PALM soil layer thicknesses in metres.
    palm_time_dt : pandas.DatetimeIndex
        PALM times as datetimes.
    plot_cfg : dict
        Plot settings.
    outdir : str
        Output directory.
    """
    dz = np.asarray(dz_soil)
    centres_cm = (np.cumsum(dz) - dz / 2.0) * 100.0
    n_layers = len(dz)
    fig_w = mm_to_inches(plot_cfg["figure_width_mm"])
    lw = plot_cfg.get("line_width", 1.5)

    cmap = plt.cm.RdYlBu_r
    depth_colors = [cmap(i / max(n_layers - 1, 1)) for i in range(n_layers)]

    for station_id, scfg in station_cfg.items():
        label = scfg["label"]
        palm_st = palm_data.get(station_id)
        if palm_st is None or "t_soil" not in palm_st:
            continue

        t_soil = palm_st["t_soil"]
        # Convert K to C
        t_soil_C = np.ma.filled(t_soil, np.nan) - 273.15

        fig_h = fig_w * 0.55
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        for k in range(n_layers):
            ax.plot(palm_time_dt, t_soil_C[:, k], color=depth_colors[k],
                    lw=lw, label=f"{centres_cm[k]:.0f} cm")

        ax.set_ylabel(r"Soil temperature [$\degree$C]")
        ax.set_xlabel("Time (UTC)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
        ax.set_title(f"PALM Soil Temperature Evolution \u2014 {label}",
                      fontsize=plot_cfg.get("title_size", 10))
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
                   frameon=True, fancybox=False, edgecolor="#CCCCCC",
                   fontsize=plot_cfg.get("legend_size", 7.5), title="Depth")
        fig.tight_layout()
        _save_figure(fig, outdir, f"soil_temperature_ts_{station_id}", plot_cfg)


def plot_soil_vertical_profile_mean(obs_data, palm_data, station_cfg,
                                     depth_mapping, dz_soil, palm_time_dt,
                                     plot_cfg, outdir):
    """Plot time-averaged vertical profiles: obs vs PALM.

    Two panels: soil moisture (left) and soil temperature (right).
    Depth on y-axis (inverted), value on x-axis, error bars = temporal std dev.

    Parameters
    ----------
    obs_data : dict
        ``{station_id: DataFrame}`` with depth columns.
    palm_data : dict
        ``{station_id: {"m_soil": ..., "t_soil": ...}}``.
    station_cfg : dict
        Station config.
    depth_mapping : dict
        ``{obs_depth_cm: palm_layer_index}``.
    dz_soil : list of float
        PALM soil layer thicknesses (m).
    palm_time_dt : pandas.DatetimeIndex
        PALM times as datetimes.
    plot_cfg : dict
        Plot settings.
    outdir : str
        Output directory.
    """
    dz = np.asarray(dz_soil)
    centres_cm = (np.cumsum(dz) - dz / 2.0) * 100.0
    obs_depths = sorted(depth_mapping.keys())

    fig_w = mm_to_inches(plot_cfg["figure_width_mm"])
    fig_h = fig_w * 0.65
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_w, fig_h), sharey=True)
    lw = plot_cfg.get("line_width", 1.5)

    # Aggregate across all stations for a combined profile
    for station_id, scfg in station_cfg.items():
        obs_df = obs_data.get(station_id)
        palm_st = palm_data.get(station_id)
        if obs_df is None or palm_st is None:
            continue
        slabel = scfg["label"]

        # Obs moisture profile
        obs_means = []
        obs_stds = []
        for d in obs_depths:
            if d in obs_df.columns:
                col = obs_df[d].dropna()
                obs_means.append(col.mean())
                obs_stds.append(col.std())
            else:
                obs_means.append(np.nan)
                obs_stds.append(np.nan)

        ax1.errorbar(obs_means, obs_depths, xerr=obs_stds, fmt="o-",
                      color=TOL_BLUE, lw=lw * 0.8, markersize=3, capsize=2,
                      label=f"Obs ({slabel})", alpha=0.7)

        # PALM moisture profile at matched layers
        palm_means = []
        palm_stds = []
        palm_depths = []
        for d in obs_depths:
            idx = depth_mapping[d]
            m = np.ma.filled(palm_st["m_soil"][:, idx], np.nan)
            palm_means.append(np.nanmean(m))
            palm_stds.append(np.nanstd(m))
            palm_depths.append(centres_cm[idx])

        ax1.errorbar(palm_means, palm_depths, xerr=palm_stds, fmt="s--",
                      color=TOL_RED, lw=lw * 0.8, markersize=3, capsize=2,
                      label=f"PALM ({slabel})", alpha=0.7)

        # PALM temperature profile (no obs)
        if "t_soil" in palm_st:
            t_means = []
            t_stds = []
            t_depths = []
            for k in range(len(dz)):
                t = np.ma.filled(palm_st["t_soil"][:, k], np.nan) - 273.15
                t_means.append(np.nanmean(t))
                t_stds.append(np.nanstd(t))
                t_depths.append(centres_cm[k])
            ax2.errorbar(t_means, t_depths, xerr=t_stds, fmt="s--",
                          color=TOL_RED, lw=lw * 0.8, markersize=3, capsize=2,
                          label=f"PALM ({slabel})", alpha=0.7)

    ax1.set_xlabel(r"Soil moisture [m$^3$ m$^{-3}$]")
    ax1.set_ylabel("Depth [cm]")
    ax1.invert_yaxis()
    ax1.set_title("Soil Moisture", fontsize=plot_cfg.get("title_size", 10))
    ax1.legend(fontsize=plot_cfg.get("legend_size", 7.5) * 0.85, loc="lower left",
                frameon=True, fancybox=False, edgecolor="#CCCCCC")

    ax2.set_xlabel(r"Soil temperature [$\degree$C]")
    ax2.set_title("Soil Temperature", fontsize=plot_cfg.get("title_size", 10))
    ax2.legend(fontsize=plot_cfg.get("legend_size", 7.5) * 0.85, loc="lower right",
                frameon=True, fancybox=False, edgecolor="#CCCCCC")

    fig.suptitle("Time-Averaged Vertical Soil Profiles",
                  fontsize=plot_cfg.get("title_size", 10), y=1.01)
    fig.tight_layout()
    _save_figure(fig, outdir, "soil_vertical_profile_mean", plot_cfg)


def plot_soil_station_comparison(obs_data, palm_data, station_cfg,
                                  depth_mapping, palm_time_dt, plot_cfg,
                                  outdir):
    """Multi-panel bar chart comparing all stations vs PALM at each depth.

    Parameters
    ----------
    obs_data : dict
        ``{station_id: DataFrame}``.
    palm_data : dict
        ``{station_id: {"m_soil": ...}}``.
    station_cfg : dict
        Station config.
    depth_mapping : dict
        ``{obs_depth_cm: palm_layer_index}``.
    palm_time_dt : pandas.DatetimeIndex
        PALM times.
    plot_cfg : dict
        Plot settings.
    outdir : str
        Output directory.
    """
    obs_depths = sorted(depth_mapping.keys())
    stations = [s for s in station_cfg if s in obs_data and s in palm_data]
    if not stations:
        print("  Skipping soil_station_comparison — no data.")
        return

    n_depths = len(obs_depths)
    n_stations = len(stations)
    fig_w = mm_to_inches(plot_cfg["figure_width_mm"])
    fig_h = fig_w * 0.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    bar_w = 0.8 / (n_stations * 2)  # obs + PALM per station
    x = np.arange(n_depths)

    for i, sid in enumerate(stations):
        slabel = station_cfg[sid]["label"]
        obs_df = obs_data[sid]
        palm_st = palm_data[sid]

        # Time-averaged obs
        obs_means = []
        for d in obs_depths:
            if d in obs_df.columns:
                obs_means.append(obs_df[d].mean())
            else:
                obs_means.append(np.nan)

        # Time-averaged PALM
        palm_means = []
        for d in obs_depths:
            idx = depth_mapping[d]
            m = np.ma.filled(palm_st["m_soil"][:, idx], np.nan)
            palm_means.append(np.nanmean(m))

        offset_obs = (i * 2) * bar_w - (n_stations - 0.5) * bar_w
        offset_palm = (i * 2 + 1) * bar_w - (n_stations - 0.5) * bar_w

        ax.bar(x + offset_obs, obs_means, bar_w, label=f"{slabel} Obs",
               color=TOL_BLUE, alpha=0.5 + 0.2 * i, edgecolor="white", lw=0.3)
        ax.bar(x + offset_palm, palm_means, bar_w, label=f"{slabel} PALM",
               color=TOL_RED, alpha=0.5 + 0.2 * i, edgecolor="white", lw=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}" for d in obs_depths])
    ax.set_xlabel("Depth [cm]")
    ax.set_ylabel(r"Mean $\theta$ [m$^3$ m$^{-3}$]")
    ax.set_title("Station Comparison: Mean Soil Moisture by Depth",
                  fontsize=plot_cfg.get("title_size", 10))
    ax.legend(fontsize=plot_cfg.get("legend_size", 7.5) * 0.85, loc="upper right",
               ncol=2, frameon=True, fancybox=False, edgecolor="#CCCCCC")
    fig.tight_layout()
    _save_figure(fig, outdir, "soil_station_comparison", plot_cfg)


def export_soil_statistics_csv(all_stats, outdir):
    """Export soil comparison statistics to CSV.

    Parameters
    ----------
    all_stats : list of dict
        Per-station, per-depth statistics from plot_soil_moisture_timeseries.
    outdir : str
        Output directory.
    """
    if not all_stats:
        print("  No soil statistics to export.")
        return

    outpath = Path(outdir) / "soil_comparison_statistics.csv"
    fieldnames = ["station", "depth_cm", "palm_layer_index",
                  "palm_layer_center_cm", "offset_cm",
                  "rmse", "mbe", "r", "kge", "nse", "n_valid"]
    with open(outpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_stats:
            writer.writerow(row)
    print(f"  Exported: {outpath}")


# =============================================================================
# Leaf/air temperature comparison plots
# =============================================================================

def _resample_obs_to_palm(obs_series, palm_dt_index):
    """Resample observation series to PALM 15-min timestamps via nearest.

    Parameters
    ----------
    obs_series : pandas.Series
        Observation timeseries with DatetimeIndex (10-min frequency).
    palm_dt_index : pandas.DatetimeIndex
        PALM timestamps (15-min frequency).

    Returns
    -------
    aligned : pandas.Series
        Obs values reindexed to PALM timestamps using nearest-neighbour
        within a 10-minute tolerance.
    """
    return obs_series.reindex(palm_dt_index, method="nearest",
                              tolerance=pd.Timedelta("10min"))


def plot_leaf_air_temp_timeseries(obs_df, palm_ta_dict, tree_configs,
                                  sensor_meta, plot_cfg, outdir):
    """Plot leaf/air temperature timeseries: obs leaf, obs air, PALM ta.

    One figure per tree with one subplot per sensor.  Each subplot shows
    three lines: observed leaf temperature, observed air temperature, and
    PALM air temperature at the tree crown.

    Parameters
    ----------
    obs_df : pandas.DataFrame
        TOA5 data with DatetimeIndex, columns Leaf1, Air1, Leaf2, Air2, ...
    palm_ta_dict : dict
        ``{tree_name: pandas.Series}`` PALM ta at tree crown.
    tree_configs : dict
        Tree configuration from config (``leaf_temp.tree_ids`` section).
    sensor_meta : pandas.DataFrame
        Sensor metadata with sensor_id, leaf_col, air_col, position_code.
    plot_cfg : dict
        Plot settings.
    outdir : str
        Output directory.

    Returns
    -------
    all_stats : list of dict
        Per-sensor statistics for CSV export.
    """
    fig_w = mm_to_inches(plot_cfg["figure_width_mm"])
    lw = plot_cfg.get("line_width", 1.5)
    all_stats = []

    for tree_name, tcfg in tree_configs.items():
        palm_ta = palm_ta_dict.get(tree_name)
        if palm_ta is None:
            continue

        sensor_ids = tcfg["sensors"]
        valid_sensors = []
        for sid in sensor_ids:
            row = sensor_meta[sensor_meta["sensor_id"] == sid]
            if row.empty:
                continue
            leaf_col = row.iloc[0]["leaf_col"]
            if leaf_col in obs_df.columns:
                valid_sensors.append(sid)

        if not valid_sensors:
            continue

        n_sensors = len(valid_sensors)
        fig_h = fig_w * 0.35 * max(n_sensors, 2)
        fig, axes = plt.subplots(n_sensors, 1, figsize=(fig_w, fig_h),
                                  sharex=True, squeeze=False)

        for i, sid in enumerate(valid_sensors):
            ax = axes[i, 0]
            row = sensor_meta[sensor_meta["sensor_id"] == sid].iloc[0]
            leaf_col = row["leaf_col"]
            air_col = row["air_col"]
            pos_code = row["position_code"]

            obs_leaf = (obs_df[leaf_col].dropna()
                        if leaf_col in obs_df.columns
                        else pd.Series(dtype=float))
            obs_air = (obs_df[air_col].dropna()
                       if air_col in obs_df.columns
                       else pd.Series(dtype=float))

            ax.plot(obs_leaf.index, obs_leaf.values, color=TOL_GREEN, lw=lw,
                    label="Obs leaf", zorder=3)
            ax.plot(obs_air.index, obs_air.values, color=TOL_BLUE,
                    lw=lw * 0.8, label="Obs air", zorder=2, alpha=0.7)
            ax.plot(palm_ta.index, palm_ta.values, color=TOL_RED, lw=lw,
                    ls="--", label="PALM ta", zorder=4)

            # Compute stats: obs leaf vs PALM ta
            aligned_obs = _resample_obs_to_palm(obs_leaf, palm_ta.index)
            mask = ~(aligned_obs.isna() | palm_ta.isna())
            if mask.sum() >= 3:
                stats = compute_statistics(aligned_obs[mask].values,
                                           palm_ta[mask].values)
            else:
                stats = {"rmse": np.nan, "mbe": np.nan, "r": np.nan,
                         "kge": np.nan, "nse": np.nan, "n_valid": 0}

            all_stats.append({
                "tree_id": tree_name,
                "sensor_id": sid,
                "position_code": pos_code,
                "rmse": stats["rmse"],
                "mbe": stats["mbe"],
                "r": stats["r"],
                "kge": stats["kge"],
                "nse": stats["nse"],
                "n_valid": stats["n_valid"],
            })

            # Stats annotation box
            ann = (f"RMSE={stats['rmse']:.2f}  R={stats['r']:.3f}  "
                   f"KGE={stats['kge']:.3f}  n={stats['n_valid']}")
            ax.text(0.02, 0.95, ann, transform=ax.transAxes,
                    fontsize=plot_cfg.get("legend_size", 7.5),
                    va="top", ha="left",
                    bbox=dict(facecolor="white", alpha=0.8,
                              edgecolor="#CCCCCC",
                              boxstyle="round,pad=0.3"))

            # Delta_T diagnostic
            if len(obs_leaf) > 0 and len(obs_air) > 0:
                common_idx = obs_leaf.index.intersection(obs_air.index)
                if len(common_idx) > 0:
                    delta_t = obs_leaf.reindex(common_idx) - obs_air.reindex(common_idx)
                    dt_min, dt_max = delta_t.min(), delta_t.max()
                    if dt_min < -5 or dt_max > 10:
                        print(f"    WARNING S{sid}: Delta_T [{dt_min:.1f}, "
                              f"{dt_max:.1f}] outside expected range (-5, +10)")

            ax.set_ylabel("Temperature [\u00b0C]")
            ax.set_title(f"Sensor {sid} ({pos_code})",
                         fontsize=plot_cfg.get("tick_size", 8), loc="left")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))

        axes[0, 0].legend(loc="upper right", ncol=3, frameon=True,
                           fancybox=False, edgecolor="#CCCCCC",
                           framealpha=0.9)
        axes[-1, 0].set_xlabel("Time (UTC)")

        tree_label = (f"Oak {tcfg['britz_id']} "
                      f"(PALM tree_id {tcfg['palm_tree_id']})")
        fig.suptitle(f"Leaf/Air Temperature: Obs vs PALM \u2014 {tree_label}",
                      fontsize=plot_cfg.get("title_size", 10))
        fig.tight_layout()
        _save_figure(fig, outdir, f"leaf_air_temp_ts_{tree_name}", plot_cfg)

    return all_stats


def plot_leaf_temp_scatter(obs_df, palm_ta_dict, tree_configs,
                            sensor_meta, plot_cfg, outdir):
    """Scatter plot: observed leaf temperature vs PALM ta.

    All sensors combined on one plot, colored by sensor ID.  Includes
    1:1 reference line and linear regression with R^2 annotation.

    Parameters
    ----------
    obs_df : pandas.DataFrame
        TOA5 data with Leaf columns.
    palm_ta_dict : dict
        ``{tree_name: pandas.Series}`` PALM ta.
    tree_configs : dict
        Tree configuration.
    sensor_meta : pandas.DataFrame
        Sensor metadata.
    plot_cfg : dict
        Plot settings.
    outdir : str
        Output directory.
    """
    fig_w = mm_to_inches(plot_cfg["figure_width_mm"])
    fig_h = fig_w * 0.85
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    colors = [TOL_BLUE, TOL_CYAN, TOL_GREEN, TOL_YELLOW,
              TOL_RED, TOL_PURPLE, TOL_GREY, "#332288"]

    all_obs = []
    all_sim = []
    color_idx = 0

    for tree_name, tcfg in tree_configs.items():
        palm_ta = palm_ta_dict.get(tree_name)
        if palm_ta is None:
            continue

        for sid in tcfg["sensors"]:
            row = sensor_meta[sensor_meta["sensor_id"] == sid]
            if row.empty:
                continue
            leaf_col = row.iloc[0]["leaf_col"]
            if leaf_col not in obs_df.columns:
                continue
            pos_code = row.iloc[0]["position_code"]

            obs_leaf = obs_df[leaf_col].dropna()
            aligned = _resample_obs_to_palm(obs_leaf, palm_ta.index)
            mask = ~(aligned.isna() | palm_ta.isna())

            if mask.sum() < 2:
                color_idx += 1
                continue

            obs_vals = aligned[mask].values
            sim_vals = palm_ta[mask].values

            c = colors[color_idx % len(colors)]
            ax.scatter(sim_vals, obs_vals, s=12, alpha=0.6, color=c,
                       edgecolors="none",
                       label=f"S{sid} ({pos_code})", zorder=3)

            all_obs.extend(obs_vals.tolist())
            all_sim.extend(sim_vals.tolist())
            color_idx += 1

    if all_obs:
        all_obs_arr = np.array(all_obs)
        all_sim_arr = np.array(all_sim)

        vmin = min(all_obs_arr.min(), all_sim_arr.min()) - 1
        vmax = max(all_obs_arr.max(), all_sim_arr.max()) + 1
        ax.plot([vmin, vmax], [vmin, vmax], "k--", lw=0.8, alpha=0.6,
                label="1:1 line", zorder=1)

        valid = ~(np.isnan(all_obs_arr) | np.isnan(all_sim_arr))
        if valid.sum() >= 3:
            slope, intercept = np.polyfit(all_sim_arr[valid],
                                          all_obs_arr[valid], 1)
            r_val = np.corrcoef(all_sim_arr[valid],
                                all_obs_arr[valid])[0, 1]
            x_fit = np.linspace(vmin, vmax, 100)
            ax.plot(x_fit, slope * x_fit + intercept, color=TOL_RED,
                    lw=1.2, ls="-", alpha=0.8, zorder=2,
                    label=f"Regression (R\u00b2={r_val**2:.3f})")

        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("PALM ta [\u00b0C]")
    ax.set_ylabel("Obs leaf temperature [\u00b0C]")
    ax.set_title("Leaf Temperature: Obs vs PALM",
                  fontsize=plot_cfg.get("title_size", 10))
    ax.legend(fontsize=plot_cfg.get("legend_size", 7.5), loc="upper left",
               frameon=True, fancybox=False, edgecolor="#CCCCCC",
               framealpha=0.9)
    fig.tight_layout()
    _save_figure(fig, outdir, "leaf_temp_scatter", plot_cfg)


def plot_leaf_air_diurnal(obs_df, palm_ta_dict, tree_configs,
                           sensor_meta, plot_cfg, outdir):
    """Composite diurnal cycle: obs leaf, obs air, PALM ta.

    Hourly mean +/- 1 std dev bands, averaged across all sensors per tree.
    One figure per tree.

    Parameters
    ----------
    obs_df : pandas.DataFrame
        TOA5 data.
    palm_ta_dict : dict
        ``{tree_name: pandas.Series}`` PALM ta.
    tree_configs : dict
        Tree configuration.
    sensor_meta : pandas.DataFrame
        Sensor metadata.
    plot_cfg : dict
        Plot settings.
    outdir : str
        Output directory.
    """
    fig_w = mm_to_inches(plot_cfg["figure_width_mm"])
    lw = plot_cfg.get("line_width", 1.5)
    fill_alpha = plot_cfg.get("fill_alpha", 0.15)

    for tree_name, tcfg in tree_configs.items():
        palm_ta = palm_ta_dict.get(tree_name)
        if palm_ta is None:
            continue

        leaf_series_list = []
        air_series_list = []
        for sid in tcfg["sensors"]:
            row = sensor_meta[sensor_meta["sensor_id"] == sid]
            if row.empty:
                continue
            leaf_col = row.iloc[0]["leaf_col"]
            air_col = row.iloc[0]["air_col"]
            if leaf_col in obs_df.columns:
                leaf_series_list.append(obs_df[leaf_col])
            if air_col in obs_df.columns:
                air_series_list.append(obs_df[air_col])

        if not leaf_series_list:
            continue

        leaf_avg = pd.concat(leaf_series_list, axis=1).mean(axis=1)
        air_avg = (pd.concat(air_series_list, axis=1).mean(axis=1)
                   if air_series_list else None)

        leaf_hourly = leaf_avg.groupby(leaf_avg.index.hour)
        air_hourly = (air_avg.groupby(air_avg.index.hour)
                      if air_avg is not None else None)
        palm_hourly = palm_ta.groupby(palm_ta.index.hour)

        hours = np.arange(24)

        fig_h = fig_w * 0.55
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        leaf_mean = leaf_hourly.mean().reindex(hours)
        leaf_std = leaf_hourly.std().reindex(hours)
        ax.plot(hours, leaf_mean.values, color=TOL_GREEN, lw=lw,
                label="Obs leaf", zorder=3)
        ax.fill_between(hours,
                         (leaf_mean - leaf_std).values,
                         (leaf_mean + leaf_std).values,
                         color=TOL_GREEN, alpha=fill_alpha, zorder=1)

        if air_hourly is not None:
            air_mean = air_hourly.mean().reindex(hours)
            air_std = air_hourly.std().reindex(hours)
            ax.plot(hours, air_mean.values, color=TOL_BLUE, lw=lw * 0.8,
                    label="Obs air", zorder=2, alpha=0.8)
            ax.fill_between(hours,
                             (air_mean - air_std).values,
                             (air_mean + air_std).values,
                             color=TOL_BLUE, alpha=fill_alpha, zorder=1)

        palm_mean = palm_hourly.mean().reindex(hours)
        palm_std = palm_hourly.std().reindex(hours)
        ax.plot(hours, palm_mean.values, color=TOL_RED, lw=lw,
                ls="--", label="PALM ta", zorder=4)
        ax.fill_between(hours,
                         (palm_mean - palm_std).values,
                         (palm_mean + palm_std).values,
                         color=TOL_RED, alpha=fill_alpha, zorder=1)

        ax.set_xlim(0, 23)
        ax.set_xticks(np.arange(0, 24, 3))
        ax.set_xlabel("Hour of day (UTC)")
        ax.set_ylabel("Temperature [\u00b0C]")

        tree_label = (f"Oak {tcfg['britz_id']} "
                      f"(PALM tree_id {tcfg['palm_tree_id']})")
        ax.set_title(f"Diurnal Cycle: Obs vs PALM \u2014 {tree_label}",
                      fontsize=plot_cfg.get("title_size", 10))
        ax.legend(loc="upper right", ncol=3, frameon=True,
                   fancybox=False, edgecolor="#CCCCCC", framealpha=0.9)
        fig.tight_layout()
        _save_figure(fig, outdir, f"leaf_air_diurnal_{tree_name}", plot_cfg)


def plot_tree_averaged_comparison(obs_df, palm_ta_dict, tree_configs,
                                   sensor_meta, plot_cfg, outdir):
    """Tree-averaged obs leaf temp vs PALM ta.

    Averages all selected sensors per tree and compares with the spatially
    averaged PALM ta.  One subplot per tree with error bands (+/- 1 std
    across sensors).

    Parameters
    ----------
    obs_df : pandas.DataFrame
        TOA5 data.
    palm_ta_dict : dict
        ``{tree_name: pandas.Series}`` PALM ta.
    tree_configs : dict
        Tree configuration.
    sensor_meta : pandas.DataFrame
        Sensor metadata.
    plot_cfg : dict
        Plot settings.
    outdir : str
        Output directory.

    Returns
    -------
    tree_stats : list of dict
        Per-tree averaged statistics.
    """
    fig_w = mm_to_inches(plot_cfg["figure_width_mm"])
    lw = plot_cfg.get("line_width", 1.5)
    fill_alpha = plot_cfg.get("fill_alpha", 0.15)
    n_trees = len(tree_configs)
    tree_stats = []

    fig_h = fig_w * 0.4 * max(n_trees, 1)
    fig, axes = plt.subplots(n_trees, 1, figsize=(fig_w, fig_h),
                              sharex=True, squeeze=False)

    for idx, (tree_name, tcfg) in enumerate(tree_configs.items()):
        ax = axes[idx, 0]
        palm_ta = palm_ta_dict.get(tree_name)
        if palm_ta is None:
            continue

        leaf_series_list = []
        for sid in tcfg["sensors"]:
            row = sensor_meta[sensor_meta["sensor_id"] == sid]
            if row.empty:
                continue
            leaf_col = row.iloc[0]["leaf_col"]
            if leaf_col in obs_df.columns:
                leaf_series_list.append(obs_df[leaf_col])

        if not leaf_series_list:
            continue

        leaf_df = pd.concat(leaf_series_list, axis=1)
        leaf_mean = leaf_df.mean(axis=1)
        leaf_std = leaf_df.std(axis=1)

        leaf_mean_aligned = _resample_obs_to_palm(leaf_mean, palm_ta.index)

        ax.plot(leaf_mean.index, leaf_mean.values, color=TOL_GREEN, lw=lw,
                label="Obs leaf (mean)", zorder=3)
        ax.fill_between(leaf_mean.index,
                         (leaf_mean - leaf_std).values,
                         (leaf_mean + leaf_std).values,
                         color=TOL_GREEN, alpha=fill_alpha, zorder=1)
        ax.plot(palm_ta.index, palm_ta.values, color=TOL_RED, lw=lw,
                ls="--", label="PALM ta", zorder=4)

        mask = ~(leaf_mean_aligned.isna() | palm_ta.isna())
        if mask.sum() >= 3:
            stats = compute_statistics(leaf_mean_aligned[mask].values,
                                       palm_ta[mask].values)
        else:
            stats = {"rmse": np.nan, "mbe": np.nan, "r": np.nan,
                     "kge": np.nan, "nse": np.nan, "n_valid": 0}

        tree_stats.append({
            "tree_id": tree_name,
            "sensor_id": "avg",
            "position_code": "tree_avg",
            "rmse": stats["rmse"],
            "mbe": stats["mbe"],
            "r": stats["r"],
            "kge": stats["kge"],
            "nse": stats["nse"],
            "n_valid": stats["n_valid"],
        })

        ann = (f"RMSE={stats['rmse']:.2f}  R={stats['r']:.3f}  "
               f"KGE={stats['kge']:.3f}  n={stats['n_valid']}")
        ax.text(0.02, 0.95, ann, transform=ax.transAxes,
                fontsize=plot_cfg.get("legend_size", 7.5),
                va="top", ha="left",
                bbox=dict(facecolor="white", alpha=0.8,
                          edgecolor="#CCCCCC",
                          boxstyle="round,pad=0.3"))

        tree_label = (f"Oak {tcfg['britz_id']} "
                      f"(PALM tree_id {tcfg['palm_tree_id']})")
        ax.set_ylabel("Temperature [\u00b0C]")
        ax.set_title(tree_label, fontsize=plot_cfg.get("tick_size", 8),
                     loc="left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))

    axes[0, 0].legend(loc="upper right", ncol=2, frameon=True,
                       fancybox=False, edgecolor="#CCCCCC", framealpha=0.9)
    axes[-1, 0].set_xlabel("Time (UTC)")
    fig.suptitle("Tree-Averaged Leaf Temperature: Obs vs PALM",
                  fontsize=plot_cfg.get("title_size", 10))
    fig.tight_layout()
    _save_figure(fig, outdir, "tree_averaged_comparison", plot_cfg)

    return tree_stats


def export_leaf_temp_statistics_csv(all_stats, outdir):
    """Export leaf temperature comparison statistics to CSV.

    Parameters
    ----------
    all_stats : list of dict
        Per-sensor and per-tree statistics.
    outdir : str
        Output directory.
    """
    if not all_stats:
        print("  No leaf temperature statistics to export.")
        return

    outpath = Path(outdir) / "leaf_temp_comparison_statistics.csv"
    fieldnames = ["tree_id", "sensor_id", "position_code",
                  "rmse", "mbe", "r", "kge", "nse", "n_valid"]
    with open(outpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_stats:
            writer.writerow(row)
    print(f"  Exported: {outpath}")


# =============================================================================
# Met tower comparison plots
# =============================================================================

def _annotate_stats(ax, stats, plot_cfg):
    """Add RMSE/MBE/R/KGE stats annotation box to an axis."""
    ann = (f"RMSE={stats['rmse']:.2f}  MBE={stats['mbe']:.2f}\n"
           f"R={stats['r']:.3f}  KGE={stats['kge']:.3f}  n={stats['n_valid']}")
    ax.text(0.02, 0.95, ann, transform=ax.transAxes,
            fontsize=plot_cfg.get("legend_size", 7.5),
            va="top", ha="left",
            bbox=dict(facecolor="white", alpha=0.8,
                      edgecolor="#CCCCCC", boxstyle="round,pad=0.3"))


def _met_tower_subplot_layout(n_heights, plot_cfg):
    """Create figure with n_heights vertically stacked subplots."""
    fig_w = mm_to_inches(plot_cfg["figure_width_mm"])
    fig_h = fig_w * 0.35 * max(n_heights, 2)
    fig, axes = plt.subplots(n_heights, 1, figsize=(fig_w, fig_h),
                              sharex=True, squeeze=False)
    return fig, axes


def _align_tower_to_palm(obs_series, palm_series, avg_rule):
    """Resample tower obs to avg_rule, then align with PALM series."""
    if obs_series is None or palm_series is None:
        return None, None, {"rmse": np.nan, "mbe": np.nan, "r": np.nan,
                            "kge": np.nan, "nse": np.nan, "n_valid": 0}
    obs_rs = obs_series.resample(avg_rule).mean().dropna()
    palm_rs = palm_series.copy()
    common = obs_rs.index.intersection(palm_rs.index).sort_values()
    if len(common) < 3:
        return obs_rs, palm_rs, {"rmse": np.nan, "mbe": np.nan, "r": np.nan,
                                  "kge": np.nan, "nse": np.nan, "n_valid": 0}
    stats = compute_statistics(obs_rs.reindex(common).values,
                               palm_rs.reindex(common).values)
    return obs_rs, palm_rs, stats


def plot_met_tower_temperature(tower_obs, palm_data, icon_data, stats_list,
                                plot_cfg, outdir, met_cfg):
    """Plot met tower temperature comparison: obs vs PALM vs optional ICON-D2.

    One subplot per observation height. Returns list of stats dicts.
    """
    heights = sorted(set(tower_obs.get("air_temperature", {}).keys()))
    if not heights:
        print("  [SKIP] No temperature observations loaded")
        return []

    avg_rule = met_cfg.get("temporal_averaging", {}).get("temp_humidity", "15T")
    lw = plot_cfg.get("line_width", 1.5)
    fig, axes = _met_tower_subplot_layout(len(heights), plot_cfg)
    all_stats = []

    for i, h in enumerate(heights):
        ax = axes[i, 0]
        obs_s = tower_obs["air_temperature"].get(h)
        palm_s = palm_data.get(("ta", h))
        icon_s = icon_data.get(("air_temperature", h)) if icon_data else None

        obs_rs, palm_rs, stats = _align_tower_to_palm(obs_s, palm_s, avg_rule)

        palm_z = met_cfg["height_to_palm_z"].get(h, h)
        if obs_rs is not None:
            ax.plot(obs_rs.index, obs_rs.values, color=TOL_BLUE, lw=lw,
                    label=f"Tower {h}m", zorder=3)
        if palm_rs is not None:
            ax.plot(palm_rs.index, palm_rs.values, color=TOL_RED, lw=lw,
                    ls="--", label=f"PALM z={palm_z}m", zorder=2)
        if icon_s is not None:
            ax.plot(icon_s.index, icon_s.values, color=TOL_GREEN, lw=lw * 0.8,
                    ls=":", label="ICON-D2", zorder=1, alpha=0.7)

        _annotate_stats(ax, stats, plot_cfg)
        ax.set_ylabel("Temperature [\u00b0C]")
        ax.set_title(f"Air Temperature — {h}m",
                     fontsize=plot_cfg.get("tick_size", 8), loc="left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))

        all_stats.append({
            "variable": "air_temperature", "height_m": h,
            "palm_z_m": palm_z, **stats,
        })
        stats_list.append(all_stats[-1])

    axes[0, 0].legend(loc="upper right", ncol=3, frameon=True,
                       fancybox=False, edgecolor="#CCCCCC", framealpha=0.9)
    axes[-1, 0].set_xlabel("Time (UTC)")
    fig.suptitle("Met Tower Temperature: Obs vs PALM",
                  fontsize=plot_cfg.get("title_size", 10))
    fig.tight_layout()
    _save_figure(fig, outdir, "met_tower_temperature", plot_cfg)
    return all_stats


def plot_met_tower_humidity(tower_obs, palm_data, icon_data, stats_list,
                             plot_cfg, outdir, met_cfg):
    """Plot met tower relative humidity comparison."""
    heights = sorted(set(tower_obs.get("relative_humidity", {}).keys()))
    if not heights:
        print("  [SKIP] No humidity observations loaded")
        return []

    avg_rule = met_cfg.get("temporal_averaging", {}).get("temp_humidity", "15T")
    lw = plot_cfg.get("line_width", 1.5)
    fig, axes = _met_tower_subplot_layout(len(heights), plot_cfg)
    all_stats = []

    for i, h in enumerate(heights):
        ax = axes[i, 0]
        obs_s = tower_obs["relative_humidity"].get(h)
        palm_s = palm_data.get(("rh", h))
        icon_s = icon_data.get(("relative_humidity", h)) if icon_data else None

        obs_rs, palm_rs, stats = _align_tower_to_palm(obs_s, palm_s, avg_rule)

        palm_z = met_cfg["height_to_palm_z"].get(h, h)
        if obs_rs is not None:
            ax.plot(obs_rs.index, obs_rs.values, color=TOL_BLUE, lw=lw,
                    label=f"Tower {h}m", zorder=3)
        if palm_rs is not None:
            ax.plot(palm_rs.index, palm_rs.values, color=TOL_RED, lw=lw,
                    ls="--", label=f"PALM z={palm_z}m", zorder=2)
        if icon_s is not None:
            ax.plot(icon_s.index, icon_s.values, color=TOL_GREEN, lw=lw * 0.8,
                    ls=":", label="ICON-D2", zorder=1, alpha=0.7)

        _annotate_stats(ax, stats, plot_cfg)
        ax.set_ylabel("Relative humidity [%]")
        ax.set_title(f"Relative Humidity — {h}m",
                     fontsize=plot_cfg.get("tick_size", 8), loc="left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))

        all_stats.append({
            "variable": "relative_humidity", "height_m": h,
            "palm_z_m": palm_z, **stats,
        })
        stats_list.append(all_stats[-1])

    axes[0, 0].legend(loc="upper right", ncol=3, frameon=True,
                       fancybox=False, edgecolor="#CCCCCC", framealpha=0.9)
    axes[-1, 0].set_xlabel("Time (UTC)")
    fig.suptitle("Met Tower Humidity: Obs vs PALM",
                  fontsize=plot_cfg.get("title_size", 10))
    fig.tight_layout()
    _save_figure(fig, outdir, "met_tower_humidity", plot_cfg)
    return all_stats


def plot_met_tower_wind(tower_obs, palm_data, icon_data, stats_list,
                         plot_cfg, outdir, met_cfg):
    """Plot met tower wind speed and direction comparison.

    Wind speed: standard time series.
    Wind direction: circular time series with 0-360 y-axis.
    """
    ws_heights = sorted(set(tower_obs.get("wind_speed", {}).keys()))
    wd_heights = sorted(set(tower_obs.get("wind_direction", {}).keys()))
    n_panels = len(ws_heights) + len(wd_heights)
    if n_panels == 0:
        print("  [SKIP] No wind observations loaded")
        return []

    avg_rule = met_cfg.get("temporal_averaging", {}).get("wind", "15T")
    lw = plot_cfg.get("line_width", 1.5)
    fig, axes = _met_tower_subplot_layout(n_panels, plot_cfg)
    all_stats = []
    panel = 0

    # Wind speed panels
    for h in ws_heights:
        ax = axes[panel, 0]
        obs_s = tower_obs["wind_speed"].get(h)
        palm_s = palm_data.get(("wspeed", h))
        icon_s = icon_data.get(("wind_speed", h)) if icon_data else None

        obs_rs, palm_rs, stats = _align_tower_to_palm(obs_s, palm_s, avg_rule)

        palm_z = met_cfg["height_to_palm_z"].get(h, h)
        if obs_rs is not None:
            ax.plot(obs_rs.index, obs_rs.values, color=TOL_BLUE, lw=lw,
                    label=f"Tower {h}m", zorder=3)
        if palm_rs is not None:
            ax.plot(palm_rs.index, palm_rs.values, color=TOL_RED, lw=lw,
                    ls="--", label=f"PALM z={palm_z}m", zorder=2)
        if icon_s is not None:
            ax.plot(icon_s.index, icon_s.values, color=TOL_GREEN, lw=lw * 0.8,
                    ls=":", label="ICON-D2", zorder=1, alpha=0.7)

        _annotate_stats(ax, stats, plot_cfg)
        ax.set_ylabel("Wind speed [m/s]")
        ax.set_title(f"Wind Speed — {h}m",
                     fontsize=plot_cfg.get("tick_size", 8), loc="left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))

        all_stats.append({
            "variable": "wind_speed", "height_m": h,
            "palm_z_m": palm_z, **stats,
        })
        stats_list.append(all_stats[-1])
        panel += 1

    # Wind direction panels
    for h in wd_heights:
        ax = axes[panel, 0]
        obs_s = tower_obs["wind_direction"].get(h)
        palm_s = palm_data.get(("wdir", h))
        icon_s = icon_data.get(("wind_direction", h)) if icon_data else None

        # For wind direction, use circular mean for resampling
        obs_rs = _circular_resample(obs_s, avg_rule) if obs_s is not None else None
        palm_rs = palm_s

        # Circular statistics for direction
        if obs_rs is not None and palm_rs is not None:
            common = obs_rs.index.intersection(palm_rs.index).sort_values()
            if len(common) >= 3:
                o = obs_rs.reindex(common).values
                s = palm_rs.reindex(common).values
                stats = _circular_statistics(o, s)
            else:
                stats = {"rmse": np.nan, "mbe": np.nan, "r": np.nan,
                         "kge": np.nan, "nse": np.nan, "n_valid": 0}
        else:
            stats = {"rmse": np.nan, "mbe": np.nan, "r": np.nan,
                     "kge": np.nan, "nse": np.nan, "n_valid": 0}

        palm_z = met_cfg["height_to_palm_z"].get(h, h)
        if obs_rs is not None:
            ax.plot(obs_rs.index, obs_rs.values, color=TOL_BLUE, lw=lw,
                    label=f"Tower {h}m", zorder=3, marker=".", markersize=2,
                    linestyle="none")
        if palm_rs is not None:
            ax.plot(palm_rs.index, palm_rs.values, color=TOL_RED, lw=lw,
                    label=f"PALM z={palm_z}m", zorder=2, marker=".",
                    markersize=2, linestyle="none")
        if icon_s is not None:
            ax.plot(icon_s.index, icon_s.values, color=TOL_GREEN,
                    marker=".", markersize=2, linestyle="none",
                    label="ICON-D2", zorder=1, alpha=0.7)

        ax.set_ylim(0, 360)
        ax.set_yticks([0, 90, 180, 270, 360])
        ax.set_yticklabels(["N", "E", "S", "W", "N"])
        _annotate_stats(ax, stats, plot_cfg)
        ax.set_ylabel("Wind direction")
        ax.set_title(f"Wind Direction — {h}m",
                     fontsize=plot_cfg.get("tick_size", 8), loc="left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))

        all_stats.append({
            "variable": "wind_direction", "height_m": h,
            "palm_z_m": palm_z, **stats,
        })
        stats_list.append(all_stats[-1])
        panel += 1

    axes[0, 0].legend(loc="upper right", ncol=3, frameon=True,
                       fancybox=False, edgecolor="#CCCCCC", framealpha=0.9)
    axes[-1, 0].set_xlabel("Time (UTC)")
    fig.suptitle("Met Tower Wind: Obs vs PALM",
                  fontsize=plot_cfg.get("title_size", 10))
    fig.tight_layout()
    _save_figure(fig, outdir, "met_tower_wind", plot_cfg)
    return all_stats


def _circular_resample(series, rule):
    """Resample a circular variable (degrees) using vector mean."""
    if series is None:
        return None
    rad = np.deg2rad(series.values)
    sin_s = pd.Series(np.sin(rad), index=series.index)
    cos_s = pd.Series(np.cos(rad), index=series.index)
    sin_m = sin_s.resample(rule).mean()
    cos_m = cos_s.resample(rule).mean()
    mean_dir = np.rad2deg(np.arctan2(sin_m, cos_m)) % 360.0
    return mean_dir.dropna()


def _circular_statistics(obs_deg, sim_deg):
    """Compute circular statistics for wind direction comparison."""
    n = len(obs_deg)
    diff = np.mod(sim_deg - obs_deg + 180, 360) - 180
    rmse = np.sqrt(np.mean(diff ** 2))
    mbe = np.mean(diff)
    # Circular correlation using Jammalamadaka & Sarma
    o_rad = np.deg2rad(obs_deg)
    s_rad = np.deg2rad(sim_deg)
    o_mean = np.arctan2(np.mean(np.sin(o_rad)), np.mean(np.cos(o_rad)))
    s_mean = np.arctan2(np.mean(np.sin(s_rad)), np.mean(np.cos(s_rad)))
    sin_o = np.sin(o_rad - o_mean)
    sin_s = np.sin(s_rad - s_mean)
    denom = np.sqrt(np.sum(sin_o ** 2) * np.sum(sin_s ** 2))
    r = np.sum(sin_o * sin_s) / denom if denom > 0 else np.nan
    return {"rmse": rmse, "mbe": mbe, "r": r, "kge": np.nan,
            "nse": np.nan, "n_valid": n}


def plot_met_tower_radiation(tower_obs, palm_data, icon_data, stats_list,
                              plot_cfg, outdir, met_cfg):
    """Plot met tower radiation comparison: incoming and outgoing shortwave.

    Shows rsd (incoming) and rsu (outgoing) at available heights, plus
    derived net radiation Rn = rsd - rsu.
    """
    rsd_heights = sorted(set(tower_obs.get("shortwave_down", {}).keys()))
    rsu_heights = sorted(set(tower_obs.get("shortwave_up", {}).keys()))
    all_heights = sorted(set(rsd_heights + rsu_heights))
    if not all_heights:
        print("  [SKIP] No radiation observations loaded")
        return []

    avg_rule = met_cfg.get("temporal_averaging", {}).get("radiation", "10T")
    lw = plot_cfg.get("line_width", 1.5)
    # 2 panels per height (incoming + outgoing), plus 1 net radiation panel
    n_panels = len(rsd_heights) + len(rsu_heights) + (1 if rsd_heights and rsu_heights else 0)
    fig, axes = _met_tower_subplot_layout(n_panels, plot_cfg)
    all_stats = []
    panel = 0

    # Shortwave incoming panels
    net_obs_parts = {}
    net_palm_parts = {}
    for h in rsd_heights:
        ax = axes[panel, 0]
        obs_s = tower_obs["shortwave_down"].get(h)
        palm_s = palm_data.get(("rtm_rad_insw_down", h))

        obs_rs, palm_rs, stats = _align_tower_to_palm(obs_s, palm_s, avg_rule)

        palm_z = met_cfg["height_to_palm_z"].get(h, h)
        if obs_rs is not None:
            ax.plot(obs_rs.index, obs_rs.values, color=TOL_BLUE, lw=lw,
                    label=f"Tower {h}m", zorder=3)
            net_obs_parts.setdefault(h, {})["rsd"] = obs_rs
        if palm_rs is not None:
            ax.plot(palm_rs.index, palm_rs.values, color=TOL_RED, lw=lw,
                    ls="--", label=f"PALM z={palm_z}m", zorder=2)
            net_palm_parts.setdefault(h, {})["rsd"] = palm_rs

        _annotate_stats(ax, stats, plot_cfg)
        ax.set_ylabel(r"SW$\downarrow$ [W/m$^2$]")
        ax.set_title(f"Shortwave Incoming — {h}m",
                     fontsize=plot_cfg.get("tick_size", 8), loc="left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))

        all_stats.append({
            "variable": "shortwave_down", "height_m": h,
            "palm_z_m": palm_z, **stats,
        })
        stats_list.append(all_stats[-1])
        panel += 1

    # Shortwave outgoing panels
    for h in rsu_heights:
        ax = axes[panel, 0]
        obs_s = tower_obs["shortwave_up"].get(h)
        palm_s = palm_data.get(("rtm_rad_outsw_down", h))

        obs_rs, palm_rs, stats = _align_tower_to_palm(obs_s, palm_s, avg_rule)

        palm_z = met_cfg["height_to_palm_z"].get(h, h)
        if obs_rs is not None:
            ax.plot(obs_rs.index, obs_rs.values, color=TOL_BLUE, lw=lw,
                    label=f"Tower {h}m", zorder=3)
            net_obs_parts.setdefault(h, {})["rsu"] = obs_rs
        if palm_rs is not None:
            ax.plot(palm_rs.index, palm_rs.values, color=TOL_RED, lw=lw,
                    ls="--", label=f"PALM z={palm_z}m", zorder=2)
            net_palm_parts.setdefault(h, {})["rsu"] = palm_rs

        _annotate_stats(ax, stats, plot_cfg)
        ax.set_ylabel(r"SW$\uparrow$ [W/m$^2$]")
        ax.set_title(f"Shortwave Outgoing — {h}m",
                     fontsize=plot_cfg.get("tick_size", 8), loc="left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))

        all_stats.append({
            "variable": "shortwave_up", "height_m": h,
            "palm_z_m": palm_z, **stats,
        })
        stats_list.append(all_stats[-1])
        panel += 1

    # Net radiation panel (first height with both rsd and rsu)
    if rsd_heights and rsu_heights and panel < n_panels:
        ax = axes[panel, 0]
        h_net = rsd_heights[0]
        obs_rsd = net_obs_parts.get(h_net, {}).get("rsd")
        obs_rsu = net_obs_parts.get(h_net, {}).get("rsu")
        palm_rsd = net_palm_parts.get(h_net, {}).get("rsd")
        palm_rsu = net_palm_parts.get(h_net, {}).get("rsu")

        if obs_rsd is not None and obs_rsu is not None:
            common_obs = obs_rsd.index.intersection(obs_rsu.index)
            obs_net = obs_rsd.reindex(common_obs) - obs_rsu.reindex(common_obs)
            ax.plot(obs_net.index, obs_net.values, color=TOL_BLUE, lw=lw,
                    label=f"Tower {h_net}m", zorder=3)
        else:
            obs_net = None

        if palm_rsd is not None and palm_rsu is not None:
            common_palm = palm_rsd.index.intersection(palm_rsu.index)
            palm_net = palm_rsd.reindex(common_palm) - palm_rsu.reindex(common_palm)
            palm_z = met_cfg["height_to_palm_z"].get(h_net, h_net)
            ax.plot(palm_net.index, palm_net.values, color=TOL_RED, lw=lw,
                    ls="--", label=f"PALM z={palm_z}m", zorder=2)
        else:
            palm_net = None

        if obs_net is not None and palm_net is not None:
            common = obs_net.index.intersection(palm_net.index).sort_values()
            if len(common) >= 3:
                net_stats = compute_statistics(obs_net.reindex(common).values,
                                               palm_net.reindex(common).values)
            else:
                net_stats = {"rmse": np.nan, "mbe": np.nan, "r": np.nan,
                             "kge": np.nan, "nse": np.nan, "n_valid": 0}
        else:
            net_stats = {"rmse": np.nan, "mbe": np.nan, "r": np.nan,
                         "kge": np.nan, "nse": np.nan, "n_valid": 0}

        _annotate_stats(ax, net_stats, plot_cfg)
        ax.set_ylabel(r"R$_{net}$ [W/m$^2$]")
        ax.set_title(f"Net SW Radiation — {h_net}m",
                     fontsize=plot_cfg.get("tick_size", 8), loc="left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))

        all_stats.append({
            "variable": "net_radiation", "height_m": h_net,
            "palm_z_m": met_cfg["height_to_palm_z"].get(h_net, h_net),
            **net_stats,
        })
        stats_list.append(all_stats[-1])

    axes[0, 0].legend(loc="upper right", ncol=3, frameon=True,
                       fancybox=False, edgecolor="#CCCCCC", framealpha=0.9)
    axes[-1, 0].set_xlabel("Time (UTC)")
    fig.suptitle("Met Tower Radiation: Obs vs PALM",
                  fontsize=plot_cfg.get("title_size", 10))
    fig.tight_layout()
    _save_figure(fig, outdir, "met_tower_radiation", plot_cfg)
    return all_stats


def export_met_tower_statistics_csv(all_stats, outdir):
    """Export met tower comparison statistics to CSV."""
    if not all_stats:
        print("  No met tower statistics to export.")
        return

    outpath = Path(outdir) / "met_tower_comparison_statistics.csv"
    fieldnames = ["variable", "height_m", "palm_z_m",
                  "rmse", "mbe", "r", "kge", "nse", "n_valid"]
    with open(outpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames,
                                extrasaction="ignore")
        writer.writeheader()
        for row in all_stats:
            writer.writerow(row)
    print(f"  Exported: {outpath}")


# =============================================================================
# Phase 3: Comprehensive statistics, Taylor diagram, and heatmap
# =============================================================================

class TaylorDiagram(object):
    """Taylor diagram using mpl_toolkits.axisartist floating axes.

    Based on ycopin's implementation (MIT license, DOI: 10.5281/zenodo.5548061).
    """

    def __init__(self, refstd, fig=None, rect=111, label='_', srange=(0, 1.5)):
        self.refstd = refstd
        tr = PolarAxes.PolarTransform()

        # Correlation labels
        rlocs = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1])
        tlocs = np.arccos(rlocs)
        gl1 = GF.FixedLocator(tlocs)
        tf1 = GF.DictFormatter(dict(zip(tlocs, map(str, rlocs))))

        # Standard deviation axis
        gl2 = GF.MaxNLocator(nbins=6)

        ghelper = FA.GridHelperCurveLinear(
            tr,
            extremes=(0, np.pi / 2, srange[0], srange[1]),
            grid_locator1=gl1, tick_formatter1=tf1,
            grid_locator2=gl2,
        )

        if fig is None:
            fig = plt.gcf()

        ax = FA.FloatingSubplot(fig, rect, grid_helper=ghelper)
        fig.add_subplot(ax)

        # Adjust axes
        ax.axis["top"].set_axis_direction("bottom")
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("Correlation")

        ax.axis["left"].set_axis_direction("bottom")
        ax.axis["left"].label.set_text("Normalized standard deviation")

        ax.axis["right"].set_axis_direction("top")
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction(
            "bottom" if srange[0] == 0 else "left"
        )

        ax.axis["bottom"].toggle(ticklabels=False, label=False)

        self._ax = ax
        self.ax = ax.get_aux_axes(tr)

        # Plot reference point
        self.ax.plot([0], self.refstd, 'k*', ms=12, label=label, clip_on=False)

        # Collect sample points for contour drawing
        self.samplePoints = [
            self.ax.plot([0], self.refstd, 'k*', ms=12, clip_on=False)[0]
        ]

    def add_sample(self, stddev, corrcoef, *args, **kwargs):
        l, = self.ax.plot(np.arccos(corrcoef), stddev, *args, **kwargs)
        self.samplePoints.append(l)
        return l

    def add_contours(self, levels=5, **kwargs):
        rs, ts = np.meshgrid(
            np.linspace(0, self.ax.get_ylim()[1], 100),
            np.linspace(0, np.pi / 2, 100),
        )
        # Centered RMS difference
        rms = np.sqrt(self.refstd**2 + rs**2
                       - 2 * self.refstd * rs * np.cos(ts))
        contours = self.ax.contour(ts, rs, rms, levels, **kwargs)
        return contours

    def add_grid(self, *args, **kwargs):
        self._ax.grid(*args, **kwargs)


# Need PolarAxes import for TaylorDiagram
from matplotlib.projections import PolarAxes


def build_statistics_table(all_comparisons):
    """Aggregate all Phase 1 and Phase 2 comparison statistics into a DataFrame.

    Parameters
    ----------
    all_comparisons : list of dict
        Each dict has: variable, station_or_sensor, depth_or_position,
        rmse, mbe, r, kge, nse, n_valid, and optionally r_beta, r_gamma,
        obs_std, sim_std.

    Returns
    -------
    stats_df : pandas.DataFrame
        One row per comparison pair with all metrics.
    """
    if not all_comparisons:
        return pd.DataFrame()

    stats_df = pd.DataFrame(all_comparisons)

    # Ensure column order
    cols = ["variable", "station_or_sensor", "depth_or_position",
            "rmse", "mbe", "r", "kge", "nse", "n_valid"]
    extra = [c for c in ["r_beta", "r_gamma", "obs_std", "sim_std"]
             if c in stats_df.columns]
    stats_df = stats_df[[c for c in cols + extra if c in stats_df.columns]]

    return stats_df


def plot_taylor_diagram(all_comparisons, plot_cfg, outdir):
    """Taylor diagram showing all variables with normalized std dev and CRMSD.

    Parameters
    ----------
    all_comparisons : list of dict
        Each dict has: variable, station_or_sensor, depth_or_position,
        r, obs_std, sim_std.
    plot_cfg : dict
        Plot settings.
    outdir : str
        Output directory.
    """
    # Filter to valid entries
    valid = [c for c in all_comparisons
             if not np.isnan(c.get("r", np.nan))
             and not np.isnan(c.get("obs_std", np.nan))
             and not np.isnan(c.get("sim_std", np.nan))
             and c.get("obs_std", 0) > 0]

    if not valid:
        print("  Skipping Taylor diagram — no valid comparisons.")
        return

    # Variable type -> color mapping
    var_colors = {
        "soil_moisture": TOL_BLUE,
        "soil_temp": TOL_CYAN,
        "air_temp": TOL_GREEN,
        "leaf_temp": TOL_RED,
    }

    # Station/sensor -> marker mapping
    marker_pool = ["o", "s", "^", "D", "v", "P", "*", "X"]
    station_markers = {}
    marker_idx = 0

    fig_w = mm_to_inches(plot_cfg["figure_width_mm"])
    fig = plt.figure(figsize=(fig_w, fig_w * 0.85))

    # Reference std = 1.0 (normalized)
    dia = TaylorDiagram(1.0, fig=fig, rect=111, label="Observations",
                        srange=(0, 1.65))

    legend_handles = []
    legend_labels = []
    seen_var_colors = set()
    seen_markers = set()

    for comp in valid:
        variable = comp["variable"]
        station = str(comp["station_or_sensor"])
        r_val = comp["r"]
        obs_std = comp["obs_std"]
        sim_std = comp["sim_std"]

        # Normalize std dev by observed std dev
        norm_std = sim_std / obs_std

        color = var_colors.get(variable, TOL_GREY)

        if station not in station_markers:
            station_markers[station] = marker_pool[marker_idx % len(marker_pool)]
            marker_idx += 1
        marker = station_markers[station]

        dia.add_sample(norm_std, r_val, marker=marker, ms=7,
                       color=color, mec=color, mew=0.5, clip_on=False,
                       zorder=5)

        # Legend entries for variable color
        if variable not in seen_var_colors:
            seen_var_colors.add(variable)
            label_map = {
                "soil_moisture": "Soil moisture",
                "soil_temp": "Soil temperature",
                "air_temp": "Air temperature",
                "leaf_temp": "Leaf temperature",
            }
            h = Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                       ms=7, label=label_map.get(variable, variable))
            legend_handles.append(h)
            legend_labels.append(label_map.get(variable, variable))

        # Legend entries for station marker
        if station not in seen_markers:
            seen_markers.add(station)
            h = Line2D([0], [0], marker=marker, color="w",
                       markerfacecolor="gray", ms=7, label=station)
            legend_handles.append(h)
            legend_labels.append(station)

    # Add centered RMSE contours
    contours = dia.add_contours(levels=5, colors="0.5", linewidths=0.5,
                                 linestyles="--")
    dia.ax.clabel(contours, inline=True, fontsize=6, fmt="%.1f")

    # Add std dev = 1.0 reference arc
    theta = np.linspace(0, np.pi / 2, 100)
    dia.ax.plot(theta, np.ones_like(theta), "k--", lw=0.8, alpha=0.5)

    dia.add_grid(True, linestyle=":", alpha=0.3)

    # Build legend
    fig.legend(handles=legend_handles, labels=legend_labels,
               loc="lower right", fontsize=plot_cfg.get("legend_size", 7.5),
               frameon=True, fancybox=False, edgecolor="#CCCCCC",
               ncol=2, bbox_to_anchor=(0.95, 0.05))

    fig.suptitle("Taylor Diagram: All Variables (Normalized)",
                 fontsize=plot_cfg.get("title_size", 10))
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    _save_figure(fig, outdir, "taylor_diagram", plot_cfg)


def plot_statistics_heatmap(stats_table, plot_cfg, outdir):
    """Heatmap of KGE values across all variable/station/depth combinations.

    Parameters
    ----------
    stats_table : pandas.DataFrame
        Statistics table from build_statistics_table.
    plot_cfg : dict
        Plot settings.
    outdir : str
        Output directory.
    """
    if stats_table.empty or "kge" not in stats_table.columns:
        print("  Skipping statistics heatmap — no data.")
        return

    # Build row label: variable + station_or_sensor
    stats_table = stats_table.copy()
    stats_table["row_label"] = (stats_table["variable"].astype(str) + " | "
                                 + stats_table["station_or_sensor"].astype(str))
    stats_table["col_label"] = stats_table["depth_or_position"].astype(str)

    # Pivot to matrix
    pivot = stats_table.pivot_table(
        values="kge", index="row_label", columns="col_label", aggfunc="first"
    )

    if pivot.empty:
        print("  Skipping statistics heatmap — empty pivot.")
        return

    fig_w = mm_to_inches(plot_cfg["figure_width_mm"])
    n_rows, n_cols = pivot.shape
    fig_h = max(fig_w * 0.4, mm_to_inches(30 + n_rows * 12))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Diverging colormap: red (poor) -> white (neutral) -> green (good)
    from matplotlib.colors import TwoSlopeNorm
    vmin = min(pivot.min().min(), -0.41)
    vmax = max(pivot.max().max(), 1.0)
    if np.isnan(vmin):
        vmin = -1.0
    if np.isnan(vmax):
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "kge_diverging",
        [(0.0, TOL_RED), (0.5, "#FFFFFF"), (1.0, TOL_GREEN)],
    )

    data = pivot.values.astype(float)
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

    # Annotate cells
    for i in range(n_rows):
        for j in range(n_cols):
            val = data[i, j]
            if not np.isnan(val):
                text_color = "white" if abs(val) > 0.7 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=plot_cfg.get("tick_size", 8) * 0.9,
                        color=text_color)

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right",
                        fontsize=plot_cfg.get("tick_size", 8))
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(pivot.index,
                        fontsize=plot_cfg.get("tick_size", 8))

    ax.set_xlabel("Depth / Position")
    ax.set_ylabel("Variable | Station/Sensor")
    ax.set_title("KGE Performance Heatmap",
                  fontsize=plot_cfg.get("title_size", 10))

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("KGE", fontsize=plot_cfg.get("label_size", 9))

    # Reference line at KGE = -0.41 (better than mean, Knoben et al. 2019)
    cbar.ax.axhline(y=-0.41, color="black", linewidth=1.0, linestyle="--")
    cbar.ax.text(0.5, -0.41, " -0.41", transform=cbar.ax.get_yaxis_transform(),
                 fontsize=6, va="center")

    fig.tight_layout()
    _save_figure(fig, outdir, "statistics_heatmap", plot_cfg)


def export_statistics_latex(stats_df, outdir):
    """Export statistics table as LaTeX.

    Parameters
    ----------
    stats_df : pandas.DataFrame
        Statistics table.
    outdir : str
        Output directory.
    """
    if stats_df.empty:
        print("  No statistics to export as LaTeX.")
        return

    outpath = Path(outdir) / "britz_comparison_statistics.tex"

    # Format numeric columns
    fmt_df = stats_df.copy()
    for col in ["rmse", "mbe"]:
        if col in fmt_df.columns:
            fmt_df[col] = fmt_df[col].apply(
                lambda x: f"{x:.4f}" if not np.isnan(x) else "--")
    for col in ["r", "kge", "nse"]:
        if col in fmt_df.columns:
            fmt_df[col] = fmt_df[col].apply(
                lambda x: f"{x:.3f}" if not np.isnan(x) else "--")
    if "n_valid" in fmt_df.columns:
        fmt_df["n_valid"] = fmt_df["n_valid"].apply(
            lambda x: str(int(x)) if not np.isnan(x) else "--")

    # Select display columns
    display_cols = ["variable", "station_or_sensor", "depth_or_position",
                    "rmse", "mbe", "r", "kge", "nse", "n_valid"]
    display_cols = [c for c in display_cols if c in fmt_df.columns]
    fmt_df = fmt_df[display_cols]

    # Column headers for LaTeX
    col_headers = {
        "variable": "Variable",
        "station_or_sensor": "Station/Sensor",
        "depth_or_position": "Depth/Position",
        "rmse": "RMSE",
        "mbe": "MBE",
        "r": r"$r$",
        "kge": "KGE",
        "nse": "NSE",
        "n_valid": r"$n$",
    }
    fmt_df.columns = [col_headers.get(c, c) for c in fmt_df.columns]

    # Build LaTeX table
    n_cols = len(fmt_df.columns)
    col_fmt = "l" * 3 + "r" * (n_cols - 3)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Comprehensive validation statistics: "
                 r"PALM vs.\ Britz observations.}")
    lines.append(r"\label{tab:britz_statistics}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{" + col_fmt + "}")
    lines.append(r"\toprule")
    lines.append(" & ".join(fmt_df.columns) + r" \\")
    lines.append(r"\midrule")
    for _, row in fmt_df.iterrows():
        # Escape underscores in text columns
        vals = []
        for v in row.values:
            s = str(v).replace("_", r"\_")
            vals.append(s)
        lines.append(" & ".join(vals) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(outpath, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Exported LaTeX: {outpath}")


# =============================================================================
# Main orchestration
# =============================================================================

def main():
    """Run the full BritzPlot analysis pipeline.

    Parses CLI arguments, loads config, sets up matplotlib, and dispatches
    all enabled plot functions based on analysis_toggles.
    """
    parser = argparse.ArgumentParser(
        description="BritzPlot: PALM validation against Britz observations"
    )
    parser.add_argument(
        "--config", default="config.yml",
        help="Path to YAML config file (default: config.yml)"
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = script_dir / config_path

    cfg = load_config(config_path)
    setup_matplotlib(cfg["plot"])

    paths = cfg["paths"]
    plot_dir = paths["output_dir"]
    plot_cfg = cfg["plot"]
    toggles = cfg.get("analysis_toggles", {})

    print("BritzPlot — PALM Model Validation Against Britz Observations")
    print(f"Output directory: {plot_dir}")
    print(f"Time window: {cfg['time']['sim_start']} to {cfg['time']['sim_end']}")
    print()

    # Unified comparison list for Phase 3 (Taylor diagram, heatmap, stats table)
    all_comparisons = []

    # ---- Phase 1: Soil comparison ----
    soil_toggles = [
        toggles.get("soil_moisture_timeseries", False),
        toggles.get("soil_temperature_timeseries", False),
        toggles.get("soil_vertical_profile_mean", False),
        toggles.get("soil_station_comparison", False),
    ]

    if any(soil_toggles):
        print("Phase 1: Loading soil data ...")
        soil_cfg = cfg["soil"]
        palm_cfg = cfg["palm"]
        station_cfg = soil_cfg["stations"]
        dz_soil = palm_cfg["dz_soil"]
        station_ids = list(station_cfg.keys())

        # Inject dz_soil into plot_cfg for helper access
        plot_cfg["_dz_soil"] = dz_soil

        # Load PALM data
        print("  Loading PALM restart series ...")
        palm_3d = load_palm_restart_series(
            paths["palm_output_dir"], paths["palm_job_name"],
            soil_cfg["palm_file_type"], soil_cfg["palm_domain"],
            variables=["m_soil", "t_soil"],
        )

        # Build PALM datetime index
        palm_idx, _ = align_time_axes(
            palm_3d["time"], pd.DatetimeIndex([]), cfg["time"]["reference_time"]
        )

        # Extract PALM soil at each station location
        print("  Extracting PALM soil at station locations ...")
        palm_soil = {}
        for sid, scfg in station_cfg.items():
            palm_soil[sid] = extract_palm_soil_at_location(
                palm_3d, scfg["lat"], scfg["lon"],
                palm_cfg["origin_E"], palm_cfg["origin_N"],
                palm_cfg["dx"], palm_cfg["dy"],
            )

        # Load soil observations
        print("  Loading soil observations ...")
        obs_data = load_soil_observations(
            paths["soil_obs_csv"], station_ids,
            cfg["time"]["sim_start"], cfg["time"]["sim_end"],
        )

        # Depth matching
        obs_depths = soil_cfg.get("obs_depths_cm", [10, 20, 30, 40, 50,
                                                     100, 200, 300, 400, 460])
        depth_mapping = match_obs_to_palm_depths(obs_depths, dz_soil)

        # Filter to requested layers
        layers_filter = soil_cfg.get("layers", "all_layers")
        if layers_filter != "all_layers" and isinstance(layers_filter, list):
            depth_mapping = {d: idx for d, idx in depth_mapping.items()
                             if d in layers_filter}

        all_soil_stats = []
        print()

    if toggles.get("soil_moisture_timeseries", False):
        print("  Plotting soil moisture timeseries ...")
        stats = plot_soil_moisture_timeseries(
            obs_data, palm_soil, station_cfg, depth_mapping,
            palm_idx, plot_cfg, plot_dir,
        )
        all_soil_stats.extend(stats)

    if toggles.get("soil_temperature_timeseries", False):
        print("  Plotting soil temperature timeseries ...")
        plot_soil_temperature_timeseries(
            palm_soil, station_cfg, dz_soil, palm_idx, plot_cfg, plot_dir,
        )

    if toggles.get("soil_vertical_profile_mean", False):
        print("  Plotting soil vertical profiles ...")
        plot_soil_vertical_profile_mean(
            obs_data, palm_soil, station_cfg, depth_mapping, dz_soil,
            palm_idx, plot_cfg, plot_dir,
        )

    if toggles.get("soil_station_comparison", False):
        print("  Plotting soil station comparison ...")
        plot_soil_station_comparison(
            obs_data, palm_soil, station_cfg, depth_mapping,
            palm_idx, plot_cfg, plot_dir,
        )

    if any(soil_toggles):
        export_soil_statistics_csv(all_soil_stats, plot_dir)

    # Collect Phase 1 comparisons in unified format for Phase 3
    if any(soil_toggles) and all_soil_stats:
        for s in all_soil_stats:
            # Recompute obs_std and sim_std for Taylor diagram
            comp = {
                "variable": "soil_moisture",
                "station_or_sensor": SPECIES_LABELS.get(s["station"],
                                                         s["station"]),
                "depth_or_position": f"{s['depth_cm']} cm",
                "rmse": s["rmse"],
                "mbe": s["mbe"],
                "r": s["r"],
                "kge": s["kge"],
                "nse": s["nse"],
                "n_valid": s["n_valid"],
                "r_beta": s.get("r_beta", np.nan),
                "r_gamma": s.get("r_gamma", np.nan),
                "obs_std": np.nan,
                "sim_std": np.nan,
            }
            all_comparisons.append(comp)

    # We need obs_std/sim_std for Taylor diagram — recompute from aligned data
    if any(soil_toggles) and all_soil_stats:
        for comp_dict, raw_s in zip(
            [c for c in all_comparisons if c["variable"] == "soil_moisture"],
            all_soil_stats,
        ):
            sid = raw_s["station"]
            depth_cm = raw_s["depth_cm"]
            layer_idx = raw_s["palm_layer_index"]
            obs_df_s = obs_data.get(sid)
            palm_st = palm_soil.get(sid)
            if obs_df_s is None or palm_st is None:
                continue
            if depth_cm in obs_df_s.columns:
                obs_ts = obs_df_s[depth_cm].dropna()
            else:
                continue
            palm_m = palm_st["m_soil"][:, layer_idx]
            palm_series = pd.Series(
                np.ma.filled(palm_m, np.nan), index=palm_idx
            )
            common = obs_ts.index.intersection(palm_series.index).sort_values()
            if len(common) >= 3:
                oa = obs_ts.reindex(common).values
                sa = palm_series.reindex(common).values
                valid_mask = ~(np.isnan(oa) | np.isnan(sa))
                if valid_mask.sum() >= 3:
                    comp_dict["obs_std"] = float(np.std(oa[valid_mask]))
                    comp_dict["sim_std"] = float(np.std(sa[valid_mask]))

    # ---- Phase 2: Leaf/Air temperature ----
    leaf_toggles = [
        toggles.get("leaf_air_temp_timeseries", False),
        toggles.get("leaf_temp_scatter", False),
        toggles.get("leaf_air_temp_diurnal", False),
        toggles.get("tree_id_averaged_comparison", False),
    ]

    if any(leaf_toggles):
        print("\nPhase 2: Loading leaf/air temperature data ...")
        lt_cfg = cfg["leaf_temp"]

        obs_df, sensor_meta = load_toa5_data(
            paths["leaf_temp_dat"],
            paths["sensor_metadata_csv"],
            sensors=lt_cfg.get("sensors", "all_sensors"),
            exclude_sensors=lt_cfg.get("exclude_sensors"),
            exclude_after=lt_cfg.get("exclude_after"),
            time_start=cfg["time"]["sim_start"],
            time_end=cfg["time"]["sim_end"],
        )

        print("  Loading PALM 3D data for ta ...")
        palm_3d_ta = load_palm_restart_series(
            paths["palm_output_dir"], paths["palm_job_name"],
            "av_3d", lt_cfg["palm_domain"],
            variables=["ta"],
        )

        palm_ta_dict = {}
        tree_configs = lt_cfg["tree_ids"]
        static_path = paths["static_driver"]

        for tree_name, tcfg in tree_configs.items():
            print(f"  Processing {tree_name} "
                  f"(PALM tree_id {tcfg['palm_tree_id']}) ...")
            tree_mask, _ = build_tree_id_mask(static_path,
                                               tcfg["palm_tree_id"])
            palm_ta_dict[tree_name] = extract_palm_ta_at_tree(
                palm_3d_ta, tree_mask, cfg["time"]["reference_time"]
            )

        all_leaf_stats = []
        print()

    if toggles.get("leaf_air_temp_timeseries", False):
        print("  Plotting leaf/air temperature timeseries ...")
        stats = plot_leaf_air_temp_timeseries(
            obs_df, palm_ta_dict, tree_configs, sensor_meta,
            plot_cfg, plot_dir,
        )
        all_leaf_stats.extend(stats)

    if toggles.get("leaf_temp_scatter", False):
        print("  Plotting leaf temperature scatter ...")
        plot_leaf_temp_scatter(
            obs_df, palm_ta_dict, tree_configs, sensor_meta,
            plot_cfg, plot_dir,
        )

    if toggles.get("leaf_air_temp_diurnal", False):
        print("  Plotting leaf/air diurnal composite ...")
        plot_leaf_air_diurnal(
            obs_df, palm_ta_dict, tree_configs, sensor_meta,
            plot_cfg, plot_dir,
        )

    if toggles.get("tree_id_averaged_comparison", False):
        print("  Plotting tree-averaged comparison ...")
        tree_stats = plot_tree_averaged_comparison(
            obs_df, palm_ta_dict, tree_configs, sensor_meta,
            plot_cfg, plot_dir,
        )
        all_leaf_stats.extend(tree_stats)

    if any(leaf_toggles):
        export_leaf_temp_statistics_csv(all_leaf_stats, plot_dir)

    # Collect Phase 2 comparisons into unified format for Phase 3
    if any(leaf_toggles) and all_leaf_stats:
        for s in all_leaf_stats:
            tree_name = s.get("tree_id", "")
            sid = s.get("sensor_id", "")
            pos = s.get("position_code", "")
            comp = {
                "variable": "leaf_temp",
                "station_or_sensor": f"{tree_name}/S{sid}",
                "depth_or_position": pos,
                "rmse": s["rmse"],
                "mbe": s["mbe"],
                "r": s["r"],
                "kge": s["kge"],
                "nse": s["nse"],
                "n_valid": s["n_valid"],
                "r_beta": np.nan,
                "r_gamma": np.nan,
                "obs_std": np.nan,
                "sim_std": np.nan,
            }
            all_comparisons.append(comp)

        # Compute obs_std/sim_std for leaf temp comparisons
        for comp_dict in [c for c in all_comparisons
                          if c["variable"] == "leaf_temp"]:
            parts = comp_dict["station_or_sensor"].split("/S")
            if len(parts) != 2:
                continue
            tree_name_c = parts[0]
            sid_str = parts[1]
            palm_ta = palm_ta_dict.get(tree_name_c)
            if palm_ta is None:
                continue

            if sid_str == "avg":
                # Tree-averaged: average all sensors
                tcfg_c = tree_configs.get(tree_name_c)
                if tcfg_c is None:
                    continue
                leaf_list = []
                for s_id in tcfg_c["sensors"]:
                    row = sensor_meta[sensor_meta["sensor_id"] == s_id]
                    if row.empty:
                        continue
                    lc = row.iloc[0]["leaf_col"]
                    if lc in obs_df.columns:
                        leaf_list.append(obs_df[lc])
                if not leaf_list:
                    continue
                obs_leaf = pd.concat(leaf_list, axis=1).mean(axis=1)
            else:
                try:
                    s_id = int(sid_str)
                except ValueError:
                    continue
                row = sensor_meta[sensor_meta["sensor_id"] == s_id]
                if row.empty:
                    continue
                lc = row.iloc[0]["leaf_col"]
                if lc not in obs_df.columns:
                    continue
                obs_leaf = obs_df[lc].dropna()

            aligned = _resample_obs_to_palm(obs_leaf, palm_ta.index)
            valid_mask = ~(aligned.isna() | palm_ta.isna())
            if valid_mask.sum() >= 3:
                oa = aligned[valid_mask].values
                sa = palm_ta[valid_mask].values
                comp_dict["obs_std"] = float(np.std(oa))
                comp_dict["sim_std"] = float(np.std(sa))

    # ---- Phase 3: Statistics ----
    phase3_toggles = [
        toggles.get("statistics_summary_table", False),
        toggles.get("taylor_diagram", False),
    ]

    if any(phase3_toggles) and all_comparisons:
        print("\nPhase 3: Statistical Summary & Advanced Metrics ...")

    if toggles.get("statistics_summary_table", False):
        print("  Building comprehensive statistics table ...")
        stats_df = build_statistics_table(all_comparisons)
        if not stats_df.empty:
            csv_path = Path(plot_dir) / "britz_comparison_statistics.csv"
            stats_df.to_csv(csv_path, index=False)
            print(f"  Exported CSV: {csv_path}")
            export_statistics_latex(stats_df, plot_dir)
            print("  Plotting statistics heatmap ...")
            plot_statistics_heatmap(stats_df, plot_cfg, plot_dir)

    if toggles.get("taylor_diagram", False):
        print("  Plotting Taylor diagram ...")
        plot_taylor_diagram(all_comparisons, plot_cfg, plot_dir)

    # ---- Phase 5: Met tower comparison ----
    met_toggles = [
        toggles.get("met_tower_temperature", False),
        toggles.get("met_tower_humidity", False),
        toggles.get("met_tower_wind", False),
        toggles.get("met_tower_radiation", False),
    ]

    met_cfg = cfg.get("met_tower", {})
    if any(met_toggles) and met_cfg.get("enabled", False):
        print("\nPhase 5: Loading met tower data ...")
        tower_data_dir = Path(met_cfg["tower_data_dir"])
        t_start = cfg["time"]["sim_start"]
        t_end = cfg["time"]["sim_end"]

        # Load tower CSVs grouped by variable
        tower_obs = {}
        for var_name, vcfg in met_cfg.get("variables", {}).items():
            tower_obs[var_name] = {}
            for fname, height in zip(vcfg["files"], vcfg["heights_m"]):
                csv_path = tower_data_dir / fname
                s = load_tower_csv(csv_path, t_start, t_end)
                if s is not None:
                    tower_obs[var_name][height] = s
                    print(f"    {var_name} {height}m: {len(s)} obs")

        # Load PALM data at met tower grid cell
        palm_vars = ["ta", "rh", "wspeed", "wdir",
                     "rtm_rad_insw_down", "rtm_rad_outsw_down"]
        print("  Loading PALM 3D data for met tower variables ...")
        palm_3d_met = load_palm_restart_series(
            paths["palm_output_dir"], paths["palm_job_name"],
            "av_3d", cfg.get("soil", {}).get("palm_domain", "N02"),
            variables=palm_vars,
        )

        palm_cfg_m = cfg["palm"]
        ix, iy = calculate_met_tower_grid_cell(
            met_cfg["tower_lat"], met_cfg["tower_lon"],
            palm_3d_met, palm_cfg_m["origin_E"], palm_cfg_m["origin_N"],
        )

        palm_tower = extract_palm_at_tower(
            palm_3d_met, ix, iy, met_cfg["height_to_palm_z"],
            palm_vars, cfg["time"]["reference_time"],
        )

        # Load ICON-D2 boundary data if enabled
        icon_data = None
        icon_cfg = met_cfg.get("icon_d2", {})
        if icon_cfg.get("enabled", False) and icon_cfg.get("dynamic_driver_path"):
            print("  Loading ICON-D2 boundary forcing ...")
            icon_data = {}
            for var_name in ("air_temperature", "relative_humidity",
                             "wind_speed", "wind_direction"):
                for h in met_cfg["height_to_palm_z"].keys():
                    h = int(h) if isinstance(h, str) else h
                    s = load_icon_d2_boundary_mean(
                        icon_cfg["dynamic_driver_path"], var_name, h,
                        t_start, t_end,
                    )
                    if s is not None:
                        icon_data[(var_name, h)] = s

        all_met_stats = []
        print()

    if toggles.get("met_tower_temperature", False) and met_cfg.get("enabled", False):
        print("  Plotting met tower temperature ...")
        plot_met_tower_temperature(
            tower_obs, palm_tower, icon_data, all_met_stats,
            plot_cfg, plot_dir, met_cfg,
        )

    if toggles.get("met_tower_humidity", False) and met_cfg.get("enabled", False):
        print("  Plotting met tower humidity ...")
        plot_met_tower_humidity(
            tower_obs, palm_tower, icon_data, all_met_stats,
            plot_cfg, plot_dir, met_cfg,
        )

    if toggles.get("met_tower_wind", False) and met_cfg.get("enabled", False):
        print("  Plotting met tower wind ...")
        plot_met_tower_wind(
            tower_obs, palm_tower, icon_data, all_met_stats,
            plot_cfg, plot_dir, met_cfg,
        )

    if toggles.get("met_tower_radiation", False) and met_cfg.get("enabled", False):
        print("  Plotting met tower radiation ...")
        plot_met_tower_radiation(
            tower_obs, palm_tower, icon_data, all_met_stats,
            plot_cfg, plot_dir, met_cfg,
        )

    if any(met_toggles) and met_cfg.get("enabled", False):
        export_met_tower_statistics_csv(all_met_stats, plot_dir)

        # Collect met tower stats into all_comparisons for Phase 3
        for s in all_met_stats:
            comp = {
                "variable": s["variable"],
                "station_or_sensor": f"Tower {s['height_m']}m",
                "depth_or_position": f"z={s['palm_z_m']}m",
                "rmse": s["rmse"],
                "mbe": s["mbe"],
                "r": s["r"],
                "kge": s["kge"],
                "nse": s["nse"],
                "n_valid": s["n_valid"],
                "r_beta": np.nan,
                "r_gamma": np.nan,
                "obs_std": np.nan,
                "sim_std": np.nan,
            }
            all_comparisons.append(comp)

    # ---- Phase 4: Sap flow / dendrometer ----
    if toggles.get("sap_flow_timeseries", False):
        print("  [SKIP] sap_flow_timeseries — not yet implemented")

    if toggles.get("dendrometer_twd", False):
        print("  [SKIP] dendrometer_twd — not yet implemented")

    if toggles.get("mds_vs_transpiration", False):
        print("  [SKIP] mds_vs_transpiration — not yet implemented")

    if toggles.get("cross_correlation_lag", False):
        print("  [SKIP] cross_correlation_lag — not yet implemented")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
