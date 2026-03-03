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

    # ---- Phase 2: Leaf/Air temperature ----
    if toggles.get("leaf_air_temp_timeseries", False):
        print("  [SKIP] leaf_air_temp_timeseries — not yet implemented")

    if toggles.get("leaf_temp_scatter", False):
        print("  [SKIP] leaf_temp_scatter — not yet implemented")

    if toggles.get("leaf_air_temp_diurnal", False):
        print("  [SKIP] leaf_air_temp_diurnal — not yet implemented")

    if toggles.get("tree_id_averaged_comparison", False):
        print("  [SKIP] tree_id_averaged_comparison — not yet implemented")

    # ---- Phase 3: Statistics ----
    if toggles.get("statistics_summary_table", False):
        print("  [SKIP] statistics_summary_table — not yet implemented")

    if toggles.get("taylor_diagram", False):
        print("  [SKIP] taylor_diagram — not yet implemented")

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
