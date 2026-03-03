"""
Britz soil moisture analysis for PALM LSM initial conditions.

Extracts soil moisture from Britz lysimeter observations at the PALM simulation
origin time, converts units, generates initial condition profiles, and produces
diagnostic plots.

Usage:
    python britz_soil_analysis.py [--config config.yml]
"""

import argparse
import os
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import pyreadr
import yaml
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(path="config.yml"):
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def load_observations(cfg):
    """Load RData and return dict of DataFrames keyed by station ID."""
    rdata_path = Path(cfg["data"]["rdata_path"])
    result = pyreadr.read_r(str(rdata_path))

    station_map = {
        "wc_data_9801_02": "9801_02",
        "wc_data_9801_04": "9801_04",
        "wc_data_9801_08": "9801_08",
    }

    dfs = {}
    for rname, station_id in station_map.items():
        df = result[rname].copy()
        df["depth_cm"] = -df["Messtiefe_in_cm"]
        df["station"] = station_id
        dfs[station_id] = df

    return dfs


def extract_initial_conditions(dfs, cfg):
    """Extract replicate-averaged soil moisture at target time for each station.

    Returns dict[station_id] -> DataFrame with columns [depth_cm, wc_vol_pct, wc_m3m3].
    Also includes 'mean' key for the cross-station average.
    """
    target_time = pd.Timestamp(cfg["palm"]["target_time"])
    obs_depths = cfg["palm"]["obs_depths_cm"]

    profiles = {}
    for station_id, df in dfs.items():
        mask = df["UTC_time"] == target_time
        subset = df[mask]
        by_depth = (
            subset.groupby("depth_cm")["hourly_mean"]
            .mean()
            .reindex(obs_depths)
            .reset_index()
        )
        by_depth.columns = ["depth_cm", "wc_vol_pct"]
        by_depth["wc_m3m3"] = by_depth["wc_vol_pct"] / 100.0
        profiles[station_id] = by_depth

    # Cross-station mean
    all_wc = pd.concat(
        [p[["depth_cm", "wc_vol_pct"]].set_index("depth_cm") for p in profiles.values()],
        axis=1,
    )
    mean_profile = pd.DataFrame({
        "depth_cm": obs_depths,
        "wc_vol_pct": all_wc.mean(axis=1).values,
    })
    mean_profile["wc_m3m3"] = mean_profile["wc_vol_pct"] / 100.0
    profiles["mean"] = mean_profile

    return profiles


def extract_timeseries(dfs, cfg):
    """Extract replicate-averaged soil moisture time series for each station.

    Returns dict[station_id] -> DataFrame pivoted with depth columns.
    """
    t_start = pd.Timestamp(cfg["palm"]["timeseries_start"])
    t_end = pd.Timestamp(cfg["palm"]["timeseries_end"])
    obs_depths = cfg["palm"]["obs_depths_cm"]

    timeseries = {}
    for station_id, df in dfs.items():
        mask = (df["UTC_time"] >= t_start) & (df["UTC_time"] <= t_end)
        subset = df[mask].copy()

        # Average across replicates per (time, depth)
        avg = (
            subset.groupby(["UTC_time", "depth_cm"])["hourly_mean"]
            .mean()
            .reset_index()
        )
        pivot = avg.pivot(index="UTC_time", columns="depth_cm", values="hourly_mean")
        pivot = pivot[obs_depths]  # ensure column order
        timeseries[station_id] = pivot

    return timeseries


# ---------------------------------------------------------------------------
# PALM soil layer computation
# ---------------------------------------------------------------------------

def compute_palm_layers(cfg):
    """Compute PALM soil layer thicknesses and centers from observation depths.

    Strategy: layer bottom boundaries are placed at the observation depths.
    dz_soil(k) = obs_depth(k) - obs_depth(k-1), with obs_depth(0) = 0.
    """
    obs_depths_m = np.array(cfg["palm"]["obs_depths_cm"]) / 100.0
    boundaries = np.concatenate([[0.0], obs_depths_m])
    dz_soil = np.diff(boundaries)
    centers = boundaries[:-1] + dz_soil / 2.0

    return {
        "dz_soil": dz_soil,
        "boundaries": boundaries,
        "centers_m": centers,
        "bottoms_m": obs_depths_m,
        "n_layers": len(dz_soil),
    }


def compute_soil_temperature(layers, cfg):
    """Compute soil temperature at layer centers using a damped wave model.

    Fitted to Britz tensiometer observations at 15, 40, 100 cm on 2024-09-04.
    Model: T(z) = T_mean + A * exp(-z/d) * cos(phase - z/d)

    Parameters fitted with kappa=6e-7 m²/s (moist sand):
      T_mean = 11.0 °C (annual mean from SMT100 7-year record)
      A = 7.09 °C (surface amplitude, fitted to tensiometer obs)
      d = 2.46 m (damping depth from kappa)
    """
    T_mean = 11.0  # °C
    kappa = 6e-7   # m²/s, thermal diffusivity for moist sand
    omega = 2 * np.pi / (365.25 * 86400)
    d = np.sqrt(2 * kappa / omega)  # 2.46 m

    doy = 248  # 2024-09-04
    t_peak = 205  # late July
    phase = 2 * np.pi / 365.25 * (doy - t_peak)

    # Fit amplitude to tensiometer observations
    obs_z = np.array([0.15, 0.40, 1.00])
    obs_T = np.array([15.7, 16.0, 16.1])  # °C, tensiometer

    def T_model(z, A):
        return T_mean + A * np.exp(-z / d) * np.cos(phase - z / d)

    from scipy.optimize import curve_fit
    popt, _ = curve_fit(T_model, obs_z, obs_T, p0=[8.0])
    A_fit = popt[0]

    T_profile_C = T_model(layers["centers_m"], A_fit)
    T_profile_K = T_profile_C + 273.15

    return T_profile_K


def compute_root_fraction(layers):
    """Redistribute root fraction to new layers.

    Assumes roots concentrated in top 50cm with exponential decay.
    Uses a realistic temperate deciduous forest distribution.
    """
    n = layers["n_layers"]
    fractions = np.zeros(n)
    # Exponential root distribution: most roots in top 50cm
    # Based on Jackson (1996) temperate deciduous beta=0.966
    beta = 0.966
    for k in range(n):
        top = layers["boundaries"][k]
        bottom = layers["boundaries"][k + 1]
        # Cumulative root fraction = 1 - beta^depth(cm)
        top_cm = top * 100
        bottom_cm = bottom * 100
        fractions[k] = (1 - beta ** bottom_cm) - (1 - beta ** top_cm)

    # Normalise to sum to 1
    fractions /= fractions.sum()
    return fractions


# ---------------------------------------------------------------------------
# P3D parameter block generation
# ---------------------------------------------------------------------------

def format_p3d_block(label, layers, soil_moisture, soil_temp, root_frac, deep_temp):
    """Format a PALM &land_surface_parameters block."""
    def fmt_array(arr, fmt=".6f"):
        return ", ".join(f"{v:{fmt}}" for v in arr)

    block = textwrap.dedent(f"""\
    ! --- Initial conditions from Britz lysimeter: {label} ---
    ! Observation time: 2024-09-04 00:00:00 UTC
    ! Unit conversion: vol-% / 100 -> m3/m3
    ! Layer bottoms (m): {fmt_array(layers['bottoms_m'], '.2f')}
    ! Layer centers (m): {fmt_array(layers['centers_m'], '.4f')}
    dz_soil = {fmt_array(layers['dz_soil'], '.2f')},
    soil_moisture = {fmt_array(soil_moisture, '.6f')},
    soil_temperature = {fmt_array(soil_temp, '.4f')},
    root_fraction = {fmt_array(root_frac, '.4f')},
    deep_soil_temperature = {deep_temp:.4f},
    """)
    return block


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

SPECIES_LABELS = {
    "9801_02": "Beech (Ly2)",
    "9801_04": "Oak (Ly4)",
    "9801_08": "Douglas fir (Ly8)",
    "mean": "Site mean",
}

SPECIES_COLORS = {
    "9801_02": "#2ca02c",
    "9801_04": "#d62728",
    "9801_08": "#1f77b4",
    "mean": "#333333",
}

SPECIES_LINESTYLES = {
    "9801_02": "-",
    "9801_04": "-",
    "9801_08": "-",
    "mean": "--",
}


def plot_initial_profiles(profiles, layers, soil_temps, outdir, formats):
    """Vertical soil moisture + temperature profiles at initialisation time."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7), sharey=True)

    obs_depths = profiles["mean"]["depth_cm"].values
    layer_centers_cm = layers["centers_m"] * 100

    for sid in ["9801_02", "9801_04", "9801_08", "mean"]:
        p = profiles[sid]
        ax1.plot(
            p["wc_m3m3"], -p["depth_cm"],
            marker="o", markersize=4,
            color=SPECIES_COLORS[sid],
            linestyle=SPECIES_LINESTYLES[sid],
            linewidth=1.8 if sid == "mean" else 1.2,
            label=SPECIES_LABELS[sid],
        )

    # Mark PALM layer centers
    ax1.axhline(0, color="brown", linewidth=0.5, linestyle=":")
    for c in layer_centers_cm:
        ax1.axhline(-c, color="gray", linewidth=0.3, alpha=0.5)

    ax1.set_xlabel(r"Soil moisture (m$^3$ m$^{-3}$)", fontsize=11)
    ax1.set_ylabel("Depth (cm)", fontsize=11)
    ax1.set_title("Soil Moisture", fontsize=12)
    ax1.legend(fontsize=9, loc="lower right")
    ax1.set_xlim(left=0)
    ax1.grid(True, alpha=0.3)

    # Soil temperature (interpolated)
    for sid in ["9801_02", "9801_04", "9801_08", "mean"]:
        ax2.plot(
            soil_temps[sid], -layer_centers_cm,
            marker="s", markersize=4,
            color=SPECIES_COLORS[sid],
            linestyle=SPECIES_LINESTYLES[sid],
            linewidth=1.8 if sid == "mean" else 1.2,
            label=SPECIES_LABELS[sid],
        )

    ax2.set_xlabel("Soil temperature (K)", fontsize=11)
    ax2.set_title("Soil Temperature (interpolated)", fontsize=12)
    ax2.legend(fontsize=9, loc="lower left")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "PALM Initial Soil Conditions from Britz Lysimeters\n"
        "2024-09-04 00:00 UTC",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()

    for fmt in formats:
        fig.savefig(os.path.join(outdir, f"initial_soil_profiles.{fmt}"), dpi=200)
    plt.close(fig)
    print(f"  Saved: initial_soil_profiles.{{{','.join(formats)}}}")


def plot_timeseries(timeseries, cfg, outdir, formats):
    """Time series of soil moisture at each depth, one subplot per station."""
    obs_depths = cfg["palm"]["obs_depths_cm"]
    stations = ["9801_02", "9801_04", "9801_08"]

    cmap = plt.cm.viridis_r
    depth_colors = {d: cmap(i / (len(obs_depths) - 1)) for i, d in enumerate(obs_depths)}

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    for ax, sid in zip(axes, stations):
        ts = timeseries[sid]
        for d in obs_depths:
            ax.plot(
                ts.index, ts[d] / 100.0,
                color=depth_colors[d],
                linewidth=1.0,
                label=f"{d} cm",
            )
        ax.set_ylabel(r"$\theta$ (m$^3$ m$^{-3}$)", fontsize=10)
        ax.set_title(SPECIES_LABELS[sid], fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(ts.index[0], ts.index[-1])

    # Single legend at bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=5,
        fontsize=8, bbox_to_anchor=(0.5, -0.02),
    )

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=12))

    fig.suptitle(
        "Britz Soil Moisture Observations\n"
        "2024-09-04 to 2024-09-07 UTC",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.08)

    for fmt in formats:
        fig.savefig(os.path.join(outdir, f"soil_moisture_timeseries.{fmt}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: soil_moisture_timeseries.{{{','.join(formats)}}}")


def plot_comparison(profiles, layers, outdir, formats):
    """Bar chart comparing initial moisture across all stations by depth."""
    obs_depths = profiles["mean"]["depth_cm"].values
    stations = ["9801_02", "9801_04", "9801_08"]
    n_depths = len(obs_depths)
    n_stations = len(stations)
    bar_width = 0.25
    x = np.arange(n_depths)

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, sid in enumerate(stations):
        vals = profiles[sid]["wc_m3m3"].values
        ax.bar(
            x + i * bar_width, vals, bar_width,
            label=SPECIES_LABELS[sid],
            color=SPECIES_COLORS[sid],
            alpha=0.85,
        )

    # Mean line
    ax.plot(
        x + bar_width, profiles["mean"]["wc_m3m3"].values,
        "k--o", markersize=5, linewidth=1.5,
        label=SPECIES_LABELS["mean"],
    )

    ax.set_xticks(x + bar_width)
    ax.set_xticklabels([f"{d} cm" for d in obs_depths])
    ax.set_xlabel("Depth", fontsize=11)
    ax.set_ylabel(r"Soil moisture (m$^3$ m$^{-3}$)", fontsize=11)
    ax.set_title(
        "Initial Soil Moisture Comparison Across Lysimeters\n"
        "2024-09-04 00:00 UTC",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    for fmt in formats:
        fig.savefig(os.path.join(outdir, f"initial_moisture_comparison.{fmt}"), dpi=200)
    plt.close(fig)
    print(f"  Saved: initial_moisture_comparison.{{{','.join(formats)}}}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Britz soil analysis for PALM")
    parser.add_argument("--config", default="config.yml", help="Config file path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    outdir = cfg["output"]["dir"]
    formats = cfg["output"]["formats"]
    os.makedirs(outdir, exist_ok=True)

    print("Loading observations...")
    dfs = load_observations(cfg)

    print("Extracting initial conditions at target time...")
    profiles = extract_initial_conditions(dfs, cfg)

    print("Extracting time series...")
    timeseries = extract_timeseries(dfs, cfg)

    print("Computing PALM soil layers...")
    layers = compute_palm_layers(cfg)

    print(f"\nPALM soil layer configuration ({layers['n_layers']} layers):")
    print(f"  dz_soil (m): {[f'{v:.2f}' for v in layers['dz_soil']]}")
    print(f"  bottoms (m): {[f'{v:.2f}' for v in layers['bottoms_m']]}")
    print(f"  centers (m): {[f'{v:.4f}' for v in layers['centers_m']]}")
    print(f"  total depth: {layers['bottoms_m'][-1]:.2f} m")

    print("\nInterpolating soil temperature to new layers...")
    # Same temperature profile for all variants (no obs temp data per lysimeter)
    soil_temp = compute_soil_temperature(layers, cfg)
    soil_temps = {sid: soil_temp for sid in list(profiles.keys())}

    print("Computing root fraction...")
    root_frac = compute_root_fraction(layers)

    deep_temp = cfg["palm"]["deep_soil_temperature"]

    print("\n" + "=" * 70)
    print("PALM &land_surface_parameters blocks")
    print("=" * 70)

    p3d_blocks = {}
    for sid in ["9801_02", "9801_04", "9801_08", "mean"]:
        label = SPECIES_LABELS[sid]
        sm = profiles[sid]["wc_m3m3"].values
        block = format_p3d_block(label, layers, sm, soil_temp, root_frac, deep_temp)
        p3d_blocks[sid] = block

        print(f"\n--- {label} ---")
        print(block)

    # Save parameter blocks to files
    for sid, block in p3d_blocks.items():
        species_tag = cfg["stations"].get(sid, {}).get("species", sid)
        fname = os.path.join(outdir, f"p3d_soil_params_{species_tag}.txt")
        with open(fname, "w") as f:
            f.write(block)
        print(f"  Saved: {fname}")

    print("\nGenerating plots...")
    plot_initial_profiles(profiles, layers, soil_temps, outdir, formats)
    plot_timeseries(timeseries, cfg, outdir, formats)
    plot_comparison(profiles, layers, outdir, formats)

    # Print summary table
    print("\n" + "=" * 70)
    print("Summary: soil moisture at 2024-09-04 00:00 UTC")
    print("=" * 70)
    header = f"{'Depth':>8s}"
    for sid in ["9801_02", "9801_04", "9801_08", "mean"]:
        header += f"  {SPECIES_LABELS[sid]:>16s}"
    print(header)
    print("-" * len(header))

    for i, d in enumerate(profiles["mean"]["depth_cm"].values):
        row = f"{d:6.0f}cm"
        for sid in ["9801_02", "9801_04", "9801_08", "mean"]:
            val = profiles[sid]["wc_m3m3"].values[i]
            row += f"  {val:16.6f}"
        print(row)

    print(f"\nUnit: m3/m3 (converted from vol-% by dividing by 100)")


if __name__ == "__main__":
    main()
