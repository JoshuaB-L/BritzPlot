"""
Update Britz PALM dynamic and static drivers with observation-based soil conditions.

Modifies driver files in-place:
  - Dynamic driver: replaces 8-layer zsoil with 10-layer grid matching Britz lysimeter
    observation depths, populated with observed soil moisture and modelled temperature.
  - Static drivers (parent + child N02): adds custom soil hydraulic parameters
    (soil_type=0 + soil_pars) from laboratory analysis instead of generic soil type 3.

Usage:
    python update_drivers.py --config config.yml --variant sitemean
    python update_drivers.py --config config.yml --variant oak --dry-run
    python update_drivers.py --config config.yml --variant sitemean --output-dir output/drivers
"""

import argparse
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import netCDF4 as nc
import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_soil_params(csv_path):
    """Load custom soil hydraulic parameters from CSV.

    Returns dict with keys matching PALM soil_pars indices (0-7):
        0: alpha_vg, 1: l_vg, 2: n_vg, 3: gamma_w_sat,
        4: m_sat, 5: m_fc, 6: m_wilt, 7: m_res
    """
    with open(csv_path) as f:
        header = f.readline()
        values_line = f.readline().strip()

    # CSV has: soil_type, alpha_vg, l_vg, n_vg, gamma_w_sat, m_sat, m_fc, m_wilt, m_res
    # Note: header has a tab instead of comma between l_vg and n_vg (quirk of file)
    parts = re.split(r'[,\t]+', values_line)
    vals = [float(v) for v in parts]

    return {
        "soil_type_orig": int(vals[0]),
        "alpha_vg": vals[1],
        "l_vg": vals[2],
        "n_vg": vals[3],
        "gamma_w_sat": vals[4],
        "m_sat": vals[5],
        "m_fc": vals[6],
        "m_wilt": vals[7],
        "m_res": vals[8],
    }


# ---------------------------------------------------------------------------
# P3D file parsing
# ---------------------------------------------------------------------------

def read_p3d_soil_values(p3d_path):
    """Parse soil_moisture, soil_temperature, dz_soil, deep_soil_temperature from a p3d file.

    Returns dict with keys: soil_moisture, soil_temperature, dz_soil, deep_soil_temperature.
    """
    with open(p3d_path) as f:
        text = f.read()

    def extract_array(name):
        pattern = rf'^\s*{name}\s*=\s*(.+?),$'
        match = re.search(pattern, text, re.MULTILINE)
        if not match:
            raise ValueError(f"Could not find '{name}' in {p3d_path}")
        return np.array([float(v.strip()) for v in match.group(1).split(",")])

    def extract_scalar(name):
        pattern = rf'^\s*{name}\s*=\s*([0-9.eE+-]+)\s*,'
        match = re.search(pattern, text, re.MULTILINE)
        if not match:
            raise ValueError(f"Could not find '{name}' in {p3d_path}")
        return float(match.group(1))

    return {
        "dz_soil": extract_array("dz_soil"),
        "soil_moisture": extract_array("soil_moisture"),
        "soil_temperature": extract_array("soil_temperature"),
        "deep_soil_temperature": extract_scalar("deep_soil_temperature"),
    }


def compute_zsoil(dz_soil):
    """Compute layer center depths from layer thicknesses.

    PALM convention: zsoil values are the depth of the center of each soil layer.
    """
    boundaries = np.concatenate([[0.0], np.cumsum(dz_soil)])
    centers = boundaries[:-1] + dz_soil / 2.0
    return centers


# ---------------------------------------------------------------------------
# Dynamic driver update
# ---------------------------------------------------------------------------

def update_dynamic_driver(src_path, dst_path, soil_m, soil_t, zsoil_new, dry_run=False):
    """Create new dynamic driver with updated soil grid and initial conditions.

    Copies all non-soil variables/dimensions from source, replaces zsoil (8->10 layers),
    and writes new init_soil_m and init_soil_t with the observation-based profiles
    broadcast uniformly across (y, x).
    """
    n_new = len(zsoil_new)
    soil_vars = {"init_soil_m", "init_soil_t"}

    if dry_run:
        print(f"  [DRY RUN] Would update: {dst_path}")
        print(f"    zsoil: {len(zsoil_new)} levels at {zsoil_new}")
        print(f"    soil_m: {soil_m}")
        print(f"    soil_t: {soil_t}")
        return

    # Write to temp file first, then move into place
    dst_dir = os.path.dirname(dst_path)
    fd, tmp_path = tempfile.mkstemp(dir=dst_dir, suffix=".nc.tmp")
    os.close(fd)

    try:
        with nc.Dataset(src_path, "r") as src:
            ny = len(src.dimensions["y"])
            nx = len(src.dimensions["x"])

            # Get fill mask from original init_soil_m (where soil doesn't exist)
            orig_soil_m = src["init_soil_m"][0, :, :]
            if hasattr(orig_soil_m, "mask"):
                fill_mask = orig_soil_m.mask
            else:
                fill_mask = np.zeros((ny, nx), dtype=bool)

            with nc.Dataset(tmp_path, "w", format=src.data_model) as dst:
                # --- Copy dimensions, replacing zsoil ---
                for name, dim in src.dimensions.items():
                    if name == "zsoil":
                        dst.createDimension("zsoil", n_new)
                    else:
                        size = len(dim) if not dim.isunlimited() else None
                        dst.createDimension(name, size)

                # --- Copy global attributes ---
                dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})

                # --- Copy variables ---
                for vname, vobj in src.variables.items():
                    if vname in soil_vars:
                        continue  # handle separately

                    if vname == "zsoil":
                        v = dst.createVariable(
                            "zsoil", vobj.datatype, ("zsoil",),
                            fill_value=vobj._FillValue,
                        )
                        for attr in vobj.ncattrs():
                            if attr != "_FillValue":
                                v.setncattr(attr, vobj.getncattr(attr))
                        v[:] = zsoil_new
                        continue

                    # Standard copy
                    fill = vobj._FillValue if hasattr(vobj, "_FillValue") else None
                    kwargs = {"fill_value": fill} if fill is not None else {}
                    v = dst.createVariable(
                        vname, vobj.datatype, vobj.dimensions, **kwargs,
                    )
                    for attr in vobj.ncattrs():
                        if attr != "_FillValue":
                            v.setncattr(attr, vobj.getncattr(attr))
                    v[:] = vobj[:]

                # --- Write new init_soil_m ---
                v = dst.createVariable(
                    "init_soil_m", "f4", ("zsoil", "y", "x"),
                    fill_value=np.float32(-9999.0),
                )
                v.lod = np.int64(2)
                data = np.zeros((n_new, ny, nx), dtype=np.float32)
                for k in range(n_new):
                    layer = np.full((ny, nx), soil_m[k], dtype=np.float32)
                    layer[fill_mask] = -9999.0
                    data[k] = layer
                v[:] = data

                # --- Write new init_soil_t ---
                v = dst.createVariable(
                    "init_soil_t", "f4", ("zsoil", "y", "x"),
                    fill_value=np.float32(-9999.0),
                )
                v.lod = np.int64(2)
                data = np.zeros((n_new, ny, nx), dtype=np.float32)
                for k in range(n_new):
                    layer = np.full((ny, nx), soil_t[k], dtype=np.float32)
                    layer[fill_mask] = -9999.0
                    data[k] = layer
                v[:] = data

        # Atomic replace
        shutil.move(tmp_path, dst_path)

    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    print(f"  Written: {dst_path}")
    print(f"    zsoil: {n_new} levels, range [{zsoil_new[0]:.2f}, {zsoil_new[-1]:.2f}] m")
    print(f"    init_soil_m: shape ({n_new}, {ny}, {nx}), fill_count={fill_mask.sum()}")
    print(f"    init_soil_t: shape ({n_new}, {ny}, {nx}), fill_count={fill_mask.sum()}")


# ---------------------------------------------------------------------------
# Static driver update
# ---------------------------------------------------------------------------

def update_static_driver(src_path, dst_path, soil_params, dry_run=False):
    """Create updated static driver with custom soil_pars overriding the default lookup table.

    Copies all existing variables from source unchanged (including soil_type), and adds
    soil_pars(nsoil_pars, y, x) with custom parameters. PALM's LSM initialization applies
    soil_pars from file as a Level-3 override AFTER the soil_type lookup, so the original
    soil_type (e.g. 3) must remain valid (1-6) to pass the bounds check.

    PALM soil_pars indices (0-7):
        0: alpha_vg [1/m]
        1: l_vg [-]
        2: n_vg [-]
        3: gamma_w_sat [m/s]
        4: m_sat [m3/m3]
        5: m_fc [m3/m3]
        6: m_wilt [m3/m3]
        7: m_res [m3/m3]
    """
    N_SOIL_PARS = 8
    FILL_VALUE = np.float32(-9999.0)

    param_values = np.array([
        soil_params["alpha_vg"],
        soil_params["l_vg"],
        soil_params["n_vg"],
        soil_params["gamma_w_sat"],
        soil_params["m_sat"],
        soil_params["m_fc"],
        soil_params["m_wilt"],
        soil_params["m_res"],
    ], dtype=np.float32)

    if dry_run:
        print(f"  [DRY RUN] Would update: {dst_path}")
        print(f"    soil_pars values: {param_values}")
        print(f"    soil_type: kept unchanged (soil_pars overrides via Level-3)")
        return

    # Write to temp file first, then move into place
    dst_dir = os.path.dirname(dst_path)
    fd, tmp_path = tempfile.mkstemp(dir=dst_dir, suffix=".nc.tmp")
    os.close(fd)

    try:
        with nc.Dataset(src_path, "r") as src:
            ny = len(src.dimensions["y"])
            nx = len(src.dimensions["x"])

            # Read original soil_type to identify soil locations
            orig_soil_type = src["soil_type"][:]
            if hasattr(orig_soil_type, "mask"):
                soil_exists = ~orig_soil_type.mask
            else:
                soil_exists = orig_soil_type != src["soil_type"]._FillValue

            count_soil = int(np.sum(soil_exists))

            # Get res_orig from an existing variable (e.g. zt) for soil_pars metadata
            res_orig = 5.0
            if "zt" in src.variables and hasattr(src["zt"], "res_orig"):
                res_orig = float(src["zt"].res_orig)

            with nc.Dataset(tmp_path, "w", format=src.data_model) as dst:
                # --- Copy all existing dimensions ---
                for name, dim in src.dimensions.items():
                    size = len(dim) if not dim.isunlimited() else None
                    dst.createDimension(name, size)

                # --- Add nsoil_pars dimension ---
                dst.createDimension("nsoil_pars", N_SOIL_PARS)

                # --- Copy global attributes ---
                dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})

                # --- Copy all existing variables unchanged ---
                for vname, vobj in src.variables.items():
                    fill = vobj._FillValue if hasattr(vobj, "_FillValue") else None
                    kwargs = {"fill_value": fill} if fill is not None else {}
                    v = dst.createVariable(
                        vname, vobj.datatype, vobj.dimensions, **kwargs,
                    )
                    for attr in vobj.ncattrs():
                        if attr != "_FillValue":
                            v.setncattr(attr, vobj.getncattr(attr))
                    v[:] = vobj[:]

                # --- Add nsoil_pars coordinate variable ---
                nsp_var = dst.createVariable("nsoil_pars", "i4", ("nsoil_pars",))
                nsp_var[:] = np.arange(N_SOIL_PARS, dtype=np.int32)

                # --- Add soil_pars variable ---
                sp = dst.createVariable(
                    "soil_pars", "f4", ("nsoil_pars", "y", "x"),
                    fill_value=FILL_VALUE,
                )
                sp.long_name = "soil parameters"
                sp.units = ""
                sp.lod = np.int64(1)
                sp.coordinates = "E_UTM N_UTM lon lat"
                sp.grid_mapping = "crs"
                sp.res_orig = res_orig

                data = np.full((N_SOIL_PARS, ny, nx), FILL_VALUE, dtype=np.float32)
                for k in range(N_SOIL_PARS):
                    data[k, soil_exists] = param_values[k]
                sp[:] = data

        # Atomic replace
        shutil.move(tmp_path, dst_path)

    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    print(f"  Written: {dst_path}")
    print(f"    grid: ({ny}, {nx}), res_orig={res_orig}")
    print(f"    soil_type: kept unchanged ({count_soil} soil cells)")
    print(f"    soil_pars: ({N_SOIL_PARS}, {ny}, {nx}), {count_soil} non-fill per layer (overrides lookup)")
    print(f"    Parameters: alpha_vg={param_values[0]:.3f}, n_vg={param_values[2]:.5f}, "
          f"gamma_w_sat={param_values[3]:.3e}, m_sat={param_values[4]:.4f}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_dynamic(path, expected):
    """Validate dynamic driver output."""
    errors = []
    with nc.Dataset(path, "r") as d:
        zsoil = d["zsoil"][:]
        if len(zsoil) != expected["n_layers"]:
            errors.append(f"  zsoil has {len(zsoil)} levels, expected {expected['n_layers']}")
        else:
            print(f"  OK: zsoil has {len(zsoil)} levels")

        if not np.allclose(zsoil, expected["zsoil"], atol=1e-6):
            errors.append("  zsoil values don't match expected")
        else:
            print("  OK: zsoil values match expected layer centers")

        for vname in ("init_soil_m", "init_soil_t"):
            shape = d[vname].shape
            if shape[0] != expected["n_layers"]:
                errors.append(f"  {vname} shape[0]={shape[0]}, expected {expected['n_layers']}")
            else:
                print(f"  OK: {vname} shape = {shape}")

        soil_m = d["init_soil_m"][:]
        valid = soil_m[soil_m > -9000]
        if len(valid) > 0:
            if valid.min() < 0 or valid.max() > 1:
                errors.append(f"  init_soil_m outside [0,1]: [{valid.min():.4f}, {valid.max():.4f}]")
            else:
                print(f"  OK: init_soil_m range [{valid.min():.6f}, {valid.max():.6f}]")

        soil_t = d["init_soil_t"][:]
        valid_t = soil_t[soil_t > -9000]
        if len(valid_t) > 0:
            if valid_t.min() < 250 or valid_t.max() > 320:
                errors.append(f"  init_soil_t outside [250,320]: [{valid_t.min():.2f}, {valid_t.max():.2f}]")
            else:
                print(f"  OK: init_soil_t range [{valid_t.min():.4f}, {valid_t.max():.4f}] K")
    return errors


def validate_static(path, label, expected_soil_count, expected_params):
    """Validate a single static driver output."""
    errors = []
    with nc.Dataset(path, "r") as s:
        st = s["soil_type"][:]
        if hasattr(st, "mask"):
            non_fill = ~st.mask
        else:
            non_fill = st != s["soil_type"]._FillValue

        unique_vals = np.unique(st[non_fill])
        # soil_type must be in valid PALM range 1-6
        invalid = [v for v in unique_vals if v < 1 or v > 6]
        if invalid:
            errors.append(f"  [{label}] soil_type has invalid values: {invalid} (must be 1-6)")
        else:
            print(f"  OK [{label}]: soil_type values {unique_vals} all valid (1-6), {non_fill.sum()} cells")

        soil_count = int(np.sum(non_fill))
        if soil_count != expected_soil_count:
            errors.append(
                f"  [{label}] soil cell count {soil_count} != expected {expected_soil_count}"
            )
        else:
            print(f"  OK [{label}]: soil cell count matches original ({soil_count})")

        if "soil_pars" not in s.variables:
            errors.append(f"  [{label}] soil_pars variable missing")
        else:
            sp = s["soil_pars"]
            if sp.shape[0] != 8:
                errors.append(f"  [{label}] soil_pars has {sp.shape[0]} parameters, expected 8")
            else:
                print(f"  OK [{label}]: soil_pars shape = {sp.shape}")

            sp_data = sp[:]
            param_ok = True
            for k, (pname, pval) in enumerate(expected_params.items()):
                layer = sp_data[k]
                valid_vals = layer[layer > -9000]
                if len(valid_vals) > 0 and not np.allclose(valid_vals, pval, rtol=1e-4):
                    errors.append(
                        f"  [{label}] soil_pars[{k}] ({pname}): "
                        f"got {valid_vals[0]:.6g}, expected {pval:.6g}"
                    )
                    param_ok = False
            if param_ok:
                print(f"  OK [{label}]: soil_pars values match CSV input")
    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def resolve_p3d_path(input_dir, variant):
    """Find the p3d file for the given variant."""
    job_prefix = None
    for f in sorted(os.listdir(input_dir)):
        if f.endswith("_p3d"):
            job_prefix = f[:-4]  # strip _p3d suffix
            break

    if job_prefix is None:
        raise FileNotFoundError(f"No *_p3d file found in {input_dir}")

    if variant == "beech":
        p3d_name = f"{job_prefix}_p3d"
    else:
        p3d_name = f"{job_prefix}_p3d_{variant}"

    p3d_path = os.path.join(input_dir, p3d_name)
    if not os.path.exists(p3d_path):
        # Fall back to base p3d if variant not found
        base_p3d = os.path.join(input_dir, f"{job_prefix}_p3d")
        if os.path.exists(base_p3d):
            print(f"  Note: variant '{variant}' not found, falling back to base p3d")
            return base_p3d
        raise FileNotFoundError(f"P3D file not found: {p3d_path}")
    return p3d_path


def find_driver_files(input_dir):
    """Find dynamic and all static driver files in the INPUT directory.

    Returns dict with keys: dynamic, statics (list of (path, label) tuples).
    """
    dynamic = None
    statics = []

    for f in sorted(os.listdir(input_dir)):
        full = os.path.join(input_dir, f)
        if f.endswith("_dynamic"):
            dynamic = full
        elif "_static" in f and not f.endswith(".tmp"):
            if f.endswith("_static"):
                statics.insert(0, (full, "parent"))  # parent first
            elif "_static_N" in f:
                # Extract child label like "N02"
                label = f.rsplit("_static_", 1)[1]
                statics.append((full, label))

    return {"dynamic": dynamic, "statics": statics}


def main():
    parser = argparse.ArgumentParser(
        description="Update Britz PALM drivers with observation-based soil conditions",
    )
    parser.add_argument("--config", default="config.yml", help="Config YAML file")
    parser.add_argument(
        "--variant", default="sitemean", choices=["beech", "oak", "sitemean"],
        help="P3D variant to read soil values from (default: sitemean)",
    )
    parser.add_argument(
        "--input-dir", default=None,
        help="Input directory containing driver files (default: from config p3d_template)",
    )
    parser.add_argument(
        "--p3d", default=None,
        help="Explicit path to p3d file for soil values (overrides --variant lookup)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: same as input-dir, i.e. in-place update)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would change without writing files",
    )
    args = parser.parse_args()

    # --- Load configuration ---
    cfg = load_config(args.config)

    # Resolve input directory
    if args.input_dir:
        input_dir = args.input_dir
    else:
        p3d_template = cfg["palm"]["p3d_template"]
        input_dir = str(Path(p3d_template).parent)

    input_dir = str(Path(input_dir).resolve())

    # Output defaults to same as input (in-place)
    output_dir = args.output_dir if args.output_dir else input_dir
    output_dir = str(Path(output_dir).resolve())
    in_place = (output_dir == input_dir)

    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}" + (" (in-place)" if in_place else ""))

    # --- Load soil hydraulic parameters ---
    csv_path = str(Path(cfg["data"]["csv_dir"]) / "soil_static_pars.csv")
    soil_params = load_soil_params(csv_path)
    print(f"\nLoaded soil parameters from: {csv_path}")
    print(f"  alpha_vg={soil_params['alpha_vg']}, n_vg={soil_params['n_vg']}, "
          f"gamma_w_sat={soil_params['gamma_w_sat']:.3e}, m_sat={soil_params['m_sat']}")

    # --- Parse p3d for soil values ---
    if args.p3d:
        p3d_path = str(Path(args.p3d).resolve())
    else:
        p3d_path = resolve_p3d_path(input_dir, args.variant)
    p3d_vals = read_p3d_soil_values(p3d_path)
    print(f"\nRead soil values from: {os.path.basename(p3d_path)}")
    print(f"  variant: {args.variant}")
    print(f"  dz_soil: {p3d_vals['dz_soil']}")
    print(f"  soil_moisture: {p3d_vals['soil_moisture']}")
    print(f"  soil_temperature: {p3d_vals['soil_temperature']}")

    zsoil_new = compute_zsoil(p3d_vals["dz_soil"])
    print(f"  zsoil (layer centers): {zsoil_new}")

    # --- Find all driver files ---
    drivers = find_driver_files(input_dir)

    if drivers["dynamic"] is None:
        print("ERROR: No *_dynamic file found in input directory", file=sys.stderr)
        sys.exit(1)
    if not drivers["statics"]:
        print("ERROR: No *_static files found in input directory", file=sys.stderr)
        sys.exit(1)

    print(f"\nDriver files found:")
    print(f"  Dynamic: {os.path.basename(drivers['dynamic'])}")
    for path, label in drivers["statics"]:
        print(f"  Static [{label}]: {os.path.basename(path)}")

    if not in_place:
        os.makedirs(output_dir, exist_ok=True)

    # --- Update dynamic driver ---
    dynamic_src = drivers["dynamic"]
    dynamic_dst = os.path.join(output_dir, os.path.basename(dynamic_src))
    print(f"\n--- Updating dynamic driver ---")
    update_dynamic_driver(
        dynamic_src, dynamic_dst,
        soil_m=p3d_vals["soil_moisture"],
        soil_t=p3d_vals["soil_temperature"],
        zsoil_new=zsoil_new,
        dry_run=args.dry_run,
    )

    # --- Update all static drivers ---
    static_results = []  # (dst_path, label, soil_count)
    for static_src, label in drivers["statics"]:
        static_dst = os.path.join(output_dir, os.path.basename(static_src))
        print(f"\n--- Updating static driver [{label}] ---")

        # Count original soil cells for validation
        with nc.Dataset(static_src, "r") as s:
            st = s["soil_type"][:]
            if hasattr(st, "mask"):
                soil_count = int(np.sum(~st.mask))
            else:
                soil_count = int(np.sum(st != s["soil_type"]._FillValue))

        update_static_driver(
            static_src, static_dst,
            soil_params=soil_params,
            dry_run=args.dry_run,
        )
        static_results.append((static_dst, label, soil_count))

    # --- Validation ---
    if not args.dry_run:
        expected_params = {
            "alpha_vg": soil_params["alpha_vg"],
            "l_vg": soil_params["l_vg"],
            "n_vg": soil_params["n_vg"],
            "gamma_w_sat": soil_params["gamma_w_sat"],
            "m_sat": soil_params["m_sat"],
            "m_fc": soil_params["m_fc"],
            "m_wilt": soil_params["m_wilt"],
            "m_res": soil_params["m_res"],
        }

        print("\nValidation:")
        all_errors = []

        all_errors.extend(validate_dynamic(dynamic_dst, {
            "n_layers": len(zsoil_new),
            "zsoil": zsoil_new,
        }))

        for static_dst, label, soil_count in static_results:
            all_errors.extend(
                validate_static(static_dst, label, soil_count, expected_params)
            )

        if all_errors:
            print("\nVALIDATION ERRORS:")
            for e in all_errors:
                print(e)
        else:
            print("\n  All validation checks passed.")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Variant:         {args.variant}")
    print(f"  Soil layers:     8 -> {len(zsoil_new)}")
    print(f"  zsoil range:     [{zsoil_new[0]:.2f}, {zsoil_new[-1]:.2f}] m")
    print(f"  soil_type:       unchanged (soil_pars overrides via Level-3)")
    print(f"  soil_pars:       8 custom parameters from Britz lab analysis")
    print(f"  Dynamic:         {dynamic_dst}")
    for static_dst, label, soil_count in static_results:
        print(f"  Static [{label:>6s}]: {static_dst}")

    if args.dry_run:
        print("\n  [DRY RUN] No files were written.")


if __name__ == "__main__":
    main()
