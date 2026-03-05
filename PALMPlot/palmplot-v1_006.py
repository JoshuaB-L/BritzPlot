#!/usr/bin/env python3
"""
Enhanced Meteorological Comparison Analysis Tool - Publication Quality Version

This script provides comprehensive comparison analysis of meteorological data from multiple sources:
- Tower observations at different heights for multiple variables
- Dynamic driver data (ICON-D2 and WRF) with robust height mapping and variable derivation
- PALM simulation outputs for corresponding variables

Enhanced Features for Publication:
- Nature Journal quality output formats (PDF, SVG, PNG)
- Professional color schemes with vibrant, publication-ready palettes
- Optimized line weights and styling for academic publications
- Editable text objects in vector formats
- Scientifically robust variable derivation methods

Usage:
    python meteorological_comparison.py --config config.yaml
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List, Union
import warnings
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import xarray as xr
import numpy as np
import yaml
from datetime import datetime, timedelta

# Configure matplotlib for publication quality output
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches

# Configure matplotlib for publication quality
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'serif'],
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'text.usetex': False,  # Set to True if LaTeX is available
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'patch.linewidth': 1.0,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.minor.width': 0.8,
    'ytick.minor.width': 0.8,
    'axes.edgecolor': 'black',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.axisbelow': True
})

# Suppress specific warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class VariableGroup(Enum):
    """Enumeration of variable groups for organized analysis."""
    TEMPERATURE_HUMIDITY = "temperature_humidity"
    WIND_DYNAMICS = "wind_dynamics"
    RADIATION = "radiation"
    PRESSURE = "pressure"


@dataclass
class VariableMetadata:
    """Metadata container for meteorological variables."""
    name: str
    unit: str
    tower_column: str
    palm_variable: str
    group: VariableGroup
    description: str


@dataclass
class HeightMapping:
    """Mapping between tower measurement heights and PALM grid levels."""
    tower_height: float
    palm_z_coordinate: float
    label_suffix: str


@dataclass
class PALMFileMapping:
    """Mapping for PALM variables to their respective files and coordinate systems."""
    filepath: str
    variable_name: str
    coordinate_type: str  # 'zu_3d', 'ku_above_surf', 'profile', etc.
    dimensions: List[str]  # Expected dimensions for the variable


class MeteorologicalComparisonAnalyzer:
    """Enhanced main class for comprehensive meteorological comparison analysis."""
    
    def __init__(self, config_path: str):
        """Initialize the analyzer with configuration."""
        self.config = self._load_config(config_path)
        self.data_dict = {}
        self.variable_metadata = self._initialize_variable_metadata()
        
        # Create output directory first (needed for logging setup)
        self.output_dir = Path(self.config['output']['directory'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging after output directory is created
        self.logger = self._setup_logging()
        
        # Parse overall analysis window (used for plotting axes and framework)
        # Ensure these are timezone-naive for consistent comparisons
        self.overall_start_time = pd.Timestamp(self.config['analysis']['start_time']).tz_localize(None)
        self.overall_end_time = pd.Timestamp(self.config['analysis']['end_time']).tz_localize(None)
        
        # Initialize height mappings
        self.height_mappings = self._initialize_height_mappings()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        log_level = getattr(logging, self.config.get('logging', {}).get('level', 'INFO').upper())
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(self.output_dir / 'analysis.log')
            ]
        )
        
        return logging.getLogger(__name__)
    
    def _initialize_variable_metadata(self) -> Dict[str, VariableMetadata]:
        """Initialize metadata for all supported meteorological variables."""
        variables = {
            'air_temperature': VariableMetadata(
                name='air_temperature',
                unit='°C',
                tower_column='ta',
                palm_variable='theta',  # PALM uses potential temperature
                group=VariableGroup.TEMPERATURE_HUMIDITY,
                description='Air Temperature'
            ),
            'relative_humidity': VariableMetadata(
                name='relative_humidity',
                unit='%',
                tower_column='hur',
                palm_variable='rh',
                group=VariableGroup.TEMPERATURE_HUMIDITY,
                description='Relative Humidity'
            ),
            'wind_u': VariableMetadata(
                name='wind_u',
                unit='m/s',
                tower_column='ua',
                palm_variable='u',
                group=VariableGroup.WIND_DYNAMICS,
                description='Eastward Wind Component'
            ),
            'wind_v': VariableMetadata(
                name='wind_v',
                unit='m/s',
                tower_column='va',
                palm_variable='v',
                group=VariableGroup.WIND_DYNAMICS,
                description='Northward Wind Component'
            ),
            'wind_w': VariableMetadata(
                name='wind_w',
                unit='m/s',
                tower_column='w',
                palm_variable='w',
                group=VariableGroup.WIND_DYNAMICS,
                description='Vertical Wind Component'
            ),
            'wind_speed': VariableMetadata(
                name='wind_speed',
                unit='m/s',
                tower_column='wspeed',
                palm_variable='wspeed',
                group=VariableGroup.WIND_DYNAMICS,
                description='Wind Speed'
            ),
            'wind_direction': VariableMetadata(
                name='wind_direction',
                unit='°',
                tower_column='wdir',
                palm_variable='wdir',
                group=VariableGroup.WIND_DYNAMICS,
                description='Wind Direction'
            ),
            'shortwave_down': VariableMetadata(
                name='shortwave_down',
                unit='W/m²',
                tower_column='rsd',
                palm_variable='rad_sw_in',
                group=VariableGroup.RADIATION,
                description='Downwelling Shortwave Radiation'
            ),
            'shortwave_up': VariableMetadata(
                name='shortwave_up',
                unit='W/m²',
                tower_column='rsu',
                palm_variable='rad_sw_out',
                group=VariableGroup.RADIATION,
                description='Upwelling Shortwave Radiation'
            )
        }
        return variables
    
    def _initialize_height_mappings(self) -> Dict[str, HeightMapping]:
        """Initialize height mappings between tower and PALM coordinates."""
        mappings = {}
        for height_config in self.config.get('height_mappings', []):
            key = height_config['name']
            mappings[key] = HeightMapping(
                tower_height=height_config['tower_height'],
                palm_z_coordinate=height_config['palm_z_coordinate'],
                label_suffix=height_config['label_suffix']
            )
        return mappings
    
    def _get_variable_group_name(self, variable_name: str) -> str:
        """Get the group name for a given variable for averaging configuration."""
        if variable_name not in self.variable_metadata:
            return "default"
        
        var_meta = self.variable_metadata[variable_name]
        group_mapping = {
            VariableGroup.TEMPERATURE_HUMIDITY: "temperature_humidity",
            VariableGroup.WIND_DYNAMICS: "wind_dynamics", 
            VariableGroup.RADIATION: "radiation",
            VariableGroup.PRESSURE: "pressure"
        }
        
        return group_mapping.get(var_meta.group, "default")
    
    def _apply_tower_data_averaging(self, data_series: pd.Series, variable_name: str) -> pd.Series:
        """Apply time averaging to tower data based on configuration settings."""
        try:
            # Check if tower data averaging is enabled
            tower_config = self.config['data_sources'].get('tower_data', {})
            averaging_config = tower_config.get('temporal_averaging', {})
            
            if not averaging_config.get('enabled', False):
                self.logger.debug(f"Tower data averaging disabled for {variable_name}")
                return data_series
            
            # Get variable group and corresponding averaging settings
            group_name = self._get_variable_group_name(variable_name)
            group_configs = averaging_config.get('variable_groups', {})
            
            # Use group-specific config or fall back to default
            if group_name in group_configs:
                avg_settings = group_configs[group_name]
            else:
                avg_settings = averaging_config.get('default', {})
                self.logger.warning(f"No specific averaging config for group '{group_name}', using default settings")
            
            # Extract averaging parameters with robust defaults
            frequency = avg_settings.get('frequency', '30T')
            method = avg_settings.get('method', 'mean').lower()
            min_periods = avg_settings.get('min_periods', 1)
            
            # Validate input data
            if data_series.empty:
                self.logger.warning(f"Empty data series for {variable_name}, skipping averaging")
                return data_series
            
            if len(data_series) < min_periods:
                self.logger.warning(f"Insufficient data points ({len(data_series)}) for {variable_name} averaging (min required: {min_periods})")
                return data_series
            
            # Validate frequency string
            try:
                pd.Timedelta(frequency)
            except (ValueError, TypeError) as e:
                self.logger.error(f"Invalid frequency '{frequency}' for {variable_name}: {e}")
                self.logger.info(f"Using default 30T frequency for {variable_name}")
                frequency = '30T'
            
            # Log averaging operation
            self.logger.info(f"Applying {frequency} {method} averaging to {variable_name} (group: {group_name})")
            original_count = len(data_series)
            
            # Ensure data is sorted by index (time)
            data_series = data_series.sort_index()
            
            # Apply time-based resampling with specified method
            resampler = data_series.resample(frequency, label='left', closed='left')
            
            if method == 'mean':
                averaged_series = resampler.mean()
            elif method == 'median':
                averaged_series = resampler.median()
            elif method == 'max':
                averaged_series = resampler.max()
            elif method == 'min':
                averaged_series = resampler.min()
            elif method == 'std':
                averaged_series = resampler.std()
            elif method == 'sum':
                averaged_series = resampler.sum()
            else:
                self.logger.warning(f"Unknown averaging method '{method}' for {variable_name}, using mean")
                averaged_series = resampler.mean()
            
            # Apply minimum periods requirement
            if min_periods > 1:
                # Count actual data points in each bin
                counts = resampler.count()
                # Set values to NaN where insufficient data points
                averaged_series = averaged_series.where(counts >= min_periods)
            
            # Remove NaN values resulting from insufficient data
            averaged_series = averaged_series.dropna()
            
            final_count = len(averaged_series)
            
            # Log results
            if final_count > 0:
                reduction_ratio = (1 - final_count / original_count) * 100
                self.logger.info(f"Tower data averaging for {variable_name}: {original_count} → {final_count} points ({reduction_ratio:.1f}% reduction)")
                return averaged_series
            else:
                self.logger.warning(f"No valid averaged data points for {variable_name}, returning original data")
                return data_series
                
        except Exception as e:
            self.logger.error(f"Error during tower data averaging for {variable_name}: {e}")
            self.logger.info(f"Returning original unaveraged data for {variable_name}")
            return data_series
    
    @staticmethod
    def kelvin_to_celsius(temp_kelvin: Union[np.ndarray, pd.Series]) -> Union[np.ndarray, pd.Series]:
        """Convert temperature from Kelvin to Celsius."""
        return temp_kelvin - 273.15
    
    @staticmethod
    def calculate_saturation_vapor_pressure(temperature_celsius: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
        """
        Calculate saturation vapor pressure using the Magnus formula.
        
        Args:
            temperature_celsius: Temperature in degrees Celsius
            
        Returns:
            Saturation vapor pressure in hPa
        """
        # Magnus formula coefficients for improved accuracy
        a = 17.27
        b = 237.7
        
        # Calculate saturation vapor pressure (hPa)
        es = 6.112 * np.exp((a * temperature_celsius) / (b + temperature_celsius))
        return es
    
    @staticmethod
    def derive_relative_humidity(qv: Union[np.ndarray, pd.Series], 
                                temperature_celsius: Union[np.ndarray, pd.Series], 
                                pressure_hpa: float = 1013.25) -> Union[np.ndarray, pd.Series]:
        """
        Derive relative humidity from water vapor mixing ratio, temperature, and pressure.
        
        Args:
            qv: Water vapor mixing ratio (kg/kg)
            temperature_celsius: Temperature in degrees Celsius
            pressure_hpa: Atmospheric pressure in hPa (default: standard sea level)
            
        Returns:
            Relative humidity in percent
        """
        try:
            # Convert mixing ratio to vapor pressure (hPa)
            # e = (qv * p) / (0.622 + qv)
            vapor_pressure = (qv * pressure_hpa) / (0.622 + qv)
            
            # Calculate saturation vapor pressure
            saturation_pressure = MeteorologicalComparisonAnalyzer.calculate_saturation_vapor_pressure(temperature_celsius)
            
            # Calculate relative humidity (%)
            relative_humidity = (vapor_pressure / saturation_pressure) * 100.0
            
            # Constrain to physically realistic values
            relative_humidity = np.clip(relative_humidity, 0.0, 100.0)
            
            return relative_humidity
            
        except Exception as e:
            raise ValueError(f"Error calculating relative humidity: {e}")
    
    @staticmethod
    def derive_wind_speed(u: Union[np.ndarray, pd.Series], 
                         v: Union[np.ndarray, pd.Series]) -> Union[np.ndarray, pd.Series]:
        """
        Calculate wind speed from u and v components.
        
        Args:
            u: Eastward wind component (m/s)
            v: Northward wind component (m/s)
            
        Returns:
            Wind speed (m/s)
        """
        return np.sqrt(u**2 + v**2)
    
    @staticmethod
    def derive_wind_direction(u: Union[np.ndarray, pd.Series], 
                             v: Union[np.ndarray, pd.Series]) -> Union[np.ndarray, pd.Series]:
        """
        Calculate wind direction from u and v components using meteorological convention.
        
        Args:
            u: Eastward wind component (m/s)
            v: Northward wind component (m/s)
            
        Returns:
            Wind direction in degrees (0-360°, meteorological convention)
        """
        # Calculate direction using atan2 (mathematical convention)
        direction_rad = np.arctan2(-u, -v)
        
        # Convert to degrees and apply meteorological convention
        direction_deg = np.degrees(direction_rad) + 180.0
        
        # Ensure values are in 0-360° range
        direction_deg = direction_deg % 360.0
        
        return direction_deg
    
    def _get_dynamic_driver_height_index(self, target_height: float) -> int:
        """
        Map target height to dynamic driver grid index with improved precision.
        
        Dynamic driver uses 5m vertical grid resolution starting from surface.
        Grid levels: 0→2.5m, 1→7.5m, 2→12.5m, etc.
        
        Args:
            target_height: Target height in meters
            
        Returns:
            Grid index for dynamic driver (0-based)
        """
        # Improved height mapping based on 5m grid resolution
        if target_height <= 3.0:  # 2m, 3m measurements
            return 0  # First vertical level (~2.5m center)
        elif target_height <= 9.0:  # 8m measurements  
            return 1  # Second vertical level (~7.5m center)
        elif target_height <= 12.0:  # 10m measurements
            return 1  # Second vertical level (~7.5m center, closest to 10m)
        else:
            # For higher levels, calculate based on 5m resolution with center offset
            grid_index = min(int((target_height - 2.5) / 5.0), 79)  # Ensure within z dimension bounds
            return max(0, grid_index)
    
    def _average_boundary_forcing(self, ds: xr.Dataset, variable_base: str, z_index: int, time_index: Optional[int] = None) -> float:
        """
        Average boundary forcing data across all four domain walls with improved robustness.
        
        Args:
            ds: Dynamic driver dataset
            variable_base: Base variable name (e.g., 'pt', 'qv', 'u', 'v', 'w')
            z_index: Vertical grid index
            time_index: Optional time index, if None averages over all time
            
        Returns:
            Averaged value across boundaries
        """
        try:
            boundary_values = []
            
            # Define boundary variable names (exclude top boundary)
            boundaries = ['left', 'right', 'south', 'north']
            
            for boundary in boundaries:
                var_name = f'ls_forcing_{boundary}_{variable_base}'
                
                if var_name in ds.variables:
                    var_data = ds[var_name]
                    
                    # Handle different coordinate systems based on variable
                    try:
                        if time_index is not None:
                            if variable_base == 'w':
                                # W component uses zw coordinate (staggered grid)
                                if z_index < len(ds.zw):  # Ensure index is within bounds
                                    if len(var_data.dims) >= 3 and 'zw' in var_data.dims:
                                        boundary_data = var_data.isel(time=time_index, zw=z_index)
                                    else:
                                        continue
                                else:
                                    continue
                            else:
                                # Standard variables use z coordinate
                                if z_index < len(ds.z):  # Ensure index is within bounds
                                    if len(var_data.dims) >= 3 and 'z' in var_data.dims:
                                        boundary_data = var_data.isel(time=time_index, z=z_index)
                                    else:
                                        continue
                                else:
                                    continue
                        else:
                            # Average over all time
                            if variable_base == 'w':
                                if z_index < len(ds.zw):
                                    if len(var_data.dims) >= 3 and 'zw' in var_data.dims:
                                        boundary_data = var_data.isel(zw=z_index)
                                    else:
                                        continue
                                else:
                                    continue
                            else:
                                if z_index < len(ds.z):
                                    if len(var_data.dims) >= 3 and 'z' in var_data.dims:
                                        boundary_data = var_data.isel(z=z_index)
                                    else:
                                        continue
                                else:
                                    continue
                        
                        # Calculate spatial average for this boundary
                        boundary_mean = float(boundary_data.mean().values)
                        if np.isfinite(boundary_mean):
                            boundary_values.append(boundary_mean)
                        
                    except Exception as e:
                        self.logger.debug(f"Error processing boundary {boundary} for {variable_base}: {e}")
                        continue
            
            if boundary_values:
                result = np.mean(boundary_values)
                self.logger.debug(f"Averaged {variable_base} at z_index {z_index}: {len(boundary_values)} boundaries, value={result:.4f}")
                return result
            else:
                self.logger.warning(f"No valid boundary values for {variable_base} at z_index {z_index}")
                return np.nan
                
        except Exception as e:
            self.logger.error(f"Error averaging boundary forcing for {variable_base}: {e}")
            return np.nan
    
    def _process_dynamic_driver_time_coordinate(self, ds: xr.Dataset, start_time: pd.Timestamp) -> pd.DatetimeIndex:
        """
        Process dynamic driver time coordinate with robust handling of different time formats.
        
        Args:
            ds: Dynamic driver dataset
            start_time: Analysis start time
            
        Returns:
            Processed time coordinate as pandas DatetimeIndex
        """
        try:
            time_coord = ds.time
            time_values = time_coord.values
            
            self.logger.info(f"Dynamic driver time coordinate shape: {len(time_values)} points")
            self.logger.info(f"Time values range: {time_values[0]} to {time_values[-1]}")
            
            # Try multiple time processing strategies
            times = None
            
            # Strategy 1: Check if time has units attribute
            if hasattr(time_coord, 'units'):
                units = getattr(time_coord, 'units', None)
                self.logger.info(f"Time coordinate units: {units}")
                
                try:
                    # Try xarray's time decoding
                    decoded_time = xr.decode_cf(ds).time
                    if decoded_time is not None:
                        times = pd.to_datetime(decoded_time.values)
                        self.logger.info(f"Successfully decoded time using xarray CF conventions")
                except Exception as e:
                    self.logger.warning(f"xarray time decoding failed: {e}")
            
            # Strategy 2: Assume hourly intervals from start time
            if times is None:
                self.logger.info("Using hourly interval assumption from start time")
                times = pd.date_range(
                    start=start_time, 
                    periods=len(time_values), 
                    freq='H'
                )
            
            # Strategy 3: Generate hourly series based on time values as hours
            if times is None:
                try:
                    reference_time = start_time
                    times = reference_time + pd.to_timedelta(time_values, unit='h')
                    self.logger.info(f"Generated hourly time series from time values")
                except Exception as e:
                    self.logger.warning(f"Hourly time generation failed: {e}")
            
            # Strategy 4: Fallback to simple hourly sequence
            if times is None:
                self.logger.warning("Using fallback hourly sequence")
                times = pd.date_range(
                    start=start_time, 
                    periods=len(time_values), 
                    freq='H'
                )
            
            # Ensure timezone-naive
            if hasattr(times, 'tz') and times.tz is not None:
                times = times.tz_localize(None)
            
            self.logger.info(f"Generated time series: {len(times)} points from {times[0]} to {times[-1]}")
            return times
            
        except Exception as e:
            self.logger.error(f"Error processing dynamic driver time coordinate: {e}")
            # Final fallback
            times = pd.date_range(
                start=start_time, 
                periods=len(ds.time), 
                freq='H'
            )
            return times
    
    def load_dynamic_driver_data(self, source_config: Dict[str, Any], variable_name: str) -> Optional[pd.Series]:
        """Load and process ICON-D2 dynamic driver data for a specific variable with enhanced time processing."""
        label = source_config['label']
        filepath = source_config['filepath']
        coordinates = source_config.get('coordinates', {})
        start_time, end_time = self._get_time_window_for_source(source_config, 'dynamic_driver')
        
        # Get variable metadata
        if variable_name not in self.variable_metadata:
            self.logger.error(f"Unknown variable: {variable_name}")
            return None
        
        var_meta = self.variable_metadata[variable_name]
        
        try:
            self.logger.info(f"Loading dynamic driver data: {label} for variable {variable_name}")
            self.logger.info(f"Source time window: {start_time} to {end_time}")
            
            if not os.path.exists(filepath):
                self.logger.warning(f"Dynamic driver file not found: {filepath}")
                return None
            
            # Open dynamic driver dataset
            ds = xr.open_dataset(filepath, decode_times=False)
            
            # Get target height and corresponding grid index
            target_height = coordinates.get('z', 10.0)  # Default to 10m
            z_index = self._get_dynamic_driver_height_index(target_height)
            
            self.logger.info(f"Target height: {target_height:.1f}m, using grid index: {z_index}")
            
            # Process time coordinate with enhanced robustness
            times = self._process_dynamic_driver_time_coordinate(ds, start_time)
            
            data_values = []
            
            # Process each time step
            for t_idx in range(len(times)):
                try:
                    if variable_name == 'air_temperature':
                        # Use potential temperature directly
                        value = self._average_boundary_forcing(ds, 'pt', z_index, t_idx)
                        # Convert from Kelvin to Celsius if necessary
                        if not np.isnan(value) and value > 200:
                            value = self.kelvin_to_celsius(value)
                            
                    elif variable_name == 'relative_humidity':
                        # Derive from water vapor mixing ratio with enhanced method
                        qv_value = self._average_boundary_forcing(ds, 'qv', z_index, t_idx)
                        pt_value = self._average_boundary_forcing(ds, 'pt', z_index, t_idx)
                        
                        if not (np.isnan(qv_value) or np.isnan(pt_value)):
                            # Convert potential temperature to actual temperature if needed
                            temp_celsius = self.kelvin_to_celsius(pt_value) if pt_value > 200 else pt_value
                            # Use more accurate pressure estimate based on height
                            pressure_hpa = 1013.25 * (1 - 0.0065 * target_height / 288.15)**5.255
                            value = self.derive_relative_humidity(qv_value, temp_celsius, pressure_hpa)
                        else:
                            value = np.nan
                            
                    elif variable_name == 'wind_u':
                        value = self._average_boundary_forcing(ds, 'u', z_index, t_idx)
                        
                    elif variable_name == 'wind_v':
                        value = self._average_boundary_forcing(ds, 'v', z_index, t_idx)
                        
                    elif variable_name == 'wind_w':
                        value = self._average_boundary_forcing(ds, 'w', z_index, t_idx)
                        
                    elif variable_name == 'wind_speed':
                        # Derive from u and v components
                        u_value = self._average_boundary_forcing(ds, 'u', z_index, t_idx)
                        v_value = self._average_boundary_forcing(ds, 'v', z_index, t_idx)
                        
                        if not (np.isnan(u_value) or np.isnan(v_value)):
                            value = self.derive_wind_speed(u_value, v_value)
                        else:
                            value = np.nan
                            
                    elif variable_name == 'wind_direction':
                        # Derive from u and v components
                        u_value = self._average_boundary_forcing(ds, 'u', z_index, t_idx)
                        v_value = self._average_boundary_forcing(ds, 'v', z_index, t_idx)
                        
                        if not (np.isnan(u_value) or np.isnan(v_value)):
                            value = self.derive_wind_direction(u_value, v_value)
                        else:
                            value = np.nan
                    else:
                        # Variable not supported in dynamic driver
                        self.logger.warning(f"Variable {variable_name} not supported in dynamic driver")
                        value = np.nan
                    
                    data_values.append(value)
                    
                except Exception as e:
                    self.logger.warning(f"Error processing time step {t_idx} for {variable_name}: {e}")
                    data_values.append(np.nan)
            
            ds.close()
            
            # Create data series
            data_series = pd.Series(data_values, index=times)
            
            # Remove NaN values
            original_length = len(data_series)
            data_series = data_series.dropna()
            valid_length = len(data_series)
            
            self.logger.info(f"Dynamic driver data validation: {original_length} total, {valid_length} valid points")
            
            # Apply more flexible time window filtering
            # Extend the window slightly to account for time coordinate differences
            extended_start = start_time - pd.Timedelta(hours=1)
            extended_end = end_time + pd.Timedelta(hours=1)
            
            mask = (data_series.index >= extended_start) & (data_series.index <= extended_end)
            data_series_filtered = data_series[mask]
            
            # If extended filtering yields no results, try the original window
            if len(data_series_filtered) == 0:
                mask = (data_series.index >= start_time) & (data_series.index <= end_time)
                data_series_filtered = data_series[mask]
            
            # If still no results, use all available data within a reasonable range
            if len(data_series_filtered) == 0:
                self.logger.warning(f"No data found in time window, using all available data for {label}")
                data_series_filtered = data_series
            
            # Log data characteristics
            if len(data_series_filtered) > 0:
                self.logger.info(f"Dynamic driver data for {label}: {len(data_series_filtered)} points")
                self.logger.info(f"Time range: {data_series_filtered.index[0]} to {data_series_filtered.index[-1]}")
                self.logger.info(f"Value range: {data_series_filtered.min():.2f} to {data_series_filtered.max():.2f} {var_meta.unit}")
                return data_series_filtered
            else:
                self.logger.warning(f"No valid dynamic driver data for {label}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error loading dynamic driver data {label} for variable {variable_name}: {str(e)}")
            return None
    
    def _get_time_window_for_source(self, source_config: Dict[str, Any], data_type: str, case_config: Dict[str, Any] = None) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Extract time window for a specific data source, ensuring timezone-naive timestamps."""
        # Check if source has individual time window
        if 'time_window' in source_config:
            start_time = pd.Timestamp(source_config['time_window']['start_time']).tz_localize(None)
            end_time = pd.Timestamp(source_config['time_window']['end_time']).tz_localize(None)
            return start_time, end_time
        
        # For multi-case PALM simulations, check case-specific default time window
        if case_config and 'default_time_window' in case_config:
            start_time = pd.Timestamp(case_config['default_time_window']['start_time']).tz_localize(None)
            end_time = pd.Timestamp(case_config['default_time_window']['end_time']).tz_localize(None)
            return start_time, end_time
        
        # Check if data type has default time window (single-case configuration)
        if data_type in self.config['data_sources']:
            data_source_config = self.config['data_sources'][data_type]
            if 'default_time_window' in data_source_config:
                start_time = pd.Timestamp(data_source_config['default_time_window']['start_time']).tz_localize(None)
                end_time = pd.Timestamp(data_source_config['default_time_window']['end_time']).tz_localize(None)
                return start_time, end_time
        
        # Fall back to overall analysis window
        return self.overall_start_time, self.overall_end_time
    
    def _apply_temporal_averaging(self, data_series: pd.Series, case_config: Dict[str, Any] = None, data_type: str = None) -> pd.Series:
        """Apply temporal averaging to data series based on configuration settings."""
        # Check case-specific configuration first (for multi-case PALM simulations)
        if case_config and 'temporal_averaging' in case_config:
            avg_config = case_config['temporal_averaging']
        # Fall back to data type configuration (single-case configuration)
        elif data_type and data_type in self.config['data_sources']:
            source_config = self.config['data_sources'][data_type]
            if 'temporal_averaging' not in source_config:
                return data_series
            avg_config = source_config['temporal_averaging']
        else:
            return data_series
            
        if not avg_config.get('enabled', False):
            return data_series
            
        target_freq = avg_config.get('target_frequency', '1H')
        method = avg_config.get('method', 'mean')
        
        self.logger.info(f"Applying temporal averaging: {method} resampling to {target_freq}")
        
        try:
            # Resample data according to specified method
            if method == 'mean':
                resampled = data_series.resample(target_freq).mean()
            elif method == 'median':
                resampled = data_series.resample(target_freq).median()
            elif method == 'max':
                resampled = data_series.resample(target_freq).max()
            elif method == 'min':
                resampled = data_series.resample(target_freq).min()
            else:
                self.logger.warning(f"Unknown averaging method '{method}', using mean")
                resampled = data_series.resample(target_freq).mean()
            
            # Remove NaN values that might result from resampling
            resampled = resampled.dropna()
            
            self.logger.info(f"Temporal averaging reduced {len(data_series)} points to {len(resampled)} points")
            return resampled
            
        except Exception as e:
            self.logger.error(f"Error during temporal averaging: {e}")
            return data_series
    
    def load_tower_data(self, source_config: Dict[str, Any], variable_name: str) -> Optional[pd.Series]:
        """Load and process tower observation data for a specific variable."""
        label = source_config['label']
        filepath = source_config['filepath']
        start_time, end_time = self._get_time_window_for_source(source_config, 'tower_data')
        
        # Get variable metadata
        if variable_name not in self.variable_metadata:
            self.logger.error(f"Unknown variable: {variable_name}")
            return None
            
        var_meta = self.variable_metadata[variable_name]
        column_name = var_meta.tower_column
        
        try:
            self.logger.info(f"Loading tower data: {label} for variable {variable_name}")
            self.logger.info(f"Source time window: {start_time} to {end_time}")
            
            if not os.path.exists(filepath):
                self.logger.warning(f"Tower data file not found: {filepath}")
                return None
            
            # Use numpy to read raw data avoiding pandas recursion completely
            try:
                # Read entire file as text first
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                    raw_lines = file.readlines()
                
                # Find header row manually
                header_row_idx = None
                header_line = None
                
                for i, line in enumerate(raw_lines):
                    if line.strip().startswith('DATE'):
                        header_row_idx = i
                        header_line = line.strip()
                        break
                
                if header_row_idx is None:
                    raise ValueError(f"Could not find header row starting with 'DATE' in file: {filepath}")
                
                # Parse header to find column positions
                header_parts = header_line.split(';')
                column_indices = {}
                for idx, col in enumerate(header_parts):
                    column_indices[col.strip()] = idx
                
                # Check if required columns exist
                date_col = 'DATE (DD.MM.YYYY)'
                time_col = 'TIME (UTC) (HH:MM:SS)'
                
                if date_col not in column_indices or time_col not in column_indices:
                    raise ValueError(f"Required datetime columns not found in {filepath}")
                
                if column_name not in column_indices:
                    self.logger.warning(f"Column '{column_name}' not found in file: {filepath}")
                    return None
                
                # Get column positions
                date_idx = column_indices[date_col]
                time_idx = column_indices[time_col]
                data_idx = column_indices[column_name]
                
                # Process data rows manually
                data_rows = []
                for line_idx in range(header_row_idx + 1, len(raw_lines)):
                    line = raw_lines[line_idx].strip()
                    if not line:
                        continue
                        
                    parts = line.split(';')
                    if len(parts) <= max(date_idx, time_idx, data_idx):
                        continue
                    
                    try:
                        date_str = parts[date_idx].strip()
                        time_str = parts[time_idx].strip()
                        data_str = parts[data_idx].strip()
                        
                        # Skip invalid data values
                        if data_str in ['-9999.00', '-999.00', '-99.00', '', 'NaN', 'nan']:
                            continue
                            
                        # Parse datetime
                        datetime_str = f"{date_str} {time_str}"
                        dt = pd.to_datetime(datetime_str, format='%d.%m.%Y %H:%M:%S', errors='coerce')
                        
                        # Parse data value
                        try:
                            data_val = float(data_str)
                            if np.isfinite(data_val):  # Check for inf and nan
                                data_rows.append((dt, data_val))
                        except (ValueError, TypeError):
                            continue
                            
                    except (IndexError, ValueError):
                        continue
                
                if not data_rows:
                    self.logger.warning(f"No valid data found in {filepath}")
                    return None
                
                # Convert to pandas Series
                timestamps, values = zip(*data_rows)
                df = pd.Series(values, index=timestamps)
                
                # Remove any NaT timestamps
                df = df.dropna()
                df = df[df.index.notna()]
                
            except Exception as e:
                self.logger.error(f"Error reading file {filepath}: {e}")
                return None
            
            # Filter by source-specific time window
            mask = (df.index >= start_time) & (df.index <= end_time)
            df = df[mask]
            
            # Apply tower data averaging if configured
            df = self._apply_tower_data_averaging(df, variable_name)
            
            self.logger.info(f"Loaded {len(df)} records for {label}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading tower data {label} for variable {variable_name}: {str(e)}")
            return None
    
    def _get_palm_file_for_variable(self, variable_name: str, source_config: Dict[str, Any]) -> Optional[PALMFileMapping]:
        """Determine which PALM file and coordinate system to use for a given variable."""
        var_meta = self.variable_metadata[variable_name]
        palm_var = var_meta.palm_variable
        
        # Get base filepath from source config
        base_filepath = source_config['filepath']
        base_dir = os.path.dirname(base_filepath)
        
        # Define file mappings for different variable types
        if palm_var in ['wspeed', 'wdir']:
            # Wind speed and direction are in the masked file
            filepath = os.path.join(base_dir, "pcm_study_britz_stretch_snap_10m_2024090400_av_masked_N02_M001_merged.nc")
            return PALMFileMapping(
                filepath=filepath,
                variable_name=palm_var,
                coordinate_type='ku_above_surf',
                dimensions=['time', 'ku_above_surf', 'y', 'x']
            )
        elif palm_var in ['rad_sw_in', 'rad_sw_out']:
            # Radiation variables are in the profile file
            filepath = os.path.join(base_dir, "pcm_study_britz_stretch_snap_10m_2024090400_pr_N02_merged.nc")
            return PALMFileMapping(
                filepath=filepath,
                variable_name=palm_var,
                coordinate_type='profile',
                dimensions=['time', f'z{palm_var}']
            )
        elif palm_var == 'u':
            # U component uses staggered grid
            return PALMFileMapping(
                filepath=base_filepath,
                variable_name=palm_var,
                coordinate_type='zu_3d',
                dimensions=['time', 'zu_3d', 'y', 'xu']
            )
        elif palm_var == 'v':
            # V component uses staggered grid
            return PALMFileMapping(
                filepath=base_filepath,
                variable_name=palm_var,
                coordinate_type='zu_3d',
                dimensions=['time', 'zu_3d', 'yv', 'x']
            )
        else:
            # Standard variables (theta, w, q, etc.)
            return PALMFileMapping(
                filepath=base_filepath,
                variable_name=palm_var,
                coordinate_type='zu_3d' if palm_var != 'w' else 'zw_3d',
                dimensions=['time', 'zu_3d' if palm_var != 'w' else 'zw_3d', 'y', 'x']
            )
    
    def _convert_height_to_grid_point(self, height: float, coordinate_type: str) -> int:
        """Convert height in meters to grid point index for ku_above_surf coordinate system."""
        if coordinate_type == 'ku_above_surf':
            # Refined mapping based on PALM grid structure
            # Grid points represent levels above terrain
            if height <= 2.0 or height <= 18.5:
                return 1  # Near-surface level (approximately 2m)
            elif height <= 8.0 or height <= 27.5:
                return 7  # Mid-level (approximately 8m)  
            elif height <= 10.0 or height <= 29.5:
                return 9  # Upper level (approximately 10m)
            elif height == 20.0:  # Special case for 3m wind measurements
                return 2  # Lower mid-level for 3m measurements
            else:
                # Fallback: convert height directly to grid index
                return max(1, min(19, int(height / 3)))  # Ensure within bounds
        else:
            return height
    
    def load_palm_data(self, source_config: Dict[str, Any], variable_name: str, case_config: Dict[str, Any] = None) -> Optional[pd.Series]:
        """Load and process PALM simulation data for a specific variable."""
        label = source_config['label']
        coordinates = source_config['coordinates']
        
        # Get variable metadata and file mapping
        if variable_name not in self.variable_metadata:
            self.logger.error(f"Unknown variable: {variable_name}")
            return None
            
        var_meta = self.variable_metadata[variable_name]
        file_mapping = self._get_palm_file_for_variable(variable_name, source_config)
        
        if file_mapping is None:
            self.logger.error(f"Could not determine file mapping for variable: {variable_name}")
            return None
        
        filepath = file_mapping.filepath
        palm_var = file_mapping.variable_name
        coord_type = file_mapping.coordinate_type
        
        # Use case configuration if provided, otherwise fall back to legacy single-case configuration
        if case_config:
            start_time, end_time = self._get_time_window_for_source(source_config, 'palm_simulations', case_config)
            ref_date = pd.Timestamp(case_config.get('reference_time', '2024-08-11 00:00:00')).tz_localize(None)
            frequency = case_config.get('frequency', '15min')
        else:
            start_time, end_time = self._get_time_window_for_source(source_config, 'palm')
            ref_date = pd.Timestamp(self.config['data_sources']['palm']['reference_time']).tz_localize(None)
            frequency = self.config['data_sources']['palm'].get('frequency', '15min')
        
        try:
            self.logger.info(f"Loading PALM data: {label} for variable {variable_name}")
            self.logger.info(f"Using file: {filepath}")
            self.logger.info(f"Source time window: {start_time} to {end_time}")
            
            if not os.path.exists(filepath):
                self.logger.warning(f"PALM data file not found: {filepath}")
                return None
            
            # Open dataset with manual time decoding to avoid overflow error
            ds = xr.open_dataset(filepath, decode_times=False)
            
            # Check if the required variable exists
            if palm_var not in ds.variables:
                self.logger.warning(f"Variable '{palm_var}' not found in PALM file: {filepath}")
                ds.close()
                return None
            
            # Handle different coordinate systems
            if coord_type == 'profile':
                # Handle 1D profile data
                z_coord_name = f'z{palm_var}'
                if z_coord_name not in ds.coords:
                    self.logger.error(f"Profile coordinate '{z_coord_name}' not found in {filepath}")
                    ds.close()
                    return None
                
                z_coords = ds[z_coord_name].values
                target_z = coordinates['z']
                z_index = np.abs(z_coords - target_z).argmin()
                
                var_data = ds[palm_var]
                extracted_data = var_data.isel({z_coord_name: z_index}).squeeze()
                
                self.logger.info(f"Selected profile level for {label}: z={z_coords[z_index]:.1f}m (target: {target_z:.1f}m)")
                
            elif coord_type == 'ku_above_surf':
                # Handle masked file with grid points above terrain
                x_coords = ds.x.values
                y_coords = ds.y.values
                k_coords = ds.ku_above_surf.values
                
                # Find closest horizontal indices
                x_index = np.abs(x_coords - coordinates['x']).argmin()
                y_index = np.abs(y_coords - coordinates['y']).argmin()
                
                # Convert height to grid point index
                target_grid_point = self._convert_height_to_grid_point(coordinates['z'], coord_type)
                k_index = min(target_grid_point, len(k_coords) - 1)  # Ensure within bounds
                
                var_data = ds[palm_var]
                extracted_data = var_data.isel(x=x_index, y=y_index, ku_above_surf=k_index).squeeze()
                
                actual_coords = {
                    'x': x_coords[x_index],
                    'y': y_coords[y_index],
                    'k': k_coords[k_index]
                }
                
                self.logger.info(f"Selected coordinates for {label}: "
                               f"x={actual_coords['x']:.1f}m, y={actual_coords['y']:.1f}m, "
                               f"grid_point={actual_coords['k']:.0f} (target: {target_grid_point})")
                
            else:
                # Handle standard 3D files
                x_coord_name = 'xu' if palm_var == 'u' else 'x'
                y_coord_name = 'yv' if palm_var == 'v' else 'y'
                z_coord_name = coord_type
                
                x_coords = ds[x_coord_name].values
                y_coords = ds[y_coord_name].values
                z_coords = ds[z_coord_name].values
                
                # Find closest indices to target coordinates
                x_index = np.abs(x_coords - coordinates['x']).argmin()
                y_index = np.abs(y_coords - coordinates['y']).argmin()
                z_index = np.abs(z_coords - coordinates['z']).argmin()
                
                actual_coords = {
                    'x': x_coords[x_index],
                    'y': y_coords[y_index],
                    'z': z_coords[z_index]
                }
                
                self.logger.info(f"Selected coordinates for {label}: "
                               f"x={actual_coords['x']:.1f}m, y={actual_coords['y']:.1f}m, z={actual_coords['z']:.1f}m")
                
                # Extract variable data based on dimensions
                var_data = ds[palm_var]
                
                if palm_var == 'u':
                    extracted_data = var_data.isel(xu=x_index, y=y_index, zu_3d=z_index).squeeze()
                elif palm_var == 'v':
                    extracted_data = var_data.isel(x=x_index, yv=y_index, zu_3d=z_index).squeeze()
                elif palm_var == 'w':
                    extracted_data = var_data.isel(x=x_index, y=y_index, zw_3d=z_index).squeeze()
                else:
                    extracted_data = var_data.isel(x=x_index, y=y_index, zu_3d=z_index).squeeze()
            
            # Handle time conversion
            time_values = ds.time.values
            time_coord = ds.time
            time_units = getattr(time_coord, 'units', None)
            origin_time_attr = getattr(ds, 'origin_time', None)
            
            self.logger.info(f"Time coordinate attributes for {label}: units='{time_units}', origin_time='{origin_time_attr}'")
            
            times = None
            
            # Strategy 1: Try using origin_time attribute with seconds
            if origin_time_attr and times is None:
                try:
                    ref_time = pd.to_datetime(origin_time_attr).tz_localize(None)
                    times = ref_time + pd.to_timedelta(time_values, unit='s')
                    self.logger.info(f"Successfully converted {label} times using origin_time + seconds")
                except (OverflowError, ValueError) as e:
                    self.logger.warning(f"Origin time conversion failed for {label}: {e}")
            
            # Strategy 2: Try xarray built-in decoding
            if times is None:
                try:
                    ds_temp = xr.open_dataset(filepath, decode_times=True)
                    decoded_times = pd.to_datetime(ds_temp.time.values)
                    if decoded_times.tz is not None:
                        decoded_times = decoded_times.tz_localize(None)
                    times = decoded_times
                    ds_temp.close()
                    self.logger.info(f"Successfully converted {label} times using xarray built-in decoding")
                except Exception as e:
                    self.logger.warning(f"xarray built-in decoding failed for {label}: {e}")
            
            # Strategy 3: Generate time series based on configuration
            if times is None:
                self.logger.warning(f"All automatic time conversion methods failed for {label}, generating time series")
                times = pd.date_range(start=start_time, periods=len(time_values), freq=frequency)
                self.logger.info(f"Generated time series for {label} with {frequency} frequency")
            
            # Final validation
            if times is None:
                self.logger.error(f"Failed to create valid time series for {label}")
                ds.close()
                return None
            
            # Ensure times are timezone-naive
            if hasattr(times, 'tz') and times.tz is not None:
                times = times.tz_localize(None)
            
            # Create data series
            data_series = pd.Series(extracted_data.values, index=times)
            
            # Apply unit conversions if necessary
            if variable_name == 'air_temperature' and data_series.mean() > 200:
                self.logger.info(f"Converting {label} temperature from Kelvin to Celsius")
                data_series = self.kelvin_to_celsius(data_series)
            
            # Apply data validation and filtering for physical realism
            if variable_name in ['shortwave_down', 'shortwave_up']:
                # Filter radiation data for physically realistic values
                # Normal shortwave radiation: 0 to ~1400 W/m²
                original_count = len(data_series)
                data_series = data_series[(data_series >= 0) & (data_series <= 2000)]
                filtered_count = len(data_series)
                
                if filtered_count < original_count:
                    self.logger.warning(f"Filtered {original_count - filtered_count} invalid radiation values from {label}")
                    
                # Replace any remaining extreme values with NaN
                data_series = data_series.replace([np.inf, -np.inf], np.nan)
                data_series = data_series.dropna()
            
            # Log data characteristics before filtering
            self.logger.info(f"PALM data for {label} before filtering: {len(data_series)} points")
            if len(data_series) > 0:
                self.logger.info(f"Value range: {data_series.min():.2f} to {data_series.max():.2f} {var_meta.unit}")
            
            # Filter by source-specific time window
            mask = (data_series.index >= start_time) & (data_series.index <= end_time)
            data_series = data_series[mask]
            
            self.logger.info(f"PALM data for {label} after time filtering: {len(data_series)} points")
            
            if len(data_series) == 0:
                self.logger.warning(f"No PALM data points for {label} fall within the source time window")
                ds.close()
                return None
            
            # Apply temporal averaging if configured
            data_series = self._apply_temporal_averaging(data_series, case_config, 'palm')
            
            ds.close()
            self.logger.info(f"Final dataset for {label}: {len(data_series)} records")
            return data_series
            
        except Exception as e:
            self.logger.error(f"Error loading PALM data {label} for variable {variable_name}: {str(e)}")
            return None
    
    def calculate_statistics(self, ref_data: pd.Series, comp_data: pd.Series) -> Tuple[float, float, int]:
        """Calculate RMSE and bias between two datasets."""
        common_times = ref_data.index.intersection(comp_data.index)
        
        if len(common_times) == 0:
            return np.nan, np.nan, 0
        
        ref_subset = ref_data[common_times]
        comp_subset = comp_data[common_times]
        
        rmse = np.sqrt(np.mean((ref_subset - comp_subset)**2))
        bias = np.mean(comp_subset - ref_subset)
        
        return rmse, bias, len(common_times)
    
    def _save_plot_in_formats(self, fig: plt.Figure, base_filename: str) -> None:
        """
        Save plot in multiple formats based on configuration.
        
        Args:
            fig: Matplotlib figure object
            base_filename: Base filename without extension
        """
        output_config = self.config['output']
        
        # Get DPI and format settings
        dpi = output_config.get('dpi', 300)
        
        # Save PNG if enabled
        if output_config.get('png_enabled', True):
            png_path = self.output_dir / f"{base_filename}.png"
            fig.savefig(png_path, dpi=dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none', format='png')
            self.logger.info(f"PNG plot saved as: {png_path}")
        
        # Save PDF if enabled
        if output_config.get('pdf_enabled', False):
            pdf_path = self.output_dir / f"{base_filename}.pdf"
            fig.savefig(pdf_path, dpi=dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none', format='pdf')
            self.logger.info(f"PDF plot saved as: {pdf_path}")
        
        # Save SVG if enabled
        if output_config.get('svg_enabled', False):
            svg_path = self.output_dir / f"{base_filename}.svg"
            fig.savefig(svg_path, bbox_inches='tight', 
                       facecolor='white', edgecolor='none', format='svg')
            self.logger.info(f"SVG plot saved as: {svg_path}")
    
    def create_variable_group_plot(self, variable_group: VariableGroup) -> None:
        """Create and save comparison plot for a specific variable group with publication quality."""
        group_name = variable_group.value
        self.logger.info(f"Creating publication-quality comparison plot for variable group: {group_name}")
        
        # Filter data for this variable group
        group_variables = {var_name: meta for var_name, meta in self.variable_metadata.items() 
                          if meta.group == variable_group}
        
        if not group_variables:
            self.logger.warning(f"No variables found for group: {group_name}")
            return
        
        # Create publication-quality figure
        n_vars = len(group_variables)
        fig_width = 16  # Wider for Nature journal format
        fig_height = 8 * n_vars
        
        fig, axes = plt.subplots(n_vars, 1, figsize=(fig_width, fig_height), sharex=True)
        
        # Handle single subplot case
        if n_vars == 1:
            axes = [axes]
        
        # Plot styling configuration
        styles = self.config['plotting']['styles']
        
        for idx, (var_name, var_meta) in enumerate(group_variables.items()):
            ax = axes[idx]
            
            # Plot all datasets for this variable
            plotted_lines = []
            plotted_labels = []
            
            for label, data in self.data_dict.items():
                if data is not None and not data.empty and var_name in label:
                    style = styles.get(label, styles['default'])
                    
                    line = ax.plot(data.index, data.values,
                                 marker=style.get('marker', 'o'),
                                 color=style.get('color', 'gray'),
                                 linestyle=style.get('linestyle', '-'),
                                 linewidth=style.get('linewidth', 1.2),
                                 label=label,
                                 markersize=style.get('markersize', 4),
                                 markevery=style.get('markevery', 30),
                                 alpha=style.get('alpha', 1.0),
                                 markeredgewidth=0.5,
                                 markeredgecolor='white')
                    
                    plotted_lines.extend(line)
                    plotted_labels.append(label)
            
            # Configure subplot with publication standards
            ax.set_ylabel(f"{var_meta.description} ({var_meta.unit})", fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, linewidth=0.8)
            ax.set_xlim(self.overall_start_time, self.overall_end_time)
            
            # Enhanced legend positioning and styling
            if plotted_lines:
                legend = ax.legend(plotted_labels, 
                                 bbox_to_anchor=(1.02, 1), 
                                 loc='upper left',
                                 fontsize=11,
                                 frameon=True,
                                 fancybox=True,
                                 shadow=True,
                                 framealpha=0.9,
                                 edgecolor='black',
                                 facecolor='white')
                legend.get_frame().set_linewidth(1.0)
            
            # Set spine properties for publication quality
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)
                spine.set_edgecolor('black')
            
            # Enhanced tick parameters
            ax.tick_params(axis='both', which='major', labelsize=12, width=1.0, length=6)
            ax.tick_params(axis='both', which='minor', width=0.8, length=4)
            
            # Add statistics for this variable
            self._add_variable_statistics_to_plot(ax, var_name)
        
        # Configure overall plot
        axes[-1].set_xlabel('Time (UTC)', fontsize=14, fontweight='bold')
        axes[-1].tick_params(axis='x', rotation=45, labelsize=12)
        
        # Main title with publication formatting
        title = f"{group_name.replace('_', ' ').title()} Comparison"
        fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
        
        # Adjust layout for publication quality
        plt.tight_layout(rect=[0, 0.03, 0.85, 0.97])
        
        # Save in multiple formats
        self._save_plot_in_formats(fig, f"{group_name}_comparison")
        
        # Show plot if configured
        if self.config['plotting'].get('show_plot', False):
            plt.show()
        
        plt.close(fig)
    
    def _add_variable_statistics_to_plot(self, ax, variable_name: str) -> None:
        """Add statistics text box to a subplot for a specific variable with publication formatting."""
        stats_config = self.config['plotting'].get('statistics_box', {})
        
        if not stats_config.get('enabled', True):
            return
            
        stats_text = []
        
        # Find comparison pairs for this variable
        comparison_pairs = self.config['analysis'].get('comparison_pairs', [])
        
        for pair in comparison_pairs:
            ref_label = pair['reference']
            comp_label = pair['comparison']
            
            # Check if both labels contain the variable name
            if (variable_name in ref_label and variable_name in comp_label and
                ref_label in self.data_dict and comp_label in self.data_dict):
                
                ref_data = self.data_dict[ref_label]
                comp_data = self.data_dict[comp_label]
                
                if ref_data is not None and comp_data is not None:
                    rmse, bias, n_points = self.calculate_statistics(ref_data, comp_data)
                    
                    if not np.isnan(rmse):
                        # Shortened labels for cleaner statistics box
                        comp_short = comp_label.replace(f'_{variable_name}', '')
                        ref_short = ref_label.replace(f'_{variable_name}', '')
                        
                        stats_text.append(
                            f'{comp_short} vs {ref_short}:\n'
                            f'RMSE: {rmse:.2f}, Bias: {bias:.2f}, N: {n_points}\n'
                        )
        
        # Add statistics box to plot with publication formatting
        if stats_text:
            position = stats_config.get('position', 'top_right')
            font_size = stats_config.get('font_size', 8)
            bg_alpha = stats_config.get('background_alpha', 0.95)
            
            # Position mapping for subplots
            position_coords = {
                'top_left': (0.02, 0.98, 'top', 'left'),
                'top_right': (0.98, 0.98, 'top', 'right'),
                'bottom_left': (0.02, 0.02, 'bottom', 'left'),
                'bottom_right': (0.98, 0.02, 'bottom', 'right')
            }
            
            if position in position_coords:
                x, y, valign, halign = position_coords[position]
                
                text_box = ax.text(x, y, '\n'.join(stats_text),
                                 transform=ax.transAxes,
                                 verticalalignment=valign,
                                 horizontalalignment=halign,
                                 bbox=dict(facecolor='white', 
                                         alpha=bg_alpha, 
                                         edgecolor='black', 
                                         linewidth=1.0,
                                         pad=2.0,
                                         boxstyle='round,pad=0.3'),
                                 fontsize=font_size,
                                 fontfamily='monospace')
    
    def save_comprehensive_statistics_report(self) -> None:
        """Save detailed statistics report to CSV for all variables."""
        self.logger.info("Generating comprehensive statistics report")
        
        results = []
        comparison_pairs = self.config['analysis'].get('comparison_pairs', [])
        
        for pair in comparison_pairs:
            ref_label = pair['reference']
            comp_label = pair['comparison']
            
            if ref_label in self.data_dict and comp_label in self.data_dict:
                ref_data = self.data_dict[ref_label]
                comp_data = self.data_dict[comp_label]
                
                if ref_data is not None and comp_data is not None:
                    rmse, bias, n_points = self.calculate_statistics(ref_data, comp_data)
                    
                    # Extract variable name from labels
                    variable_name = "unknown"
                    for var_name in self.variable_metadata.keys():
                        if var_name in ref_label:
                            variable_name = var_name
                            break
                    
                    results.append({
                        'Variable': variable_name,
                        'Reference': ref_label,
                        'Comparison': comp_label,
                        'RMSE': rmse,
                        'Bias': bias,
                        'N_points': n_points,
                        'Reference_mean': ref_data.mean(),
                        'Comparison_mean': comp_data.mean(),
                        'Reference_std': ref_data.std(),
                        'Comparison_std': comp_data.std(),
                        'Unit': self.variable_metadata.get(variable_name, VariableMetadata('', '', '', '', VariableGroup.TEMPERATURE_HUMIDITY, '')).unit
                    })
        
        if results:
            stats_df = pd.DataFrame(results)
            stats_path = self.output_dir / 'comprehensive_statistics.csv'
            stats_df.to_csv(stats_path, index=False)
            self.logger.info(f"Comprehensive statistics report saved as: {stats_path}")
    
    def run_analysis(self) -> None:
        """Run the complete meteorological comparison analysis."""
        self.logger.info("Starting comprehensive meteorological comparison analysis")
        
        try:
            # Load tower data for all configured variables
            for variable_name in self.variable_metadata.keys():
                if variable_name in self.config.get('variables_to_analyze', []):
                    tower_sources = self.config['data_sources']['tower_data']['sources']
                    
                    for source in tower_sources:
                        if variable_name in source.get('variables', []):
                            label = f"{source['label']}_{variable_name}"
                            self.data_dict[label] = self.load_tower_data(source, variable_name)
            
            # Load dynamic driver data if configured
            if 'dynamic_driver' in self.config['data_sources']:
                self.logger.info("Processing ICON-D2 dynamic driver data")
                driver_sources = self.config['data_sources']['dynamic_driver']['sources']
                
                for source in driver_sources:
                    for variable_name in source.get('variables', []):
                        if variable_name in self.config.get('variables_to_analyze', []):
                            label = f"{source['label']}_{variable_name}"
                            self.data_dict[label] = self.load_dynamic_driver_data(source, variable_name)
            
            # Load PALM data for all configured variables and cases
            if 'palm_simulations' in self.config['data_sources']:
                for case in self.config['data_sources']['palm_simulations']:
                    case_name = case.get('name', 'unknown')
                    self.logger.info(f"Processing PALM simulation case: {case_name}")
                    
                    for source in case['sources']:
                        for variable_name in source.get('variables', []):
                            if variable_name in self.config.get('variables_to_analyze', []):
                                label = f"{source['label']}_{variable_name}"
                                self.data_dict[label] = self.load_palm_data(source, variable_name, case)
            
            # Create plots for each variable group
            for variable_group in VariableGroup:
                self.create_variable_group_plot(variable_group)
            
            # Save comprehensive statistics report
            self.save_comprehensive_statistics_report()
            
            self.logger.info("Comprehensive meteorological analysis completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during analysis: {str(e)}")
            raise


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(
        description='Enhanced Meteorological Comparison Analysis Tool - Publication Quality',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to the YAML configuration file'
    )
    
    parser.add_argument(
        '--version', '-v',
        action='version',
        version='Enhanced Meteorological Comparison Tool v4.0 (Publication Quality)'
    )
    
    args = parser.parse_args()
    
    try:
        analyzer = MeteorologicalComparisonAnalyzer(args.config)
        analyzer.run_analysis()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()