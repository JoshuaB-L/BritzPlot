"""
Soil Moisture Content (m_soil) Calculator

This script calculates current volumetric soil water content (m_soil) from known
soil hydraulic parameters using various methods including the van Genuchten equation.

Author: JoshuaB-L
Date: June 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class SoilMoistureCalculator:
    """Class for calculating soil moisture content from hydraulic parameters."""
    
    def __init__(self, soil_params):
        """
        Initialize with soil hydraulic parameters.
        
        Parameters:
        -----------
        soil_params : dict
            Dictionary containing soil hydraulic parameters:
            - alpha_vg: van Genuchten alpha parameter [1/m]
            - l_vg: tortuosity parameter [-]
            - n_vg: van Genuchten n parameter [-]
            - gamma_w_sat: saturated hydraulic conductivity [m/s]
            - m_sat: saturated moisture content [m³/m³]
            - m_fc: field capacity [m³/m³]
            - m_wilt: wilting point [m³/m³]
            - m_res: residual moisture content [m³/m³]
        """
        self.alpha_vg = soil_params['alpha_vg']  # [1/m]
        self.l_vg = soil_params['l_vg']
        self.n_vg = soil_params['n_vg']
        self.gamma_w_sat = soil_params['gamma_w_sat']  # [m/s]
        self.m_sat = soil_params['m_sat']  # [m³/m³]
        self.m_fc = soil_params['m_fc']  # [m³/m³]
        self.m_wilt = soil_params['m_wilt']  # [m³/m³]
        self.m_res = soil_params['m_res']  # [m³/m³]
        
        # Calculate derived parameters
        self.m_vg = 1.0 - 1.0/self.n_vg
        self.alpha_cm = self.alpha_vg / 100.0  # Convert from 1/m to 1/cm
        
        # Validate parameters
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate soil hydraulic parameters."""
        if self.m_res >= self.m_wilt:
            print(f"Warning: m_res ({self.m_res}) >= m_wilt ({self.m_wilt})")
        if self.m_wilt >= self.m_fc:
            print(f"Warning: m_wilt ({self.m_wilt}) >= m_fc ({self.m_fc})")
        if self.m_fc >= self.m_sat:
            print(f"Warning: m_fc ({self.m_fc}) >= m_sat ({self.m_sat})")
        if self.n_vg <= 1.0:
            print(f"Warning: n_vg ({self.n_vg}) <= 1.0")
    
    def van_genuchten_theta(self, h_cm):
        """
        Calculate volumetric water content using van Genuchten equation.
        
        Parameters:
        -----------
        h_cm : float or array
            Pressure head magnitude [cm] (positive values)
            
        Returns:
        --------
        float or array
            Volumetric water content [m³/m³]
        """
        h_cm = np.abs(h_cm)  # Ensure positive pressure head
        
        # van Genuchten equation
        theta = self.m_res + (self.m_sat - self.m_res) / \
                (1.0 + (self.alpha_cm * h_cm)**self.n_vg)**self.m_vg
        
        return theta
    
    def pressure_head_from_theta(self, theta):
        """
        Calculate pressure head from volumetric water content (inverse van Genuchten).
        
        Parameters:
        -----------
        theta : float
            Volumetric water content [m³/m³]
            
        Returns:
        --------
        float
            Pressure head magnitude [cm]
        """
        if theta <= self.m_res:
            return 1e6  # Very high suction for dry conditions
        if theta >= self.m_sat:
            return 0.0  # Saturated conditions
        
        # Inverse van Genuchten equation
        Se = (theta - self.m_res) / (self.m_sat - self.m_res)
        h_cm = (1.0/self.alpha_cm) * (Se**(-1.0/self.m_vg) - 1.0)**(1.0/self.n_vg)
        
        return h_cm
    
    def calculate_m_soil_from_conditions(self, condition='field_capacity'):
        """
        Calculate m_soil based on predefined soil moisture conditions.
        
        Parameters:
        -----------
        condition : str
            Soil moisture condition:
            - 'saturation': at saturation
            - 'field_capacity': at field capacity
            - 'wilting_point': at wilting point
            - 'residual': at residual moisture content
            
        Returns:
        --------
        float
            Soil moisture content [m³/m³]
        """
        conditions = {
            'saturation': self.m_sat,
            'field_capacity': self.m_fc,
            'wilting_point': self.m_wilt,
            'residual': self.m_res
        }
        
        if condition not in conditions:
            raise ValueError(f"Unknown condition: {condition}")
        
        return conditions[condition]
    
    def calculate_m_soil_from_awc_fraction(self, fraction=0.5):
        """
        Calculate m_soil as a fraction of available water capacity.
        
        Parameters:
        -----------
        fraction : float
            Fraction of available water capacity (0.0 to 1.0)
            0.0 = wilting point, 1.0 = field capacity
            
        Returns:
        --------
        float
            Soil moisture content [m³/m³]
        """
        if not (0.0 <= fraction <= 1.0):
            raise ValueError("Fraction must be between 0.0 and 1.0")
        
        awc = self.m_fc - self.m_wilt  # Available water capacity
        m_soil = self.m_wilt + fraction * awc
        
        return m_soil
    
    def calculate_m_soil_from_saturation_fraction(self, fraction=0.5):
        """
        Calculate m_soil as a fraction of saturation.
        
        Parameters:
        -----------
        fraction : float
            Fraction of saturation (0.0 to 1.0)
            
        Returns:
        --------
        float
            Soil moisture content [m³/m³]
        """
        if not (0.0 <= fraction <= 1.0):
            raise ValueError("Fraction must be between 0.0 and 1.0")
        
        m_soil = fraction * self.m_sat
        
        return m_soil
    
    def calculate_m_soil_from_pressure_head(self, h_cm):
        """
        Calculate m_soil from specified pressure head using van Genuchten equation.
        
        Parameters:
        -----------
        h_cm : float
            Pressure head magnitude [cm]
            
        Returns:
        --------
        float
            Soil moisture content [m³/m³]
        """
        return self.van_genuchten_theta(h_cm)
    
    def get_standard_pressure_heads(self):
        """
        Get pressure heads for standard soil moisture conditions.
        
        Returns:
        --------
        dict
            Dictionary with condition names and corresponding pressure heads [cm]
        """
        return {
            'field_capacity': 330.0,      # -1/3 bar
            'wilting_point': 15000.0,     # -15 bar
            'very_wet': 10.0,             # -10 cm
            'moderately_wet': 100.0,      # -100 cm
            'dry': 1000.0,                # -1000 cm
            'very_dry': 10000.0           # -10000 cm
        }
    
    def calculate_all_methods(self):
        """
        Calculate m_soil using all available methods for comparison.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with all calculation methods and results
        """
        results = []
        
        # Method 1: Direct assignment from conditions
        conditions = ['saturation', 'field_capacity', 'wilting_point', 'residual']
        for condition in conditions:
            m_soil = self.calculate_m_soil_from_conditions(condition)
            results.append({
                'Method': f'Direct Assignment ({condition})',
                'm_soil': m_soil,
                'Description': f'Set to {condition} value'
            })
        
        # Method 2: Available water capacity fractions
        awc_fractions = [0.25, 0.5, 0.75, 1.0]
        for frac in awc_fractions:
            m_soil = self.calculate_m_soil_from_awc_fraction(frac)
            results.append({
                'Method': f'AWC Fraction ({frac:.0%})',
                'm_soil': m_soil,
                'Description': f'{frac:.0%} of available water capacity'
            })
        
        # Method 3: Saturation fractions
        sat_fractions = [0.3, 0.5, 0.7, 0.9]
        for frac in sat_fractions:
            m_soil = self.calculate_m_soil_from_saturation_fraction(frac)
            results.append({
                'Method': f'Saturation Fraction ({frac:.0%})',
                'm_soil': m_soil,
                'Description': f'{frac:.0%} of saturation'
            })
        
        # Method 4: Pressure head based
        pressure_heads = self.get_standard_pressure_heads()
        for condition, h_cm in pressure_heads.items():
            m_soil = self.calculate_m_soil_from_pressure_head(h_cm)
            results.append({
                'Method': f'Pressure Head ({condition})',
                'm_soil': m_soil,
                'Description': f'At {h_cm} cm pressure head'
            })
        
        return pd.DataFrame(results)
    
    def plot_water_retention_curve_with_options(self, h_range=(1, 15000)):
        """
        Plot water retention curve with various m_soil options highlighted.
        
        Parameters:
        -----------
        h_range : tuple
            Range of pressure heads to plot [cm]
        """
        # Create pressure head values
        h_values = np.logspace(np.log10(h_range[0]), np.log10(h_range[1]), 1000)
        
        # Calculate corresponding water contents
        theta_values = self.van_genuchten_theta(h_values)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot water retention curve
        ax.semilogx(h_values, theta_values, 'b-', linewidth=2, label='Water Retention Curve')
        
        # Add reference lines for standard conditions
        standard_heads = self.get_standard_pressure_heads()
        colors = ['red', 'orange', 'green', 'purple', 'brown', 'pink']
        
        for i, (condition, h_cm) in enumerate(standard_heads.items()):
            theta = self.van_genuchten_theta(h_cm)
            ax.axvline(x=h_cm, color=colors[i % len(colors)], linestyle='--', alpha=0.7)
            ax.axhline(y=theta, color=colors[i % len(colors)], linestyle='--', alpha=0.7)
            ax.plot(h_cm, theta, 'o', color=colors[i % len(colors)], markersize=8, 
                   label=f'{condition}: θ={theta:.3f}')
        
        # Add soil parameter reference lines
        ax.axhline(y=self.m_sat, color='blue', linestyle='-', alpha=0.5, label=f'Saturation: {self.m_sat:.3f}')
        ax.axhline(y=self.m_fc, color='green', linestyle='-', alpha=0.5, label=f'Field Capacity: {self.m_fc:.3f}')
        ax.axhline(y=self.m_wilt, color='red', linestyle='-', alpha=0.5, label=f'Wilting Point: {self.m_wilt:.3f}')
        ax.axhline(y=self.m_res, color='brown', linestyle='-', alpha=0.5, label=f'Residual: {self.m_res:.3f}')
        
        # Formatting
        ax.set_xlabel('Pressure Head |h| [cm]')
        ax.set_ylabel('Volumetric Water Content θ [m³/m³]')
        ax.set_title('Water Retention Curve with m_soil Options')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_ylim(0, max(self.m_sat * 1.1, 0.5))
        
        plt.tight_layout()
        plt.show()
        
        return fig, ax
    
    def recommend_m_soil(self, use_case='general_modeling'):
        """
        Provide recommended m_soil values for different use cases.
        
        Parameters:
        -----------
        use_case : str
            Use case for the modeling:
            - 'general_modeling': typical conditions
            - 'drought_stress': dry conditions
            - 'wet_conditions': high moisture
            - 'field_capacity': near field capacity
            
        Returns:
        --------
        dict
            Recommended m_soil value and explanation
        """
        recommendations = {
            'general_modeling': {
                'method': 'awc_fraction',
                'value': 0.6,
                'description': '60% of available water capacity - typical for modeling'
            },
            'drought_stress': {
                'method': 'awc_fraction',
                'value': 0.2,
                'description': '20% of available water capacity - stress conditions'
            },
            'wet_conditions': {
                'method': 'saturation_fraction',
                'value': 0.8,
                'description': '80% of saturation - wet soil conditions'
            },
            'field_capacity': {
                'method': 'direct',
                'value': 'field_capacity',
                'description': 'At field capacity - typical after drainage'
            }
        }
        
        if use_case not in recommendations:
            raise ValueError(f"Unknown use case: {use_case}")
        
        rec = recommendations[use_case]
        
        if rec['method'] == 'awc_fraction':
            m_soil = self.calculate_m_soil_from_awc_fraction(rec['value'])
        elif rec['method'] == 'saturation_fraction':
            m_soil = self.calculate_m_soil_from_saturation_fraction(rec['value'])
        elif rec['method'] == 'direct':
            m_soil = self.calculate_m_soil_from_conditions(rec['value'])
        
        return {
            'use_case': use_case,
            'm_soil': m_soil,
            'method': rec['method'],
            'description': rec['description'],
            'fortran_code': f'REAL(wp), PARAMETER :: m_soil = {m_soil:.4f}_wp  ! {rec["description"]}'
        }

def main():
    """Main function to demonstrate soil moisture calculations."""
    
    # Define Britz forest soil parameters
    britz_soil_params = {
        'alpha_vg': 26.437,        # [1/m]
        'l_vg': -0.594,            # [-]
        'n_vg': 1.35154,           # [-]
        'gamma_w_sat': 5.92701e-5, # [m/s]
        'm_sat': 0.3879,           # [m³/m³]
        'm_fc': 0.13,              # [m³/m³]
        'm_wilt': 0.03,            # [m³/m³]
        'm_res': 0.0               # [m³/m³]
    }
    
    # Create calculator instance
    calc = SoilMoistureCalculator(britz_soil_params)
    
    print("=== Britz Forest Soil Moisture Content (m_soil) Calculator ===\n")
    
    # Display soil parameters
    print("Soil Hydraulic Parameters:")
    for param, value in britz_soil_params.items():
        if param == 'gamma_w_sat':
            print(f"  {param}: {value:.3e}")
        else:
            print(f"  {param}: {value:.4f}")
    print()
    
    # Calculate m_soil using all methods
    results_df = calc.calculate_all_methods()
    
    print("=== All Calculation Methods ===")
    print(results_df.to_string(index=False, float_format='%.4f'))
    print()
    
    # Get recommendations for different use cases
    use_cases = ['general_modeling', 'drought_stress', 'wet_conditions', 'field_capacity']
    
    print("=== Recommended m_soil Values ===")
    for use_case in use_cases:
        rec = calc.recommend_m_soil(use_case)
        print(f"\nUse Case: {rec['use_case']}")
        print(f"  m_soil: {rec['m_soil']:.4f} m³/m³")
        print(f"  Description: {rec['description']}")
        print(f"  Fortran Code: {rec['fortran_code']}")
    
    # Calculate specific examples
    print("\n=== Specific Examples ===")
    
    # Example 1: 50% of available water capacity
    m_soil_awc = calc.calculate_m_soil_from_awc_fraction(0.5)
    print(f"50% of Available Water Capacity: {m_soil_awc:.4f} m³/m³")
    
    # Example 2: 70% of saturation
    m_soil_sat = calc.calculate_m_soil_from_saturation_fraction(0.7)
    print(f"70% of Saturation: {m_soil_sat:.4f} m³/m³")
    
    # Example 3: At specific pressure head (100 cm)
    m_soil_h = calc.calculate_m_soil_from_pressure_head(100.0)
    print(f"At 100 cm pressure head: {m_soil_h:.4f} m³/m³")
    
    # Plot water retention curve
    print("\n=== Generating Water Retention Curve ===")
    calc.plot_water_retention_curve_with_options()
    
    print("\nScript completed successfully!")

if __name__ == "__main__":
    main()