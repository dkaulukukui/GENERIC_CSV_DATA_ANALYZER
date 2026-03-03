import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import json
import os

def get_sensor_columns(df, sensor_type, sensor_family=None, include_cal=False):
    """
    Get sensor columns from dataframe supporting multiple naming conventions.
    
    Supports:
    - ch_zero_bme_temperature (original format)
    - Sensor0_BME_Temp (new format)
    
    Args:
        df: DataFrame
        sensor_type: 'temperature', 'pressure', or 'humidity'
        sensor_family: 'bme', 'hdc', or None for all
        include_cal: Whether to include calibrated columns
    
    Returns:
        List of matching column names
    """
    sensor_cols = []
    
    # Normalize inputs
    if sensor_family:
        sensor_family = sensor_family.lower()
    
    # Map sensor_type to possible column name variations
    type_patterns = {
        'temperature': ['temp', 'temperature'],
        'pressure': ['pressure', 'press'],
        'humidity': ['humidity', 'humid']
    }
    
    patterns = type_patterns.get(sensor_type.lower(), [sensor_type.lower()])
    
    for col in df.columns:
        col_lower = col.lower()
        
        # Check if calibrated column
        is_cal = 'cal' in col_lower
        if is_cal and not include_cal:
            continue
        if not is_cal and include_cal:
            continue
        
        # Check if matches sensor type
        type_match = any(pattern in col_lower for pattern in patterns)
        if not type_match:
            continue
        
        # Check if matches sensor family (if specified)
        if sensor_family:
            family_match = f'_{sensor_family}_' in col_lower or f'{sensor_family}_' in col_lower
            if not family_match:
                continue
        
        sensor_cols.append(col)
    
    return sensor_cols

def extract_channel_info(col):
    """
    Extract channel number and sensor family from column name.
    
    Supports:
    - ch_zero_bme_temperature -> ('zero', 'bme')
    - Sensor0_BME_Temp -> ('0', 'bme')
    """
    col_lower = col.lower()
    
    # Try original format: ch_zero_bme_temperature
    if col_lower.startswith('ch_'):
        parts = col.split('_')
        if len(parts) >= 3:
            channel = parts[1]  # 'zero', 'one', etc.
            family = parts[2]   # 'bme', 'hdc'
            return channel, family.lower()
    
    # Try new format: Sensor0_BME_Temp or Sensor0_HDC_Temp
    if 'sensor' in col_lower:
        parts = col.split('_')
        if len(parts) >= 2:
            # Extract number from Sensor0, Sensor1, etc.
            sensor_part = parts[0]
            channel = ''.join(filter(str.isdigit, sensor_part))
            
            # Extract family (BME or HDC)
            family = parts[1] if len(parts) > 1 else ''
            return channel, family.lower()
    
    # Fallback - try to extract any info
    parts = col.split('_')
    if len(parts) >= 2:
        return parts[1], parts[2] if len(parts) > 2 else ''
    
    return col, ''

def load_coefficients(coeff_file):
    """Load coefficients from a JSON file"""
    try:
        with open(coeff_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Coefficient file '{coeff_file}' not found")
        return None
    except json.JSONDecodeError:
        print(f"Invalid JSON in coefficient file '{coeff_file}'")
        return None

def save_coefficients(df, sensor_type, output_file, ref_channel='zero'):
    """Calculate and save coefficients to a JSON file"""
    coefficients = {}
    
    ref_col = f'ch_{ref_channel}_{sensor_type}'
    if ref_col not in df.columns:
        print(f"Reference channel '{ref_channel}' for {sensor_type} not found")
        print(f"Available columns: {[col for col in df.columns if sensor_type in col]}")
        return
    
    sensor_cols = [col for col in df.columns if sensor_type in col and col != ref_col]
    ref_data = df[ref_col].dropna()
    
    for col in sensor_cols:
        channel = col.split('_')[1]
        channel_data = df[col].dropna()
        
        common_idx = ref_data.index.intersection(channel_data.index)
        if len(common_idx) < 2:
            continue
        
        x = ref_data.loc[common_idx]
        y = channel_data.loc[common_idx]
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        coefficients[f'ch_{channel}_{sensor_type}'] = {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'reference_channel': ref_channel
        }
    
    # Get absolute path for clarity
    abs_path = os.path.abspath(output_file)
    
    with open(output_file, 'w') as f:
        json.dump(coefficients, f, indent=2)
    
    print(f"\nCoefficients saved to: {abs_path}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Also print the contents so you can see what was saved
    print(f"\nSaved coefficients:")
    print(json.dumps(coefficients, indent=2))

def calculate_first_point_coefficients(df, sensor_type, ref_channel='zero'):
    """Calculate coefficients using first data point as reference"""
    
    # Get all sensor columns for this type
    sensor_cols = [col for col in df.columns if sensor_type in col]
    
    if not sensor_cols:
        print(f"No {sensor_type} columns found")
        return {}
    
    print(f"\n{sensor_type.capitalize()} First Point Calibration:")
    print("=" * 60)
    
    # Get the first valid data point for reference
    first_row_idx = 0
    reference_values = {}
    
    # Find first row where we have valid data for all sensors
    for idx in range(len(df)):
        valid_count = 0
        temp_refs = {}
        for col in sensor_cols:
            if pd.notna(df[col].iloc[idx]):
                temp_refs[col] = df[col].iloc[idx]
                valid_count += 1
        
        if valid_count == len(sensor_cols):  # All sensors have valid data
            reference_values = temp_refs
            first_row_idx = idx
            break
    
    if not reference_values:
        print("No row found with all sensors having valid data")
        return {}
    
    print(f"Using data from row index {first_row_idx} as reference:")
    
    # Use specified reference channel as the target value
    ref_col = f'ch_{ref_channel}_{sensor_type}'
    if ref_col not in reference_values:
        print(f"Reference channel '{ref_channel}' not found in data")
        print(f"Available channels: {list(reference_values.keys())}")
        return {}
    
    target_value = reference_values[ref_col]
    reference_col = ref_col
    
    print(f"Target value (from {reference_col}): {target_value}")
    print()
    
    coefficients = {}
    
    for col in sensor_cols:
        if col == reference_col:
            continue  # Skip the reference sensor
        
        channel = col.split('_')[1]
        measured_value = reference_values[col]
        
        # Calculate offset needed: target = measured + offset
        # So: calibrated = measured + offset
        offset = target_value - measured_value
        
        # Store as slope=1, intercept=offset for consistency with existing code
        # calibrated = (measured - (-offset)) / 1 = measured + offset
        coefficients[col] = {
            'slope': 1.0,
            'intercept': -offset,  # Negative because formula is (measured - intercept) / slope
            'offset': offset,      # Store the actual offset for clarity
            'reference_channel': ref_channel,
            'reference_point': {
                'measured': measured_value,
                'target': target_value,
                'row_index': first_row_idx
            }
        }
        
        print(f"Channel {channel}:")
        print(f"  Measured at reference: {measured_value:.6f}")
        print(f"  Target value: {target_value:.6f}")
        print(f"  Offset needed: {offset:.6f}")
        print(f"  Calibration: calibrated = measured + {offset:.6f}")
        print()
    
    return coefficients

def save_first_point_coefficients(df, sensor_type, output_file, ref_channel='zero'):
    """Calculate and save first point coefficients to JSON file"""
    coefficients = calculate_first_point_coefficients(df, sensor_type, ref_channel)
    
    if not coefficients:
        return
    
    abs_path = os.path.abspath(output_file)
    
    with open(output_file, 'w') as f:
        json.dump(coefficients, f, indent=2)
    
    print(f"First point coefficients saved to: {abs_path}")
    print(f"Current working directory: {os.getcwd()}")
    
    print(f"\nSaved coefficients:")
    print(json.dumps(coefficients, indent=2))

def apply_calibration(df, coefficients, sensor_type):
    """Apply calibration coefficients to create calibrated columns"""
    calibrated_df = df.copy()
    
    for col in df.columns:
        if sensor_type in col and col in coefficients:
            coeff = coefficients[col]
            slope = coeff['slope']
            intercept = coeff['intercept']
            
            # Create calibrated column name
            cal_col = col.replace(sensor_type, f'{sensor_type}_cal')
            
            # Apply inverse calibration: original = (measured - intercept) / slope
            calibrated_df[cal_col] = (df[col] - intercept) / slope
            
            print(f"Applied calibration to {col} -> {cal_col}")
    
    return calibrated_df

def calculate_coefficients(df, sensor_type, ref_channel='zero'):
    ref_col = f'ch_{ref_channel}_{sensor_type}'
    
    if ref_col not in df.columns:
        print(f"Reference channel '{ref_channel}' for {sensor_type} not found")
        print(f"Available columns: {[col for col in df.columns if sensor_type in col]}")
        return
    
    sensor_cols = [col for col in df.columns if sensor_type in col and col != ref_col]
    
    if not sensor_cols:
        print(f"No other {sensor_type} channels found")
        return
    
    print(f"\n{sensor_type.capitalize()} Coefficients (relative to Channel {ref_channel}):")
    print("=" * 60)
    
    ref_data = df[ref_col].dropna()
    
    for col in sensor_cols:
        channel = col.split('_')[1]
        channel_data = df[col].dropna()
        
        common_idx = ref_data.index.intersection(channel_data.index)
        
        if len(common_idx) < 2:
            print(f"Channel {channel}: Insufficient data for correlation")
            continue
        
        x = ref_data.loc[common_idx]
        y = channel_data.loc[common_idx]
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        mean_diff = np.mean(y - x)
        std_diff = np.std(y - x)
        
        print(f"Channel {channel}:")
        print(f"  Linear fit: y = {slope:.6f}x + {intercept:.6f}")
        print(f"  RÂ² = {r_value**2:.6f}")
        print(f"  Mean difference from reference: {mean_diff:.6f}")
        print(f"  Std deviation of difference: {std_diff:.6f}")
        print(f"  Data points: {len(common_idx)}")
        print()

def plot_sensors(df, sensor_type, sensor_family=None, use_calibrated=False, save_path=None):
    """
    Plot sensor data.
    
    Args:
        df: DataFrame with sensor data
        sensor_type: 'temperature', 'pressure', or 'humidity'
        sensor_family: 'bme', 'hdc', or None for all sensors (case insensitive)
        use_calibrated: Whether to plot calibrated data
        save_path: Path to save the plot (None to just display)
    """
    # Get matching columns using helper function
    sensor_cols = get_sensor_columns(df, sensor_type, sensor_family, include_cal=use_calibrated)
    
    title_suffix = ' (Calibrated)' if use_calibrated else ''
    ylabel_suffix = ' (Calibrated)' if use_calibrated else ''
    
    if not sensor_cols:
        cal_text = 'calibrated ' if use_calibrated else ''
        family_text = f'{sensor_family} ' if sensor_family else ''
        print(f"No {cal_text}{family_text}{sensor_type} columns found")
        return
    
    # Convert timestamp to datetime if it exists
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        x = df['timestamp']
        xlabel = 'Time'
    else:
        x = df.index
        xlabel = 'Index'
    
    fig = plt.figure(figsize=(12, 8))
    
    # Calculate statistics if sensor_family is specified
    stats_text = ""
    if sensor_family:
        # Collect data from all matching sensors
        sensor_data = []
        for col in sensor_cols:
            sensor_data.append(df[col])
        
        # Create a DataFrame from sensor data for easier manipulation
        sensors_df = pd.concat(sensor_data, axis=1)
        
        # Calculate mean and std across all sensors at each time point
        mean_values = sensors_df.mean(axis=1)
        std_values = sensors_df.std(axis=1)
        
        # Calculate statistics
        avg_std = std_values.mean()
        min_std = std_values.min()
        max_std = std_values.max()
        
        # Create text for title
        stats_text = f"\nAvg σ={avg_std:.4f}, Min σ={min_std:.4f}, Max σ={max_std:.4f}"
        
        print(f"\n{sensor_family.upper()} {sensor_type.capitalize()} Sensor Variation Statistics:")
        print("=" * 60)
        print(f"Average Standard Deviation: {avg_std:.6f}")
        print(f"Minimum Standard Deviation: {min_std:.6f}")
        print(f"Maximum Standard Deviation: {max_std:.6f}")
        print(f"Number of sensors: {len(sensor_cols)}")
        print("=" * 60)
        print()
    
    for col in sensor_cols:
        # Extract channel and sensor family for legend using helper function
        channel, family = extract_channel_info(col)
        
        plt.plot(x, df[col], label=f'Ch {channel} ({family.upper()})', marker='o', markersize=3, linewidth=1.5)
    
    # Set units based on sensor type
    if sensor_type == 'temperature':
        units = '°C'
    elif sensor_type == 'pressure':
        units = 'hPa'
    elif sensor_type == 'humidity':
        units = '%RH'
    else:
        units = ''
    
    family_text = f' ({sensor_family.upper()})' if sensor_family else ''
    plt.title(f'All {sensor_type.capitalize()} Readings{family_text}{title_suffix}{stats_text}', 
             fontsize=16, fontweight='bold')
    plt.xlabel(xlabel, fontsize=14, fontweight='bold')
    plt.ylabel(f'{sensor_type.capitalize()}{ylabel_suffix} ({units})', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def plot_sensor_bias(df, sensor_type, sensor_family=None, save_path=None):
    """Plot sensor bias analysis - shows which sensors consistently read high or low relative to mean
    
    Creates a box plot showing the distribution of each sensor's deviation from the mean.
    """
    
    # Get matching columns using helper function
    sensor_cols = get_sensor_columns(df, sensor_type, sensor_family, include_cal=False)
    
    if not sensor_cols:
        family_text = f'{sensor_family} ' if sensor_family else ''
        print(f"No {family_text}{sensor_type} channels found")
        return
    
    # Collect data from all sensors
    sensor_data = []
    for col in sensor_cols:
        sensor_data.append(df[col])
    
    sensors_df = pd.concat(sensor_data, axis=1)
    
    # Calculate mean across all sensors at each time point
    mean_values = sensors_df.mean(axis=1)
    
    # Calculate deviation from mean for each sensor
    deviations = {}
    sensor_labels = []
    
    for col in sensor_cols:
        channel, family = extract_channel_info(col)
        
        label = f'Ch {channel}'
        sensor_labels.append(label)
        
        deviation = df[col] - mean_values
        deviations[label] = deviation.dropna()
    
    # Create box plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left plot: Box plot showing distribution of deviations
    positions = range(len(sensor_labels))
    box_data = [deviations[label].values for label in sensor_labels]
    
    bp = ax1.boxplot(box_data, positions=positions, labels=sensor_labels,
                     patch_artist=True, showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='red', markersize=8),
                     medianprops=dict(color='black', linewidth=2))
    
    # Color boxes based on mean deviation
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(sensor_labels)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=2, label='Mean reference', alpha=0.7)
    ax1.set_xlabel('Sensor Channel', fontsize=14, fontweight='bold')
    
    # Set units based on sensor type
    if sensor_type == 'temperature':
        units = '°C'
    elif sensor_type == 'pressure':
        units = 'hPa'
    elif sensor_type == 'humidity':
        units = '%RH'
    else:
        units = ''
    
    ax1.set_ylabel(f'Deviation from Mean ({units})', fontsize=14, fontweight='bold')
    ax1.set_title('Distribution of Sensor Deviations from Mean', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='both', labelsize=12)
    
    # Right plot: Mean bias ranking
    mean_biases = [deviations[label].mean() for label in sensor_labels]
    std_biases = [deviations[label].std() for label in sensor_labels]
    
    # Sort by mean bias
    sorted_indices = np.argsort(mean_biases)
    sorted_labels = [sensor_labels[i] for i in sorted_indices]
    sorted_means = [mean_biases[i] for i in sorted_indices]
    sorted_stds = [std_biases[i] for i in sorted_indices]
    
    colors_sorted = ['red' if m < 0 else 'blue' for m in sorted_means]
    
    bars = ax2.barh(sorted_labels, sorted_means, xerr=sorted_stds, 
                    color=colors_sorted, alpha=0.7, capsize=5)
    
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.7)
    ax2.set_xlabel(f'Mean Deviation from Mean (± std dev) ({units})', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Sensor Channel', fontsize=14, fontweight='bold')
    ax2.set_title('Sensor Bias Ranking\n(Red=Below Mean, Blue=Above Mean)', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.tick_params(axis='both', labelsize=12)
    
    # Add value labels on bars - position them to the right of the plot area
    xlim = ax2.get_xlim()
    x_range = xlim[1] - xlim[0]
    label_x = xlim[1] + x_range * 0.02  # Position labels just outside the right edge
    
    for i, (label, mean, std) in enumerate(zip(sorted_labels, sorted_means, sorted_stds)):
        ax2.text(label_x, i, f'{mean:.4f}±{std:.4f}', 
                va='center', ha='left', fontsize=10, fontweight='bold')
    
    # Adjust the plot limits to make room for labels
    ax2.set_xlim(xlim[0], xlim[1] + x_range * 0.35)
    
    family_text = f' ({sensor_family.upper()})' if sensor_family else ''
    fig.suptitle(f'{sensor_type.capitalize()} Sensor Bias Analysis{family_text}', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Print statistics
    print(f"\n{sensor_type.capitalize()} Sensor Bias Statistics{family_text}:")
    print("=" * 80)
    print(f"{'Sensor':<12} {'Mean Bias':<15} {'Std Dev':<15} {'Min':<12} {'Max':<12}")
    print("-" * 80)
    
    for label in sorted_labels:
        dev = deviations[label]
        print(f"{label:<12} {dev.mean():>14.6f} {dev.std():>14.6f} "
              f"{dev.min():>11.6f} {dev.max():>11.6f}")
    
    print("=" * 80)
    print(f"\nSensors reading HIGH (above mean): {[l for l, m in zip(sorted_labels, sorted_means) if m > 0]}")
    print(f"Sensors reading LOW (below mean): {[l for l, m in zip(sorted_labels, sorted_means) if m < 0]}")
    print()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def plot_relative_to_reference(df, sensor_type, ref_channel='zero', sensor_family=None, save_path=None):
    """Plot all channels relative to reference channel (difference from reference)
    
    When sensor_family is specified and ref_channel is 'zero' (default), uses the mean
    of all sensors in that family as the reference instead.
    """
    
    # Get matching columns using helper function
    sensor_cols = get_sensor_columns(df, sensor_type, sensor_family, include_cal=False)
    use_mean_reference = sensor_family and (ref_channel == 'zero')
    
    if not sensor_cols:
        family_text = f'{sensor_family} ' if sensor_family else ''
        print(f"No {family_text}{sensor_type} channels found")
        return
    
    # Convert timestamp to datetime if it exists
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        x = df['timestamp']
        xlabel = 'Time'
    else:
        x = df.index
        xlabel = 'Index'
    
    fig = plt.figure(figsize=(12, 8))
    
    if use_mean_reference:
        # Use mean of all sensors in the family as reference
        sensor_data = []
        for col in sensor_cols:
            sensor_data.append(df[col])
        
        sensors_df = pd.concat(sensor_data, axis=1)
        ref_data = sensors_df.mean(axis=1)
        
        # Calculate std dev for statistics
        std_values = sensors_df.std(axis=1)
        avg_std = std_values.mean()
        
        reference_label = f'Mean of all {sensor_family.upper()} sensors'
        title_ref = f'Mean ({sensor_family.upper()})'
        
        print(f"\nUsing mean as reference for {sensor_family.upper()} {sensor_type}")
        print(f"Average Standard Deviation from mean: {avg_std:.6f}")
        
    else:
        # Use specified reference channel
        ref_col = f'ch_{ref_channel}_{sensor_type}'
        
        if ref_col not in df.columns:
            print(f"Reference channel '{ref_channel}' for {sensor_type} not found")
            print(f"Available columns: {[col for col in df.columns if sensor_type in col]}")
            return
        
        # Remove reference column from sensor_cols if present
        if ref_col in sensor_cols:
            sensor_cols.remove(ref_col)
        
        if not sensor_cols:
            print(f"No other {sensor_type} channels found")
            return
        
        ref_data = df[ref_col]
        reference_label = f'Channel {ref_channel} (reference)'
        title_ref = f'Channel {ref_channel}'
    
    # Plot reference as baseline at y=0
    plt.axhline(y=0, color='black', linestyle='--', linewidth=2.5, 
                label=reference_label, alpha=0.7)
    
    for col in sensor_cols:
        # Extract channel and family info using helper function
        channel, family = extract_channel_info(col)
        
        # Calculate difference from reference
        difference = df[col] - ref_data
        
        label = f'Ch {channel} ({family.upper()})' if family else f'Ch {channel}'
        plt.plot(x, difference, label=label, marker='o', markersize=3, linewidth=1.5)
    
    family_text = f' ({sensor_family.upper()})' if sensor_family else ''
    
    # Set units based on sensor type
    if sensor_type == 'temperature':
        units = '°C'
    elif sensor_type == 'pressure':
        units = 'hPa'
    elif sensor_type == 'humidity':
        units = '%RH'
    else:
        units = ''
    
    plt.title(f'{sensor_type.capitalize()} Readings Relative to {title_ref}{family_text}', 
             fontsize=16, fontweight='bold')
    plt.xlabel(xlabel, fontsize=14, fontweight='bold')
    plt.ylabel(f'Difference from {title_ref} ({units})', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def plot_correlation(df, sensor_type, ref_channel='zero'):
    ref_col = f'ch_{ref_channel}_{sensor_type}'
    
    if ref_col not in df.columns:
        print(f"Reference channel '{ref_channel}' for {sensor_type} not found")
        print(f"Available columns: {[col for col in df.columns if sensor_type in col]}")
        return
    
    sensor_cols = [col for col in df.columns if sensor_type in col and col != ref_col and 'cal' not in col]
    
    if not sensor_cols:
        print(f"No other {sensor_type} channels found")
        return
    
    n_cols = min(4, len(sensor_cols))
    n_rows = (len(sensor_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    ref_data = df[ref_col].dropna()
    
    for i, col in enumerate(sensor_cols):
        channel = col.split('_')[1]
        channel_data = df[col].dropna()
        
        common_idx = ref_data.index.intersection(channel_data.index)
        
        if len(common_idx) < 2:
            continue
        
        x = ref_data.loc[common_idx]
        y = channel_data.loc[common_idx]
        
        axes[i].scatter(x, y, alpha=0.6, s=10)
        
        slope, intercept, r_value, _, _ = stats.linregress(x, y)
        line_x = np.array([x.min(), x.max()])
        line_y = slope * line_x + intercept
        axes[i].plot(line_x, line_y, 'r-', 
                    label=f'y = {slope:.3f}x + {intercept:.3f}\nRÂ² = {r_value**2:.3f}')
        
        axes[i].set_xlabel(f'Channel {ref_channel} {sensor_type}')
        axes[i].set_ylabel(f'Channel {channel} {sensor_type}')
        axes[i].set_title(f'Channel {channel} vs Channel {ref_channel}')
        axes[i].legend(loc='best')
        axes[i].grid(True, alpha=0.3)
    
    for i in range(len(sensor_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_vs_reference(df, sensor_type, ref_channel='zero'):
    """Plot all channels vs reference channel - creates three plots for better interpretation"""
    ref_col = f'ch_{ref_channel}_{sensor_type}'
    
    if ref_col not in df.columns:
        print(f"Reference channel '{ref_channel}' for {sensor_type} not found")
        print(f"Available columns: {[col for col in df.columns if sensor_type in col]}")
        return
    
    sensor_cols = [col for col in df.columns if sensor_type in col and col != ref_col and 'cal' not in col]
    
    if not sensor_cols:
        print(f"No other {sensor_type} channels found")
        return
    
    # Create figure with three subplots side by side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(35, 7))
    
    ref_data = df[ref_col].dropna()
    
    # Collect stats for summary plot
    channel_stats = []
    
    for col in sensor_cols:
        channel = col.split('_')[1]
        channel_data = df[col].dropna()
        
        common_idx = ref_data.index.intersection(channel_data.index)
        
        if len(common_idx) < 2:
            print(f"Channel {channel}: Insufficient data")
            continue
        
        x = ref_data.loc[common_idx]
        y = channel_data.loc[common_idx]
        
        # Calculate statistics
        slope, intercept, r_value, _, _ = stats.linregress(x, y)
        mean_diff = np.mean(y - x)
        std_diff = np.std(y - x)
        
        channel_stats.append({
            'channel': channel,
            'slope': slope,
            'intercept': intercept,
            'mean_offset': mean_diff,
            'std_offset': std_diff,
            'r_squared': r_value**2
        })
        
        # First plot: Original scatter with regression
        ax1.scatter(x, y, alpha=0.5, s=20, label=f'Ch {channel}')
        line_x = np.array([x.min(), x.max()])
        line_y = slope * line_x + intercept
        ax1.plot(line_x, line_y, '--', alpha=0.7, linewidth=1)
        
        # Second plot: Residuals (difference from reference)
        residuals = y - x
        ax2.scatter(x, residuals, alpha=0.5, s=20, label=f'Ch {channel}')
    
    # Add diagonal reference line to first plot
    all_vals = [df[ref_col].min(), df[ref_col].max()]
    ax1.plot(all_vals, all_vals, 'k-', linewidth=2, label='y=x (perfect)', alpha=0.5)
    
    ax1.set_title(f'All {sensor_type.capitalize()} Channels vs Reference Ch {ref_channel}')
    ax1.set_xlabel(f'Reference Channel {ref_channel} {sensor_type}')
    ax1.set_ylabel(f'Other Channels {sensor_type}')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Add horizontal line at y=0 to residuals plot
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_title(f'Residuals: Deviation from Reference Channel {ref_channel}')
    ax2.set_xlabel(f'Reference Channel {ref_channel} {sensor_type}')
    ax2.set_ylabel(f'Difference from Reference ({sensor_type})')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Third plot: Offset summary (mean difference from reference)
    if channel_stats:
        channels = [s['channel'] for s in channel_stats]
        offsets = [s['mean_offset'] for s in channel_stats]
        std_offsets = [s['std_offset'] for s in channel_stats]
        
        colors = plt.cm.tab10(range(len(channels)))
        bars = ax3.barh(channels, offsets, xerr=std_offsets, 
                        color=colors, alpha=0.7, capsize=5)
        
        ax3.axvline(x=0, color='k', linestyle='--', linewidth=2, label='No offset')
        ax3.set_xlabel(f'Mean Offset from Reference (± std dev)')
        ax3.set_ylabel('Channel')
        ax3.set_title(f'Average Offset from Reference Channel {ref_channel}')
        ax3.grid(True, alpha=0.3, axis='x')
        ax3.legend(loc='best')
        
        # Add text annotations with offset values
        for i, (ch, offset, std) in enumerate(zip(channels, offsets, std_offsets)):
            ax3.text(offset, i, f' {offset:.3f}Â±{std:.3f}', 
                    va='center', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\n{sensor_type.capitalize()} Channel Statistics (relative to ch_{ref_channel}):")
    print("=" * 80)
    print(f"{'Channel':<10} {'Mean Offset':<15} {'Std Dev':<15} {'Slope':<10} {'RÂ²':<10}")
    print("-" * 80)
    for stat in channel_stats:
        print(f"{stat['channel']:<10} {stat['mean_offset']:>14.6f} {stat['std_offset']:>14.6f} "
              f"{stat['slope']:>9.6f} {stat['r_squared']:>9.6f}")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description='Load CSV and analyze sensor data')
    parser.add_argument('-i', '--input', required=True, help='Input CSV file path')
    parser.add_argument('-p', '--plot', choices=['temperature', 'pressure', 'humidity'], 
                       help='Plot all sensors of specified type')
    parser.add_argument('-f', '--family', choices=['bme', 'hdc'],
                       help='Specify sensor family (bme or hdc) - works with --plot')
    parser.add_argument('-c', '--coeff', choices=['temperature', 'pressure', 'humidity'],
                       help='Calculate coefficients relative to reference channel')
    parser.add_argument('-s', '--scatter', choices=['temperature', 'pressure', 'humidity'],
                       help='Plot scatter plots of channels vs reference channel')
    parser.add_argument('-r', '--relative', choices=['temperature', 'pressure', 'humidity'],
                       help='Plot all channels relative to reference channel (difference)')
    parser.add_argument('-v', '--vs-ref', choices=['temperature', 'pressure', 'humidity'],
                       help='Plot all channels vs reference channel on same plot (temp vs temp)')
    parser.add_argument('-b', '--bias', choices=['temperature', 'pressure', 'humidity'],
                       help='Plot sensor bias analysis showing which sensors read consistently high/low')
    parser.add_argument('--ref-channel', default='zero',
                       help='Reference channel to use for calibration (default: zero)')
    parser.add_argument('--cal-file', help='JSON file containing calibration coefficients')
    parser.add_argument('--save-coeff', help='Save calculated coefficients to JSON file')
    parser.add_argument('--plot-cal', action='store_true', 
                       help='Plot calibrated data (requires --cal-file)')
    parser.add_argument('--first-point', choices=['temperature', 'pressure', 'humidity'],
                       help='Calculate coefficients using first data point as reference')
    parser.add_argument('--save', help='Save plot to file (e.g., plot.png, plot.pdf)')

    
    args = parser.parse_args()
    
    try:
        df = pd.read_csv(args.input)
        print(f"Loaded CSV: {args.input}")
        print(f"Shape: {df.shape}")
        print(f"\nColumn names: {list(df.columns)}")
        print("\nFirst 5 rows:")
        print(df.head())
        
        # Load calibration coefficients if provided
        coefficients = None
        if args.cal_file:
            coefficients = load_coefficients(args.cal_file)
            if coefficients:
                print(f"\nLoaded calibration coefficients from {args.cal_file}")
                
                # Apply calibration to all sensor types found in coefficients
                for sensor_type in ['temperature', 'pressure', 'humidity']:
                    type_coeffs = {k: v for k, v in coefficients.items() if sensor_type in k}
                    if type_coeffs:
                        df = apply_calibration(df, type_coeffs, sensor_type)
        
        if args.coeff:
            calculate_coefficients(df, args.coeff, args.ref_channel)
            
            # Save coefficients if requested
            if args.save_coeff:
                save_coefficients(df, args.coeff, args.save_coeff, args.ref_channel)
        
        # Handle first point calibration
        if args.first_point:
            if args.save_coeff:
                save_first_point_coefficients(df, args.first_point, args.save_coeff, args.ref_channel)
            else:
                calculate_first_point_coefficients(df, args.first_point, args.ref_channel)
        
        if args.plot:
            if args.plot_cal and coefficients:
                plot_sensors(df, args.plot, sensor_family=args.family, use_calibrated=True, save_path=args.save)
            else:
                plot_sensors(df, args.plot, sensor_family=args.family, use_calibrated=False, save_path=args.save)
        
        if args.scatter:
            plot_correlation(df, args.scatter, args.ref_channel)
        
        if args.relative:
            plot_relative_to_reference(df, args.relative, args.ref_channel, sensor_family=args.family, save_path=args.save)
        
        if args.bias:
            plot_sensor_bias(df, args.bias, sensor_family=args.family, save_path=args.save)
        
        if args.vs_ref:
            plot_vs_reference(df, args.vs_ref, args.ref_channel)
            
    except FileNotFoundError:
        print(f"Error: File '{args.input}' not found")
    except Exception as e:
        print(f"Error loading CSV: {e}")

if __name__ == "__main__":
    main()
