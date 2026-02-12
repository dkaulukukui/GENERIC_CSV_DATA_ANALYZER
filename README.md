# GENERIC CSV DATA ANALYZER

A comprehensive PyQt5-based GUI application for processing, analyzing, and visualizing time-series data from CSV files. Specifically designed for meteorological data post-processing, this tool provides robust features for data quality control, calibration, statistical analysis, and advanced visualization.

![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Key Features in Detail](#key-features-in-detail)
- [Workflow](#workflow)
- [Data Processing Options](#data-processing-options)
- [Calibration Methods](#calibration-methods)
- [Export Options](#export-options)
- [Contributing](#contributing)
- [License](#license)

## Features

### 🔹 Data Import & Management
- **Multiple CSV File Import**: Load and combine multiple CSV files seamlessly
- **Automatic DateTime Detection**: Intelligent detection of datetime columns with multiple format support
- **Custom DateTime Formats**: Support for ISO8601, custom formats, and auto-detection
- **Data Preview**: View raw data in tabular format with sorting capabilities
- **Column Management**: Rename, remove, and reorder columns with intuitive dialogs

### 🔹 Data Processing
- **Time-Series Resampling**: Resample data to various intervals (hourly, daily, weekly, monthly, custom)
- **Aggregation Methods**: Multiple aggregation options (mean, median, sum, min, max, std)
- **Quality Control**: 
  - Outlier detection and removal using standard deviation thresholds
  - Missing value interpolation (linear, time-based, nearest, polynomial)
- **Timeframe Filtering**: Filter data to specific datetime ranges
- **Duplicate Handling**: Automatic handling of duplicate timestamps

### 🔹 Sensor Calibration
- **Multiple Calibration Methods**:
  - Linear calibration (slope and intercept)
  - Temperature correction (for temperature-sensitive sensors)
  - Humidity correction (relative humidity adjustments)
  - Pressure correction (barometric pressure adjustments)
  - Polynomial calibration (2nd and 3rd order)
  - Logarithmic and exponential transformations
  - Custom formula support
- **Calibration Coefficient Calculator**: Calculate calibration coefficients from reference data
- **Apply Calibrations**: Apply calculated or manual calibrations to sensor data
- **Calibration Export/Import**: Save and load calibration configurations

### 🔹 Visualization
- **Multiple Plot Types**:
  - Single axis plots
  - Dual axis plots (for comparing different scale data)
  - Scatter plots
  - Box plots
  - Histogram distributions
  - Correlation heatmaps
- **Plot Customization**:
  - Data normalization
  - Grid and legend controls
  - Rolling average visualization
  - Interactive matplotlib toolbar
  - Plot export to image files

### 🔹 Statistical Analysis
- **Descriptive Statistics**: Mean, median, std, min, max, quartiles
- **Data Distribution**: Histogram analysis and distribution plots
- **Correlation Analysis**: Correlation matrix with heatmap visualization
- **Missing Data Report**: Comprehensive analysis of missing values
- **Time-Series Statistics**: Period-based analysis and trends

### 🔹 Export Capabilities
- **CSV Export**: Export processed data to CSV format
- **Excel Export**: Export with formatted headers and data
- **Plot Export**: Save visualizations as PNG images
- **Calibration Export**: Save calibration configurations as JSON

## Prerequisites

- Python 3.7 or higher
- Operating System: Windows, macOS, or Linux

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/dkaulukukui/GENERIC_CSV_DATA_ANALYZER.git
cd GENERIC_CSV_DATA_ANALYZER
```

### 2. Install Required Dependencies
```bash
pip install PyQt5 pandas numpy matplotlib scipy openpyxl
```

Or install from a requirements file (if available):
```bash
pip install -r requirements.txt
```

### Dependencies List:
- **PyQt5** (>=5.15.0): GUI framework
- **pandas** (>=1.3.0): Data manipulation and analysis
- **numpy** (>=1.21.0): Numerical computing
- **matplotlib** (>=3.4.0): Data visualization
- **scipy** (>=1.7.0): Scientific computing (for interpolation)
- **openpyxl** (>=3.0.0): Excel file handling

## Usage

### Running the Application

```bash
python GENERIC_CSV_DATA_ANALYZER.py
```

Or make the script executable (Linux/macOS):
```bash
chmod +x GENERIC_CSV_DATA_ANALYZER.py
./GENERIC_CSV_DATA_ANALYZER.py
```

### Quick Start Guide

1. **Import Data**:
   - Navigate to the "File Import" tab
   - Click "Add CSV Files" to select one or more CSV files
   - Specify the datetime column name (or leave empty for auto-detection)
   - Choose datetime format or use auto-detection

2. **Configure Processing**:
   - Go to the "Processing Options" tab
   - Select resampling period (if desired)
   - Choose aggregation method
   - Enable quality control options (outlier removal, interpolation)
   - Set timeframe filters (optional)
   - Click "Process Data"

3. **View Data**:
   - Navigate to "Data View" tab to see the processed data
   - Use column management tools to rename, remove, or reorder columns
   - Export data to CSV or Excel format

4. **Visualize**:
   - Go to "Visualization" tab
   - Select columns to plot
   - Choose plot type (single, dual-axis, scatter, box, histogram, correlation)
   - Customize plot settings
   - Click "Generate Plot"
   - Save plots as PNG images

5. **Apply Calibration** (if needed):
   - Navigate to "Calibration" tab
   - Select source column and calibration method
   - Enter calibration parameters or calculate from reference data
   - Apply calibration to create corrected columns

6. **View Statistics**:
   - Go to "Statistics" tab
   - Review descriptive statistics
   - Analyze data distributions and correlations
   - Check missing data report

## Key Features in Detail

### Datetime Parsing
The application uses intelligent datetime parsing with multiple fallback strategies:
- Automatic format detection using pandas
- Support for ISO8601 standard formats
- Custom format strings (e.g., "%Y-%m-%d %H:%M:%S")
- Timezone-aware datetime handling (converts to UTC)
- Robust error handling for partially valid data

### Data Resampling
Supported resampling periods:
- 1Min, 5Min, 10Min, 15Min, 30Min (minute intervals)
- 1H, 2H, 3H, 6H, 12H (hour intervals)
- 1D, 7D (day intervals)
- 1W (weekly)
- 1MS (monthly)
- Custom periods using pandas offset aliases

### Quality Control
**Outlier Removal**:
- Statistical outlier detection using z-scores
- Configurable standard deviation threshold (default: 3σ)
- Applied independently to each numeric column
- Outliers replaced with NaN for optional interpolation

**Missing Value Interpolation**:
- Linear: Straight-line interpolation between points
- Time: Considers actual time intervals between measurements
- Nearest: Forward/backward fill using nearest values
- Polynomial: Higher-order polynomial fitting

### Column Management
**Rename Columns**: Easily rename columns with validation to prevent duplicates

**Remove Columns**: Select and remove multiple columns at once

**Reorder Columns**: Drag-and-drop interface to reorder columns (datetime column always first)

## Calibration Methods

### Linear Calibration
```
calibrated_value = slope * raw_value + intercept
```
Most common method for sensor calibration.

### Temperature Correction
```
corrected_temp = raw_temp + offset + (coefficient * (raw_temp - reference_temp))
```
Useful for temperature sensor drift correction.

### Humidity Correction
```
corrected_RH = raw_RH * (1 + coefficient) + offset
```
Adjusts relative humidity readings.

### Pressure Correction
```
corrected_pressure = raw_pressure + offset + (elevation_correction / 100)
```
Barometric pressure adjustment with elevation correction.

### Polynomial Calibration
```
2nd Order: y = a*x² + b*x + c
3rd Order: y = a*x³ + b*x² + c*x + d
```
For non-linear sensor responses.

### Custom Formula
Define your own calibration formula using Python syntax with 'x' as the input variable.

## Data Processing Options

### Aggregation Methods
- **Mean**: Average of values in each period
- **Median**: Middle value (robust to outliers)
- **Sum**: Total of values (useful for precipitation, counts)
- **Min/Max**: Minimum or maximum value in period
- **Std**: Standard deviation (measure of variability)

### Timeframe Filtering
- Select start and end datetime using calendar widgets
- Filter data before or after processing
- Useful for focusing analysis on specific events or periods

## Export Options

### CSV Export
- Exports processed data with datetime index
- Includes all numeric columns and metadata
- Compatible with most data analysis tools

### Excel Export
- Formatted Excel workbook (.xlsx)
- Headers and data types preserved
- Easy to share and review

### Plot Export
- High-resolution PNG images
- Customizable size and DPI
- Suitable for reports and presentations

### Calibration Export
- JSON format for calibration coefficients
- Import calibrations for future sessions
- Share calibration settings across teams

## Workflow

Typical workflow for data processing:

```
1. Import CSV Files → 2. Configure Processing → 3. Process Data →
4. View/Validate Data → 5. Apply Calibrations (optional) →
6. Visualize Results → 7. Statistical Analysis → 8. Export Results
```

## Troubleshooting

**Issue**: Datetime column not detected
- **Solution**: Manually specify the datetime column name in the "Column Settings"

**Issue**: All values become NaN after processing
- **Solution**: Check datetime format settings. Try "Auto-detect" or specify correct format

**Issue**: Empty plot or no data after resampling
- **Solution**: Try a longer resampling period or check if timeframe filter is too restrictive

**Issue**: Application doesn't start
- **Solution**: Ensure all dependencies are installed: `pip install PyQt5 pandas numpy matplotlib scipy openpyxl`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Built with PyQt5 for cross-platform GUI
- Data processing powered by pandas and numpy
- Visualization using matplotlib
- Designed for meteorological data analysis but applicable to any time-series CSV data

## Contact

For questions, issues, or suggestions, please open an issue on the GitHub repository.

---

**Note**: This tool is designed for meteorological data but can be adapted for any time-series data analysis from CSV files including sensor data, financial data, IoT measurements, and more.
