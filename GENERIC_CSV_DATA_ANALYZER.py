#!/usr/bin/env python3
"""
Meteorological Data Post-Processing Tool
A PyQt5 GUI application for processing time-stamped CSV files with features for
data averaging, resampling, quality control, calibration, and visualization.

Key Features:
- Multiple CSV file import with automatic datetime detection
- Data resampling and aggregation
- Quality control (outlier removal, interpolation)
- Sensor calibration using standard meteorological methods
- Advanced visualization with multiple plot types
- Column management (rename, remove, reorder)
- Statistical analysis and export capabilities
"""

import sys
import os
import json
import warnings
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QTableWidget, QTableWidgetItem,
                           QFileDialog, QLabel, QComboBox, QSpinBox, QGroupBox,
                           QTextEdit, QTabWidget, QCheckBox, QDoubleSpinBox,
                           QMessageBox, QProgressBar, QHeaderView, QLineEdit,
                           QGridLayout, QListWidget, QListWidgetItem, QInputDialog,
                           QDialog, QDialogButtonBox, QAbstractItemView, QMenu,
                           QDateTimeEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QDateTime
from PyQt5.QtGui import QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure


class DataProcessor(QThread):
    """Worker thread for data processing operations"""
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(pd.DataFrame)
    error = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.files = []
        self.resample_period = 'No Resampling'
        self.agg_method = 'mean'
        self.remove_outliers = False
        self.outlier_std = 3
        self.interpolate_missing = False
        self.interpolate_method = 'linear'
        self.datetime_column = None
        self.datetime_format = 'Auto-detect'
        self.custom_format = None
        self.start_time = None
        self.end_time = None
        self.apply_timeframe = False
        
    def parse_datetime_column(self, df, col_name):
        """Parse datetime column with multiple fallback strategies"""
        original_col = df[col_name].copy()
        
        # Strategy based on user selection
        if self.datetime_format == 'Auto-detect':
            strategies = [
                ('auto', lambda x: pd.to_datetime(x, errors='coerce', format='mixed')),
                ('fallback', lambda x: pd.to_datetime(x, errors='coerce')),
            ]
        elif self.datetime_format == 'ISO8601':
            strategies = [
                ('auto', lambda x: pd.to_datetime(x, errors='coerce', format='mixed')),
                ('fallback', lambda x: pd.to_datetime(x, errors='coerce'))
            ]
        elif self.datetime_format == 'Custom...' and self.custom_format:
            strategies = [
                ('custom', lambda x: pd.to_datetime(x, format=self.custom_format, errors='coerce')),
                ('auto', lambda x: pd.to_datetime(x, errors='coerce', format='mixed'))
            ]
        else:
            # Specific format selected
            strategies = [
                ('specified', lambda x: pd.to_datetime(x, format=self.datetime_format, errors='coerce')),
                ('auto', lambda x: pd.to_datetime(x, errors='coerce', format='mixed'))
            ]
        
        # Try each strategy
        for strategy_name, parser in strategies:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning)
                    parsed = parser(original_col)
                valid_count = parsed.notna().sum()
                total_count = len(original_col)
                
                if valid_count > 0:
                    self.status.emit(f"Parsed {valid_count}/{total_count} dates using {strategy_name}")
                    # Show date range for debugging
                    if valid_count > 0:
                        min_date = parsed.min()
                        max_date = parsed.max()
                        self.status.emit(f"Date range: {min_date} to {max_date}")
                    return parsed
            except Exception as e:
                continue
        
        # If all strategies fail, return NaT
        self.status.emit(f"Warning: Could not parse datetime column {col_name}")
        return pd.Series([pd.NaT] * len(df))
        
    def run(self):
        try:
            if not self.files:
                self.error.emit("No files selected")
                return
                
            all_data = []
            total_files = len(self.files)
            
            # Track if datetime column was auto-detected (needs reset for each file)
            auto_detected_datetime = self.datetime_column is None
            
            # Load and process each file
            for idx, file_path in enumerate(self.files):
                self.status.emit(f"Processing: {os.path.basename(file_path)}")
                
                # Read CSV without automatic datetime parsing
                df = pd.read_csv(file_path)
                
                # Find datetime column if not specified (only for first file)
                if self.datetime_column is None or (auto_detected_datetime and idx == 0):
                    datetime_cols = []
                    for col in df.columns:
                        try:
                            # Try to parse a sample of the column to check if it's datetime
                            sample = df[col].dropna().head(10)
                            if len(sample) > 0:
                                # Try simple parsing first, suppress warnings
                                with warnings.catch_warnings():
                                    warnings.filterwarnings('ignore', category=UserWarning)
                                    test_parse = pd.to_datetime(sample, errors='coerce', format='mixed')
                                    if test_parse.notna().sum() == 0:
                                        test_parse = pd.to_datetime(sample, errors='coerce')
                                if test_parse.notna().sum() > 0:
                                    datetime_cols.append(col)
                        except:
                            continue
                    if datetime_cols:
                        self.datetime_column = datetime_cols[0]
                        self.status.emit(f"Detected datetime column: {self.datetime_column}")
                    else:
                        self.error.emit("No datetime column found. Please specify the datetime column name.")
                        return
                
                # Convert datetime column with robust parsing
                df[self.datetime_column] = self.parse_datetime_column(df, self.datetime_column)
                
                # Check for parsing errors
                null_dates = df[self.datetime_column].isna().sum()
                if null_dates > 0:
                    self.status.emit(f"Warning: {null_dates} datetime values couldn't be parsed in {os.path.basename(file_path)}")
                    if null_dates == len(df):
                        self.error.emit(f"Failed to parse any datetime values in {os.path.basename(file_path)}")
                        return
                
                # Remove rows with invalid datetime
                df = df[df[self.datetime_column].notna()]
                
                # Handle timezone-aware datetimes by converting to UTC then removing timezone
                if df[self.datetime_column].dt.tz is not None:
                    self.status.emit("Converting timezone-aware timestamps to UTC")
                    df[self.datetime_column] = df[self.datetime_column].dt.tz_convert('UTC').dt.tz_localize(None)
                
                # Set datetime as index
                df.set_index(self.datetime_column, inplace=True)
                
                # Remove duplicates
                df = df[~df.index.duplicated(keep='first')]
                
                # Sort by datetime
                df.sort_index(inplace=True)
                
                all_data.append(df)
                self.progress.emit(int((idx + 1) / total_files * 30))
            
            # Combine all dataframes
            self.status.emit("Combining data files...")
            if len(all_data) == 1:
                combined_df = all_data[0]
            else:
                # Ensure all dataframes have consistent timezone handling before combining
                for i, df in enumerate(all_data):
                    if df.index.tz is not None:
                        all_data[i] = df.tz_convert('UTC').tz_localize(None)
                
                combined_df = pd.concat(all_data, axis=0, sort=True)
                combined_df.sort_index(inplace=True)
                
                # Handle duplicate timestamps by averaging
                if combined_df.index.duplicated().any():
                    self.status.emit("Handling duplicate timestamps by averaging...")
                    numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
                    combined_df = combined_df.groupby(level=0)[numeric_cols].mean()
            
            self.progress.emit(40)
            
            # Get numeric columns early for use throughout
            numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
            
            # Apply timeframe filter if requested
            if self.apply_timeframe and self.start_time and self.end_time:
                self.status.emit(f"Filtering to timeframe: {self.start_time} to {self.end_time}")
                before_filter = len(combined_df)
                combined_df = combined_df[(combined_df.index >= self.start_time) & 
                                         (combined_df.index <= self.end_time)]
                after_filter = len(combined_df)
                self.status.emit(f"Retained {after_filter} of {before_filter} records")
                
                if len(combined_df) == 0:
                    self.error.emit("No data in the specified timeframe!")
                    return
            
            # Log data info before quality control
            self.status.emit(f"Combined data: {len(combined_df)} records, {len(numeric_cols)} numeric columns")
            
            # Remove outliers if requested
            if self.remove_outliers:
                self.status.emit("Removing outliers...")
                for col in numeric_cols:
                    mean = combined_df[col].mean()
                    std = combined_df[col].std()
                    lower_bound = mean - (self.outlier_std * std)
                    upper_bound = mean + (self.outlier_std * std)
                    combined_df.loc[(combined_df[col] < lower_bound) | 
                                   (combined_df[col] > upper_bound), col] = np.nan
            
            self.progress.emit(50)
            
            # Interpolate missing values if requested
            if self.interpolate_missing:
                self.status.emit("Interpolating missing values...")
                combined_df = combined_df.interpolate(method=self.interpolate_method)
            
            self.progress.emit(60)
            
            # Resample data or keep original sampling
            if self.resample_period == 'None' or self.resample_period == 'No Resampling':
                # Keep original sampling - just organize the data
                self.status.emit("Keeping original data sampling (no resampling)")
                resampled = combined_df[numeric_cols].copy()
                # Add observation count (always 1 for original data)
                resampled['observation_count'] = 1
                # Reset index to make datetime a column
                resampled.reset_index(inplace=True)
                
            else:
                # Perform resampling
                self.status.emit(f"Resampling to {self.resample_period}...")
                # numeric_cols already defined earlier
                
                if len(numeric_cols) == 0:
                    self.error.emit("No numeric columns found in the data!")
                    return
                
                if self.agg_method == 'mean':
                    resampled = combined_df[numeric_cols].resample(self.resample_period).mean()
                elif self.agg_method == 'median':
                    resampled = combined_df[numeric_cols].resample(self.resample_period).median()
                elif self.agg_method == 'sum':
                    resampled = combined_df[numeric_cols].resample(self.resample_period).sum()
                elif self.agg_method == 'min':
                    resampled = combined_df[numeric_cols].resample(self.resample_period).min()
                elif self.agg_method == 'max':
                    resampled = combined_df[numeric_cols].resample(self.resample_period).max()
                elif self.agg_method == 'std':
                    resampled = combined_df[numeric_cols].resample(self.resample_period).std()
                
                # Add count of observations per period
                counts = combined_df[numeric_cols].resample(self.resample_period).count()
                # Use the maximum count across all columns for each period
                resampled['observation_count'] = counts.max(axis=1)
                
                self.progress.emit(80)
                
                # Check for excessive NaN values and remove empty rows
                rows_before = len(resampled)
                # Drop rows where ALL numeric values are NaN (no observations in that period)
                resampled = resampled[resampled['observation_count'] > 0]
                rows_after = len(resampled)
                
                if rows_before > rows_after:
                    dropped = rows_before - rows_after
                    self.status.emit(f"Removed {dropped} empty time periods (no observations)")
                
                if rows_after == 0:
                    self.error.emit("No data remaining after resampling! Try a different resample period.")
                    return
                
                # Check NaN percentage in the final data
                nan_percentage = (resampled[numeric_cols].isna().sum().sum() / 
                                (len(resampled) * len(numeric_cols)) * 100)
                if nan_percentage > 50:
                    self.status.emit(f"Warning: {nan_percentage:.1f}% of values are NaN after processing")
                
                self.progress.emit(90)
                
                # Reset index to make datetime a column
                resampled.reset_index(inplace=True)
                
                # Rename the index column to something clear
                if self.datetime_column:
                    index_col_name = self.datetime_column
                else:
                    # If we detected it, find the column that looks like a datetime
                    for col in resampled.columns:
                        if pd.api.types.is_datetime64_any_dtype(resampled[col]):
                            index_col_name = col
                            break
                    else:
                        index_col_name = resampled.columns[0]
                
                # Make sure the datetime column has a proper name
                if resampled.columns[0] != index_col_name and pd.api.types.is_datetime64_any_dtype(resampled[resampled.columns[0]]):
                    resampled.rename(columns={resampled.columns[0]: index_col_name}, inplace=True)
            
            self.progress.emit(100)
            self.status.emit("Processing complete!")
            self.finished.emit(resampled)
            
        except Exception as e:
            self.error.emit(str(e))


class ColumnRenameDialog(QDialog):
    """Dialog for renaming multiple columns"""
    def __init__(self, column_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Rename Columns")
        self.setModal(True)
        self.resize(400, 400)
        
        layout = QVBoxLayout()
        
        # Instructions
        layout.addWidget(QLabel("Double-click column names to edit:"))
        
        # Table for renaming
        self.table = QTableWidget()
        self.table.setRowCount(len(column_names))
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Current Name", "New Name"])
        
        for i, col_name in enumerate(column_names):
            # Current name (read-only)
            current_item = QTableWidgetItem(col_name)
            current_item.setFlags(current_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(i, 0, current_item)
            
            # New name (editable)
            new_item = QTableWidgetItem(col_name)
            self.table.setItem(i, 1, new_item)
            
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.setLayout(layout)
        
    def validate_and_accept(self):
        """Validate column names before accepting"""
        new_names = self.get_new_names()
        
        # Check for duplicates
        if len(new_names) != len(set(new_names)):
            QMessageBox.warning(self, "Invalid Names", 
                               "Duplicate column names are not allowed!")
            return
            
        # Check for empty names
        if any(name.strip() == "" for name in new_names):
            QMessageBox.warning(self, "Invalid Names", 
                               "Column names cannot be empty!")
            return
            
        self.accept()
        
    def get_new_names(self):
        """Get the list of new column names"""
        return [self.table.item(i, 1).text() 
                for i in range(self.table.rowCount())]


class ColumnRemoveDialog(QDialog):
    """Dialog for removing multiple columns"""
    def __init__(self, column_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Remove Columns")
        self.setModal(True)
        self.resize(300, 400)
        
        layout = QVBoxLayout()
        
        # Instructions
        layout.addWidget(QLabel("Select columns to remove:"))
        
        # List widget with checkboxes
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.MultiSelection)
        
        for col_name in column_names:
            item = QListWidgetItem(col_name)
            self.list_widget.addItem(item)
            
        layout.addWidget(self.list_widget)
        
        # Select/Deselect all buttons
        btn_layout = QHBoxLayout()
        btn_select_all = QPushButton("Select All")
        btn_select_all.clicked.connect(self.select_all)
        btn_deselect_all = QPushButton("Deselect All")
        btn_deselect_all.clicked.connect(self.deselect_all)
        btn_layout.addWidget(btn_select_all)
        btn_layout.addWidget(btn_deselect_all)
        layout.addLayout(btn_layout)
        
        # Warning label
        self.warning_label = QLabel("")
        self.warning_label.setStyleSheet("color: red;")
        layout.addWidget(self.warning_label)
        
        # Connect selection change
        self.list_widget.itemSelectionChanged.connect(self.update_warning)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.setLayout(layout)
        
    def select_all(self):
        """Select all items"""
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setSelected(True)
            
    def deselect_all(self):
        """Deselect all items"""
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setSelected(False)
            
    def update_warning(self):
        """Update warning label based on selection"""
        selected = len(self.list_widget.selectedItems())
        total = self.list_widget.count()
        
        if selected == total:
            self.warning_label.setText("Warning: All columns selected!")
        elif selected > 0:
            self.warning_label.setText(f"{selected} columns will be removed")
        else:
            self.warning_label.setText("")
            
    def get_columns_to_remove(self):
        """Get list of columns to remove"""
        return [item.text() for item in self.list_widget.selectedItems()]


class ColumnReorderDialog(QDialog):
    """Dialog for reordering columns"""
    def __init__(self, column_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Reorder Columns")
        self.setModal(True)
        self.resize(300, 400)
        
        layout = QVBoxLayout()
        
        # Instructions
        layout.addWidget(QLabel("Drag items to reorder columns:"))
        
        # List widget with drag and drop
        self.list_widget = QListWidget()
        self.list_widget.setDragDropMode(QListWidget.InternalMove)
        
        for col_name in column_names:
            self.list_widget.addItem(col_name)
            
        layout.addWidget(self.list_widget)
        
        # Move buttons
        btn_layout = QHBoxLayout()
        btn_move_up = QPushButton("Move Up")
        btn_move_up.clicked.connect(self.move_up)
        btn_move_down = QPushButton("Move Down")
        btn_move_down.clicked.connect(self.move_down)
        btn_move_top = QPushButton("Move to Top")
        btn_move_top.clicked.connect(self.move_to_top)
        btn_move_bottom = QPushButton("Move to Bottom")
        btn_move_bottom.clicked.connect(self.move_to_bottom)
        
        btn_layout.addWidget(btn_move_up)
        btn_layout.addWidget(btn_move_down)
        btn_layout.addWidget(btn_move_top)
        btn_layout.addWidget(btn_move_bottom)
        layout.addLayout(btn_layout)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.setLayout(layout)
        
    def move_up(self):
        """Move selected item up"""
        current_row = self.list_widget.currentRow()
        if current_row > 0:
            item = self.list_widget.takeItem(current_row)
            self.list_widget.insertItem(current_row - 1, item)
            self.list_widget.setCurrentRow(current_row - 1)
            
    def move_down(self):
        """Move selected item down"""
        current_row = self.list_widget.currentRow()
        if current_row < self.list_widget.count() - 1:
            item = self.list_widget.takeItem(current_row)
            self.list_widget.insertItem(current_row + 1, item)
            self.list_widget.setCurrentRow(current_row + 1)
            
    def move_to_top(self):
        """Move selected item to top"""
        current_row = self.list_widget.currentRow()
        if current_row > 0:
            item = self.list_widget.takeItem(current_row)
            self.list_widget.insertItem(0, item)
            self.list_widget.setCurrentRow(0)
            
    def move_to_bottom(self):
        """Move selected item to bottom"""
        current_row = self.list_widget.currentRow()
        if current_row < self.list_widget.count() - 1:
            item = self.list_widget.takeItem(current_row)
            self.list_widget.addItem(item)
            self.list_widget.setCurrentRow(self.list_widget.count() - 1)
            
    def get_column_order(self):
        """Get the new column order"""
        return [self.list_widget.item(i).text() 
                for i in range(self.list_widget.count())]


class CalibrationEditorDialog(QDialog):
    """Dialog for viewing and editing calibration coefficients"""
    def __init__(self, calibration_coefficients, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibration Coefficients Editor")
        self.setModal(True)
        self.resize(600, 400)
        
        self.calibration_coefficients = calibration_coefficients.copy()
        
        layout = QVBoxLayout()
        
        # Instructions
        instructions = QLabel("Edit calibration coefficients. Double-click values to modify.")
        instructions.setStyleSheet("QLabel { font-weight: bold; }")
        layout.addWidget(instructions)
        
        # Table for editing
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Column", "Parameter", "Value"])
        layout.addWidget(self.table)
        
        # Populate table
        self.populate_table()
        
        # Add/Remove buttons
        btn_layout = QHBoxLayout()
        self.btn_add_column = QPushButton("Add Column Calibration")
        self.btn_add_column.clicked.connect(self.add_column_calibration)
        btn_layout.addWidget(self.btn_add_column)
        
        self.btn_remove_column = QPushButton("Remove Selected")
        self.btn_remove_column.clicked.connect(self.remove_selected)
        btn_layout.addWidget(self.btn_remove_column)
        
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept_changes)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.setLayout(layout)
        
    def populate_table(self):
        """Populate the table with calibration data"""
        self.table.setRowCount(0)
        
        row = 0
        for column_name, coeffs in self.calibration_coefficients.items():
            for param_name, value in coeffs.items():
                self.table.insertRow(row)
                
                # Column name (read-only for first param, merged for others)
                col_item = QTableWidgetItem(column_name)
                col_item.setFlags(col_item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(row, 0, col_item)
                
                # Parameter name (read-only)
                param_item = QTableWidgetItem(param_name)
                param_item.setFlags(param_item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(row, 1, param_item)
                
                # Value (editable)
                value_item = QTableWidgetItem(str(value))
                self.table.setItem(row, 2, value_item)
                
                row += 1
        
        self.table.resizeColumnsToContents()
        
    def add_column_calibration(self):
        """Add a new column calibration"""
        column_name, ok = QInputDialog.getText(self, "Add Column", 
                                               "Enter column name:")
        if ok and column_name:
            if column_name in self.calibration_coefficients:
                QMessageBox.warning(self, "Duplicate", 
                                  f"Calibration for '{column_name}' already exists!")
                return
            
            # Add default linear calibration
            self.calibration_coefficients[column_name] = {
                'gain': 1.0,
                'offset': 0.0
            }
            self.populate_table()
            
    def remove_selected(self):
        """Remove selected calibration entries"""
        selected_rows = set(item.row() for item in self.table.selectedItems())
        if not selected_rows:
            QMessageBox.warning(self, "No Selection", "Please select rows to remove")
            return
        
        # Get column names to remove
        columns_to_remove = set()
        for row in selected_rows:
            col_name = self.table.item(row, 0).text()
            columns_to_remove.add(col_name)
        
        reply = QMessageBox.question(self, "Confirm Removal",
                                    f"Remove calibration for {len(columns_to_remove)} column(s)?",
                                    QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            for col_name in columns_to_remove:
                if col_name in self.calibration_coefficients:
                    del self.calibration_coefficients[col_name]
            self.populate_table()
            
    def accept_changes(self):
        """Validate and accept changes"""
        try:
            # Update coefficients from table
            new_coeffs = {}
            current_column = None
            
            for row in range(self.table.rowCount()):
                col_name = self.table.item(row, 0).text()
                param_name = self.table.item(row, 1).text()
                value_text = self.table.item(row, 2).text()
                
                # Try to convert to float
                try:
                    value = float(value_text)
                except ValueError:
                    QMessageBox.warning(self, "Invalid Value", 
                                      f"Invalid value for {col_name}.{param_name}: {value_text}")
                    return
                
                if col_name not in new_coeffs:
                    new_coeffs[col_name] = {}
                new_coeffs[col_name][param_name] = value
            
            self.calibration_coefficients = new_coeffs
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error validating changes: {str(e)}")
            
    def get_calibration_coefficients(self):
        """Get the updated calibration coefficients"""
        return self.calibration_coefficients


class PlotCanvas(FigureCanvas):
    """Matplotlib canvas for plotting"""
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 6))
        super().__init__(self.fig)
        self.setParent(parent)
        
    def plot_data(self, df, columns, datetime_col, plot_type='single',  
                 normalize=False, show_grid=True, show_legend=True,
                 rolling_avg=False, rolling_window=10,
                 left_columns=None, right_columns=None):
        """Plot selected columns with various options"""
        self.fig.clear()
        
        if len(columns) == 0:
            return
        
        # Prepare data
        plot_df = df.copy()
        
        # Apply rolling average if requested
        if rolling_avg:
            for col in columns:
                if col in plot_df.columns:
                    plot_df[f"{col}_raw"] = plot_df[col]
                    plot_df[col] = plot_df[col].rolling(window=rolling_window, center=True).mean()
        
        # Normalize data if requested
        if normalize:
            for col in columns:
                if col in plot_df.columns:
                    col_data = plot_df[col]
                    plot_df[col] = (col_data - col_data.min()) / (col_data.max() - col_data.min())
        
        if plot_type == 'subplots':
            # Create separate subplots
            n_plots = len(columns)
            axes = self.fig.subplots(n_plots, 1, sharex=True)
            if n_plots == 1:
                axes = [axes]
            
            for idx, col in enumerate(columns):
                axes[idx].plot(plot_df[datetime_col], plot_df[col], label=col, linewidth=1.5)
                if rolling_avg and f"{col}_raw" in plot_df.columns:
                    axes[idx].plot(plot_df[datetime_col], plot_df[f"{col}_raw"], 
                                 alpha=0.3, label=f"{col} (raw)")
                axes[idx].set_ylabel(col if not normalize else f"{col} (normalized)")
                if show_legend:
                    axes[idx].legend(loc='upper right')
                if show_grid:
                    axes[idx].grid(True, alpha=0.3)
            
            axes[-1].set_xlabel('Date/Time')
            axes[-1].tick_params(axis='x', rotation=45)
            self.fig.suptitle('Meteorological Data Visualization')
            
        elif plot_type == 'single':
            # Plot all on single axis
            ax = self.fig.add_subplot(111)
            
            for col in columns:
                ax.plot(plot_df[datetime_col], plot_df[col], label=col, linewidth=1.5)
                if rolling_avg and f"{col}_raw" in plot_df.columns:
                    ax.plot(plot_df[datetime_col], plot_df[f"{col}_raw"], 
                          alpha=0.3, linestyle='--', label=f"{col} (raw)")
            
            ax.set_xlabel('Date/Time')
            ax.set_ylabel('Value' + (' (normalized)' if normalize else ''))
            ax.tick_params(axis='x', rotation=45)
            if show_grid:
                ax.grid(True, alpha=0.3)
            if show_legend:
                ax.legend(loc='best')
            self.fig.suptitle('Meteorological Data - Combined Plot')
            
        elif plot_type == 'dual':
            # Dual Y-axis plot
            ax1 = self.fig.add_subplot(111)
            ax2 = ax1.twinx()
            
            # Plot left axis columns
            if left_columns:
                for col in left_columns:
                    if col in plot_df.columns:
                        line1 = ax1.plot(plot_df[datetime_col], plot_df[col], 
                                        label=col, linewidth=1.5)
                        if rolling_avg and f"{col}_raw" in plot_df.columns:
                            ax1.plot(plot_df[datetime_col], plot_df[f"{col}_raw"], 
                                   alpha=0.3, linestyle='--', label=f"{col} (raw)")
            
            # Plot right axis columns
            if right_columns:
                for col in right_columns:
                    if col in plot_df.columns:
                        line2 = ax2.plot(plot_df[datetime_col], plot_df[col], 
                                        label=col, linewidth=1.5, linestyle='-.')
                        if rolling_avg and f"{col}_raw" in plot_df.columns:
                            ax2.plot(plot_df[datetime_col], plot_df[f"{col}_raw"], 
                                   alpha=0.3, linestyle=':', label=f"{col} (raw)")
            
            ax1.set_xlabel('Date/Time')
            ax1.set_ylabel('Left Y-Axis' + (' (normalized)' if normalize else ''), color='b')
            ax2.set_ylabel('Right Y-Axis' + (' (normalized)' if normalize else ''), color='r')
            ax1.tick_params(axis='x', rotation=45)
            ax1.tick_params(axis='y', colors='b')
            ax2.tick_params(axis='y', colors='r')
            
            if show_grid:
                ax1.grid(True, alpha=0.3)
            
            if show_legend:
                # Combine legends
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
            
            self.fig.suptitle('Meteorological Data - Dual Y-Axis Plot')
        
        self.fig.tight_layout()
        self.draw()


class MeteoDataProcessor(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.file_list = []
        self.processed_data = None
        self.calibration_coefficients = {}
        self.calculated_calibration = {}
        self.processor_thread = DataProcessor()
        self.processor_thread.progress.connect(self.update_progress)
        self.processor_thread.status.connect(self.update_status)
        self.processor_thread.finished.connect(self.processing_finished)
        self.processor_thread.error.connect(self.show_error)
        
        # Initialize UI elements that might be referenced before creation
        self.btn_rename_column = None
        self.btn_remove_columns = None
        self.btn_reorder_columns = None
        self.data_info_label = None
        self.dual_axis_group = None
        self.left_axis_list = None
        self.right_axis_list = None
        self.plot_type_combo = None
        self.normalize_check = None
        self.show_grid_check = None
        self.show_legend_check = None
        self.rolling_avg_check = None
        self.rolling_window_spin = None
        self.detected_date_range_label = None
        self.start_datetime = None
        self.end_datetime = None
        self.apply_timeframe_check = None
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle('Meteorological Data Post-Processing Tool')
        # Get screen geometry and set window to 85% of screen size
        screen = QApplication.desktop().screenGeometry()
        width = int(screen.width() * 0.85)
        height = int(screen.height() * 0.65)
        # Center the window
        x = (screen.width() - width) // 2
        y = (screen.height() - height) // 2
        self.setGeometry(x, y, width, height)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # File Import Tab
        self.create_import_tab()
        
        # Processing Tab
        self.create_processing_tab()
        
        # Data View Tab
        self.create_data_tab()
        
        # Visualization Tab
        self.create_visualization_tab()
        
        # Calibration Tab
        self.create_calibration_tab()
        
        # Statistics Tab
        self.create_statistics_tab()
        
        # Status bar and progress
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.progress_bar)
        main_layout.addLayout(status_layout)
        
    def create_import_tab(self):
        """Create the file import tab"""
        import_widget = QWidget()
        layout = QVBoxLayout(import_widget)
        
        # File selection group
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout()
        
        btn_layout = QHBoxLayout()
        self.btn_add_files = QPushButton("Add CSV Files")
        self.btn_add_files.clicked.connect(self.add_files)
        self.btn_clear_files = QPushButton("Clear List")
        self.btn_clear_files.clicked.connect(self.clear_files)
        btn_layout.addWidget(self.btn_add_files)
        btn_layout.addWidget(self.btn_clear_files)
        btn_layout.addStretch()
        file_layout.addLayout(btn_layout)
        
        # File list
        self.file_list_widget = QListWidget()
        file_layout.addWidget(self.file_list_widget)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Column detection group
        col_group = QGroupBox("Column Settings")
        col_layout = QGridLayout()
        
        col_layout.addWidget(QLabel("DateTime Column:"), 0, 0)
        self.datetime_col_input = QLineEdit()
        self.datetime_col_input.setPlaceholderText("Auto-detect if empty")
        col_layout.addWidget(self.datetime_col_input, 0, 1)
        
        col_layout.addWidget(QLabel("DateTime Format:"), 1, 0)
        self.datetime_format_combo = QComboBox()
        self.datetime_format_combo.addItems([
            'Auto-detect',
            'ISO8601',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M:%S.%f',
            '%Y-%m-%d %H:%M:%S%z',
            '%Y-%m-%d %H:%M:%S.%f%z',
            '%m/%d/%Y %H:%M:%S',
            '%d/%m/%Y %H:%M:%S',
            'Custom...'
        ])
        col_layout.addWidget(self.datetime_format_combo, 1, 1)
        
        self.custom_format_input = QLineEdit()
        self.custom_format_input.setPlaceholderText("Enter custom format (e.g., %Y-%m-%d %H:%M:%S)")
        self.custom_format_input.setEnabled(False)
        col_layout.addWidget(self.custom_format_input, 2, 0, 1, 2)
        
        # Enable custom format input when "Custom..." is selected
        self.datetime_format_combo.currentTextChanged.connect(
            lambda text: self.custom_format_input.setEnabled(text == "Custom...")
        )
        
        col_group.setLayout(col_layout)
        layout.addWidget(col_group)
        
        # Detected date range group
        date_range_group = QGroupBox("Detected Data Range")
        date_range_layout = QVBoxLayout()
        
        self.detected_date_range_label = QLabel("Import files to see date range")
        self.detected_date_range_label.setStyleSheet("QLabel { color: #0066cc; font-weight: bold; }")
        date_range_layout.addWidget(self.detected_date_range_label)
        
        self.btn_scan_dates = QPushButton("Scan Files for Date Range")
        self.btn_scan_dates.clicked.connect(self.scan_file_dates)
        self.btn_scan_dates.setEnabled(False)
        date_range_layout.addWidget(self.btn_scan_dates)
        
        date_range_group.setLayout(date_range_layout)
        layout.addWidget(date_range_group)
        
        self.tabs.addTab(import_widget, "Import Files")
        
    def create_processing_tab(self):
        """Create the processing options tab"""
        process_widget = QWidget()
        layout = QVBoxLayout(process_widget)
        
        # Resampling group
        resample_group = QGroupBox("Resampling Options")
        resample_layout = QGridLayout()
        
        resample_layout.addWidget(QLabel("Resample Period:"), 0, 0)
        self.resample_combo = QComboBox()
        self.resample_combo.addItems(['No Resampling', '1min', '5min', '10min', '15min', '30min', 
                                     '1H', '2H', '3H', '6H', '12H', '1D'])
        self.resample_combo.setCurrentText('No Resampling')
        resample_layout.addWidget(self.resample_combo, 0, 1)
        
        resample_note = QLabel("Tip: Use 'No Resampling' to keep original data, or use a larger period if data is sparse")
        resample_note.setStyleSheet("QLabel { color: #666; font-size: 9px; font-style: italic; }")
        resample_layout.addWidget(resample_note, 1, 0, 1, 2)
        
        resample_layout.addWidget(QLabel("Aggregation Method:"), 2, 0)
        self.agg_combo = QComboBox()
        self.agg_combo.addItems(['mean', 'median', 'sum', 'min', 'max', 'std'])
        resample_layout.addWidget(self.agg_combo, 2, 1)
        
        agg_note = QLabel("Note: Aggregation is not used when 'No Resampling' is selected")
        agg_note.setStyleSheet("QLabel { color: #666; font-size: 9px; font-style: italic; }")
        resample_layout.addWidget(agg_note, 3, 0, 1, 2)
        
        # Enable/disable aggregation based on resampling
        def toggle_agg_method(text):
            self.agg_combo.setEnabled(text != 'No Resampling')
        self.resample_combo.currentTextChanged.connect(toggle_agg_method)
        
        resample_group.setLayout(resample_layout)
        layout.addWidget(resample_group)
        
        # Timeframe filter group
        timeframe_group = QGroupBox("Timeframe Filter")
        timeframe_layout = QGridLayout()
        
        self.apply_timeframe_check = QCheckBox("Apply Timeframe Filter")
        self.apply_timeframe_check.toggled.connect(self.toggle_timeframe_controls)
        timeframe_layout.addWidget(self.apply_timeframe_check, 0, 0, 1, 2)
        
        timeframe_layout.addWidget(QLabel("Start Time:"), 1, 0)
        self.start_datetime = QDateTimeEdit()
        self.start_datetime.setCalendarPopup(True)
        self.start_datetime.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.start_datetime.setDateTime(QDateTime.currentDateTime().addDays(-7))  # Default to 7 days ago
        self.start_datetime.setEnabled(False)
        timeframe_layout.addWidget(self.start_datetime, 1, 1)
        
        timeframe_layout.addWidget(QLabel("End Time:"), 2, 0)
        self.end_datetime = QDateTimeEdit()
        self.end_datetime.setCalendarPopup(True)
        self.end_datetime.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.end_datetime.setDateTime(QDateTime.currentDateTime())  # Default to now
        self.end_datetime.setEnabled(False)
        timeframe_layout.addWidget(self.end_datetime, 2, 1)
        
        timeframe_note = QLabel("Note: Timeframe will be auto-populated from imported files")
        timeframe_note.setStyleSheet("QLabel { color: #666; font-size: 9px; font-style: italic; }")
        timeframe_layout.addWidget(timeframe_note, 3, 0, 1, 2)
        
        timeframe_group.setLayout(timeframe_layout)
        layout.addWidget(timeframe_group)
        
        # Quality control group
        qc_group = QGroupBox("Quality Control")
        qc_layout = QGridLayout()
        
        self.outlier_check = QCheckBox("Remove Outliers")
        qc_layout.addWidget(self.outlier_check, 0, 0)
        
        qc_layout.addWidget(QLabel("Outlier Threshold (σ):"), 0, 1)
        self.outlier_spin = QDoubleSpinBox()
        self.outlier_spin.setRange(1, 5)
        self.outlier_spin.setValue(3)
        self.outlier_spin.setSingleStep(0.5)
        qc_layout.addWidget(self.outlier_spin, 0, 2)
        
        self.interpolate_check = QCheckBox("Interpolate Missing Values")
        qc_layout.addWidget(self.interpolate_check, 1, 0)
        
        qc_layout.addWidget(QLabel("Interpolation Method:"), 1, 1)
        self.interpolate_combo = QComboBox()
        self.interpolate_combo.addItems(['linear', 'time', 'nearest', 'cubic', 'spline'])
        qc_layout.addWidget(self.interpolate_combo, 1, 2)
        
        qc_group.setLayout(qc_layout)
        layout.addWidget(qc_group)
        
        # Process button
        self.btn_process = QPushButton("Process Data")
        self.btn_process.clicked.connect(self.process_data)
        self.btn_process.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; "
                                      "font-weight: bold; padding: 10px; }")
        layout.addWidget(self.btn_process)
        
        # Add a note about calibration
        note_label = QLabel("Note: After processing, use the Calibration tab to apply sensor calibrations")
        note_label.setStyleSheet("QLabel { color: #666; font-style: italic; }")
        layout.addWidget(note_label)
        
        layout.addStretch()
        self.tabs.addTab(process_widget, "Processing Options")
        
    def create_data_tab(self):
        """Create the data view tab"""
        data_widget = QWidget()
        layout = QVBoxLayout(data_widget)
        
        # Top button layout
        top_btn_layout = QHBoxLayout()
        
        # Column operations group
        col_ops_group = QGroupBox("Column Operations")
        col_ops_layout = QHBoxLayout()
        
        self.btn_rename_column = QPushButton("Rename Column")
        self.btn_rename_column.clicked.connect(self.rename_column)
        self.btn_rename_column.setEnabled(False)
        
        self.btn_remove_columns = QPushButton("Remove Columns")
        self.btn_remove_columns.clicked.connect(self.remove_columns)
        self.btn_remove_columns.setEnabled(False)
        
        self.btn_reorder_columns = QPushButton("Reorder Columns")
        self.btn_reorder_columns.clicked.connect(self.reorder_columns)
        self.btn_reorder_columns.setEnabled(False)
        
        col_ops_layout.addWidget(self.btn_rename_column)
        col_ops_layout.addWidget(self.btn_remove_columns)
        col_ops_layout.addWidget(self.btn_reorder_columns)
        col_ops_group.setLayout(col_ops_layout)
        
        # Export buttons group
        export_group = QGroupBox("Export Options")
        export_layout = QHBoxLayout()
        
        self.btn_export_csv = QPushButton("Export to CSV")
        self.btn_export_csv.clicked.connect(self.export_csv)
        self.btn_export_csv.setEnabled(False)
        
        self.btn_export_excel = QPushButton("Export to Excel")
        self.btn_export_excel.clicked.connect(self.export_excel)
        self.btn_export_excel.setEnabled(False)
        
        export_layout.addWidget(self.btn_export_csv)
        export_layout.addWidget(self.btn_export_excel)
        export_group.setLayout(export_layout)
        
        top_btn_layout.addWidget(col_ops_group)
        top_btn_layout.addWidget(export_group)
        top_btn_layout.addStretch()
        layout.addLayout(top_btn_layout)
        
        # Data table with context menu
        self.data_table = QTableWidget()
        self.data_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.data_table.customContextMenuRequested.connect(self.show_column_context_menu)
        layout.addWidget(self.data_table)
        
        # Info label
        self.data_info_label = QLabel("No data loaded")
        layout.addWidget(self.data_info_label)
        
        self.tabs.addTab(data_widget, "Data View")
        
    def create_visualization_tab(self):
        """Create the visualization tab"""
        viz_widget = QWidget()
        layout = QHBoxLayout(viz_widget)
        
        # Left panel for controls
        control_panel = QVBoxLayout()
        
        # Plot type selection
        plot_type_group = QGroupBox("Plot Type")
        plot_type_layout = QVBoxLayout()
        
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(['Separate Subplots', 'Single Plot', 'Dual Y-Axis'])
        self.plot_type_combo.setCurrentText('Single Plot')
        self.plot_type_combo.currentTextChanged.connect(self.on_plot_type_changed)
        plot_type_layout.addWidget(self.plot_type_combo)
        
        plot_type_group.setLayout(plot_type_layout)
        control_panel.addWidget(plot_type_group)
        
        # Column selection
        col_group = QGroupBox("Select Columns")
        col_layout = QVBoxLayout()
        
        self.plot_list = QListWidget()
        self.plot_list.setSelectionMode(QListWidget.MultiSelection)
        col_layout.addWidget(self.plot_list)
        
        col_group.setLayout(col_layout)
        control_panel.addWidget(col_group)
        
        # Y-axis assignment for dual axis mode
        self.dual_axis_group = QGroupBox("Y-Axis Assignment")
        dual_axis_layout = QVBoxLayout()
        
        dual_axis_layout.addWidget(QLabel("Left Y-Axis:"))
        self.left_axis_list = QListWidget()
        self.left_axis_list.setMaximumHeight(80)
        dual_axis_layout.addWidget(self.left_axis_list)
        
        dual_axis_layout.addWidget(QLabel("Right Y-Axis:"))
        self.right_axis_list = QListWidget()
        self.right_axis_list.setMaximumHeight(80)
        dual_axis_layout.addWidget(self.right_axis_list)
        
        self.dual_axis_group.setLayout(dual_axis_layout)
        self.dual_axis_group.setVisible(False)
        control_panel.addWidget(self.dual_axis_group)
        
        # Plot options
        options_group = QGroupBox("Plot Options")
        options_layout = QGridLayout()
        
        self.show_grid_check = QCheckBox("Show Grid")
        self.show_grid_check.setChecked(True)
        options_layout.addWidget(self.show_grid_check, 0, 0)
        
        self.show_legend_check = QCheckBox("Show Legend")
        self.show_legend_check.setChecked(True)
        options_layout.addWidget(self.show_legend_check, 0, 1)
        
        self.normalize_check = QCheckBox("Normalize Data")
        options_layout.addWidget(self.normalize_check, 1, 0)
        
        self.rolling_avg_check = QCheckBox("Rolling Average")
        options_layout.addWidget(self.rolling_avg_check, 1, 1)
        
        self.rolling_window_spin = QSpinBox()
        self.rolling_window_spin.setRange(2, 100)
        self.rolling_window_spin.setValue(10)
        self.rolling_window_spin.setPrefix("Window: ")
        self.rolling_window_spin.setEnabled(False)
        options_layout.addWidget(self.rolling_window_spin, 2, 0, 1, 2)
        
        self.rolling_avg_check.toggled.connect(self.rolling_window_spin.setEnabled)
        
        options_group.setLayout(options_layout)
        control_panel.addWidget(options_group)
        
        # Buttons
        self.btn_plot = QPushButton("Update Plot")
        self.btn_plot.clicked.connect(self.update_plot)
        self.btn_plot.setEnabled(False)
        control_panel.addWidget(self.btn_plot)
        
        self.btn_save_plot = QPushButton("Save Plot")
        self.btn_save_plot.clicked.connect(self.save_plot)
        self.btn_save_plot.setEnabled(False)
        control_panel.addWidget(self.btn_save_plot)
        
        control_panel.addStretch()
        layout.addLayout(control_panel, 1)
        
        # Plot canvas with toolbar
        plot_container = QVBoxLayout()
        self.plot_canvas = PlotCanvas()
        self.plot_toolbar = NavigationToolbar2QT(self.plot_canvas, viz_widget)
        plot_container.addWidget(self.plot_toolbar)
        plot_container.addWidget(self.plot_canvas)
        
        # Create a widget to hold the plot container
        plot_widget = QWidget()
        plot_widget.setLayout(plot_container)
        layout.addWidget(plot_widget, 3)
        
        self.tabs.addTab(viz_widget, "Visualization")
        
    def create_calibration_tab(self):
        """Create the calibration tab for sensor calibration"""
        calib_widget = QWidget()
        layout = QVBoxLayout(calib_widget)
        
        # ===== SECTION 1: Calibration File Management =====
        file_mgmt_group = QGroupBox("Calibration File Management")
        file_mgmt_layout = QHBoxLayout()
        
        self.btn_import_calib = QPushButton("Import Calibration File")
        self.btn_import_calib.clicked.connect(self.import_calibration_file)
        self.btn_import_calib.setToolTip("Import calibration coefficients from a JSON file")
        file_mgmt_layout.addWidget(self.btn_import_calib)
        
        self.btn_view_edit_calib = QPushButton("View/Edit Calibration")
        self.btn_view_edit_calib.clicked.connect(self.view_edit_calibration)
        self.btn_view_edit_calib.setEnabled(False)
        self.btn_view_edit_calib.setToolTip("View and manually edit calibration coefficients")
        file_mgmt_layout.addWidget(self.btn_view_edit_calib)
        
        self.btn_export_calib = QPushButton("Export Calibration")
        self.btn_export_calib.clicked.connect(self.export_calibration)
        self.btn_export_calib.setEnabled(False)
        self.btn_export_calib.setToolTip("Export calibration coefficients to JSON file for future use")
        file_mgmt_layout.addWidget(self.btn_export_calib)
        
        self.btn_clear_calib = QPushButton("Clear Calibration")
        self.btn_clear_calib.clicked.connect(self.clear_calibration)
        self.btn_clear_calib.setEnabled(False)
        self.btn_clear_calib.setToolTip("Clear all loaded calibration coefficients")
        file_mgmt_layout.addWidget(self.btn_clear_calib)
        
        file_mgmt_layout.addStretch()
        file_mgmt_group.setLayout(file_mgmt_layout)
        layout.addWidget(file_mgmt_group)
        
        # Current calibration status
        self.calib_status_label = QLabel("No calibration loaded")
        self.calib_status_label.setStyleSheet("QLabel { color: #666; font-style: italic; }")
        layout.addWidget(self.calib_status_label)
        
        # Separator
        separator1 = QLabel("─" * 100)
        separator1.setStyleSheet("QLabel { color: #ccc; }")
        layout.addWidget(separator1)
        
        # ===== SECTION 2: Data Selection =====
        data_selection_group = QGroupBox("Step 1: Select Data to Calibrate")
        data_selection_layout = QGridLayout()
        
        # Target columns selection
        data_selection_layout.addWidget(QLabel("Columns to Calibrate:"), 0, 0, Qt.AlignTop)
        
        target_container = QVBoxLayout()
        self.calib_target_list = QListWidget()
        self.calib_target_list.setSelectionMode(QListWidget.MultiSelection)
        self.calib_target_list.setMaximumHeight(120)
        self.calib_target_list.itemSelectionChanged.connect(self.on_target_selection_changed)
        target_container.addWidget(self.calib_target_list)
        
        # Quick selection buttons
        target_btn_layout = QHBoxLayout()
        btn_select_all_targets = QPushButton("Select All")
        btn_select_all_targets.clicked.connect(lambda: self.calib_target_list.selectAll())
        btn_clear_targets = QPushButton("Clear Selection")
        btn_clear_targets.clicked.connect(lambda: self.calib_target_list.clearSelection())
        target_btn_layout.addWidget(btn_select_all_targets)
        target_btn_layout.addWidget(btn_clear_targets)
        target_btn_layout.addStretch()
        target_container.addLayout(target_btn_layout)
        
        data_selection_layout.addLayout(target_container, 0, 1)
        
        # Reference column selection (for methods that need it)
        data_selection_layout.addWidget(QLabel("Reference Column:"), 1, 0)
        ref_container = QHBoxLayout()
        self.ref_column_combo = QComboBox()
        self.ref_column_combo.setMinimumWidth(200)
        ref_container.addWidget(self.ref_column_combo)
        self.ref_column_label = QLabel("(Used for linear regression methods)")
        self.ref_column_label.setStyleSheet("QLabel { color: #666; font-style: italic; font-size: 9px; }")
        ref_container.addWidget(self.ref_column_label)
        ref_container.addStretch()
        data_selection_layout.addLayout(ref_container, 1, 1)
        
        data_selection_group.setLayout(data_selection_layout)
        layout.addWidget(data_selection_group)
        
        # ===== SECTION 3: Calibration Method Selection =====
        method_group = QGroupBox("Step 2: Select Calibration Method and Parameters")
        method_layout = QVBoxLayout()
        
        # Method selection
        method_select_layout = QHBoxLayout()
        method_select_layout.addWidget(QLabel("Calibration Method:"))
        self.calib_method_combo = QComboBox()
        self.calib_method_combo.addItems([
            'Linear Regression (y = mx + b)',
            'Manual Coefficients (y = mx + b)',
            'Temperature Compensation',
            'Humidity Correction',
            'Pressure/Altitude Correction',
            'Wind Speed Height Correction'
        ])
        self.calib_method_combo.currentTextChanged.connect(self.on_calibration_method_changed)
        self.calib_method_combo.setMinimumWidth(300)
        method_select_layout.addWidget(self.calib_method_combo)
        method_select_layout.addStretch()
        method_layout.addLayout(method_select_layout)
        
        # Method description
        self.method_description = QLabel()
        self.method_description.setWordWrap(True)
        self.method_description.setStyleSheet("QLabel { background-color: #e3f2fd; padding: 8px; "
                                             "border-radius: 4px; color: #01579b; }")
        method_layout.addWidget(self.method_description)
        
        # Parameters section (changes based on method)
        self.params_container = QWidget()
        self.params_layout = QVBoxLayout(self.params_container)
        self.params_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create all parameter widgets
        self.create_calibration_parameter_widgets()
        
        method_layout.addWidget(self.params_container)
        
        # Calculate button
        calc_btn_layout = QHBoxLayout()
        self.btn_calculate_calib = QPushButton("Calculate Calibration Coefficients")
        self.btn_calculate_calib.clicked.connect(self.calculate_calibration_coefficients)
        self.btn_calculate_calib.setEnabled(False)
        self.btn_calculate_calib.setStyleSheet("QPushButton { background-color: #FF9800; color: white; "
                                              "font-weight: bold; padding: 10px; font-size: 11pt; }")
        self.btn_calculate_calib.setToolTip("Calculate calibration coefficients for each selected column")
        calc_btn_layout.addWidget(self.btn_calculate_calib)
        calc_btn_layout.addStretch()
        method_layout.addLayout(calc_btn_layout)
        
        method_group.setLayout(method_layout)
        layout.addWidget(method_group)
        
        # ===== SECTION 4: Calibration Results Table =====
        results_group = QGroupBox("Step 3: Review Calibration Coefficients")
        results_layout = QVBoxLayout()
        
        results_info = QLabel("Calibration coefficients calculated for each column:")
        results_info.setStyleSheet("QLabel { font-weight: bold; color: #01579b; }")
        results_layout.addWidget(results_info)
        
        # Table to show calibration for each column
        self.calib_table = QTableWidget()
        self.calib_table.setMinimumHeight(200)
        self.calib_table.setAlternatingRowColors(True)
        self.calib_table.horizontalHeader().setStretchLastSection(True)
        self.calib_table.setSelectionBehavior(QTableWidget.SelectRows)
        results_layout.addWidget(self.calib_table)
        
        # Summary text
        self.calib_summary_label = QLabel("Click 'Calculate Calibration Coefficients' to see results")
        self.calib_summary_label.setStyleSheet("QLabel { color: #666; font-style: italic; padding: 5px; }")
        results_layout.addWidget(self.calib_summary_label)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        # ===== SECTION 5: Apply Actions =====
        apply_group = QGroupBox("Step 4: Apply Calibration to Data")
        apply_layout = QVBoxLayout()
        
        warning_label = QLabel("ℹ Note: Applying calibration will create new columns with '(CALd)' suffix "
                              "containing calibrated values. Original data is preserved.")
        warning_label.setStyleSheet("QLabel { background-color: #d1ecf1; color: #0c5460; "
                                    "padding: 8px; border-radius: 4px; font-weight: bold; }")
        warning_label.setWordWrap(True)
        apply_layout.addWidget(warning_label)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.btn_apply_calculated = QPushButton("Apply Calculated Calibration")
        self.btn_apply_calculated.clicked.connect(self.apply_calculated_calibration)
        self.btn_apply_calculated.setEnabled(False)
        self.btn_apply_calculated.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; "
                                               "font-weight: bold; padding: 10px; font-size: 11pt; }")
        self.btn_apply_calculated.setToolTip("Apply the calculated calibration to the selected columns")
        button_layout.addWidget(self.btn_apply_calculated)
        
        self.btn_apply_loaded_calib = QPushButton("Apply Loaded Calibration File")
        self.btn_apply_loaded_calib.clicked.connect(self.apply_loaded_calibration)
        self.btn_apply_loaded_calib.setEnabled(False)
        self.btn_apply_loaded_calib.setStyleSheet("QPushButton { background-color: #2196F3; color: white; "
                                                 "font-weight: bold; padding: 10px; font-size: 11pt; }")
        self.btn_apply_loaded_calib.setToolTip("Apply the imported calibration file to matching columns")
        button_layout.addWidget(self.btn_apply_loaded_calib)
        
        apply_layout.addLayout(button_layout)
        apply_group.setLayout(apply_layout)
        layout.addWidget(apply_group)
        
        # Help section
        help_group = QGroupBox("Quick Guide")
        help_layout = QVBoxLayout()
        help_text = QLabel(
            "<b>Workflow A - Calculate New Calibration:</b><br>"
            "1. Select columns to calibrate and reference column (if needed)<br>"
            "2. Choose calibration method and set parameters<br>"
            "3. Click 'Calculate Calibration Coefficients' to see what will be applied<br>"
            "4. Review the table to verify coefficients for each column<br>"
            "5. Click 'Apply Calculated Calibration' to modify data<br><br>"
            "<b>Workflow B - Use Existing Calibration File:</b><br>"
            "1. Click 'Import Calibration File' to load saved coefficients<br>"
            "2. Review which columns match your data<br>"
            "3. Click 'Apply Loaded Calibration File'<br><br>"
            "<b>Export:</b> After applying, use 'Export Calibration' to save coefficients for future use"
        )
        help_text.setWordWrap(True)
        help_text.setStyleSheet("QLabel { background-color: #f5f5f5; padding: 10px; "
                               "border-radius: 4px; font-size: 9px; }")
        help_layout.addWidget(help_text)
        help_group.setLayout(help_layout)
        layout.addWidget(help_group)
        
        layout.addStretch()
        self.tabs.addTab(calib_widget, "Calibration")
        
        # Initialize with first method
        self.on_calibration_method_changed(self.calib_method_combo.currentText())

    def create_calibration_parameter_widgets(self):
        """Create all parameter widgets for different calibration methods"""
        
        # ===== Linear Regression Parameters =====
        self.linear_params_widget = QWidget()
        linear_layout = QGridLayout(self.linear_params_widget)
        linear_layout.setContentsMargins(10, 10, 10, 10)
        
        info_label = QLabel("Calculates individual slope (m) and offset (b) for each selected column by comparing to the reference column.")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("QLabel { color: #666; font-style: italic; }")
        linear_layout.addWidget(info_label, 0, 0, 1, 2)
        
        self.force_zero_check = QCheckBox("Force intercept through zero (b = 0)")
        linear_layout.addWidget(self.force_zero_check, 1, 0, 1, 2)
        
        linear_layout.setRowStretch(2, 1)
        
        # ===== Manual Coefficients Parameters =====
        self.manual_params_widget = QWidget()
        manual_layout = QGridLayout(self.manual_params_widget)
        manual_layout.setContentsMargins(10, 10, 10, 10)
        
        manual_info = QLabel("Apply the same coefficients to all selected columns:")
        manual_info.setWordWrap(True)
        manual_info.setStyleSheet("QLabel { color: #666; font-style: italic; }")
        manual_layout.addWidget(manual_info, 0, 0, 1, 2)
        
        manual_layout.addWidget(QLabel("Slope (m):"), 1, 0)
        self.manual_gain_spin = QDoubleSpinBox()
        self.manual_gain_spin.setRange(-100, 100)
        self.manual_gain_spin.setValue(1.0)
        self.manual_gain_spin.setDecimals(6)
        self.manual_gain_spin.setSingleStep(0.01)
        self.manual_gain_spin.setMinimumWidth(150)
        manual_layout.addWidget(self.manual_gain_spin, 1, 1)
        
        manual_layout.addWidget(QLabel("Offset (b):"), 2, 0)
        self.manual_offset_spin = QDoubleSpinBox()
        self.manual_offset_spin.setRange(-10000, 10000)
        self.manual_offset_spin.setValue(0.0)
        self.manual_offset_spin.setDecimals(6)
        self.manual_offset_spin.setMinimumWidth(150)
        manual_layout.addWidget(self.manual_offset_spin, 2, 1)
        
        formula_label = QLabel("Formula: y_calibrated = m × y_raw + b")
        formula_label.setStyleSheet("QLabel { font-weight: bold; color: #01579b; margin-top: 10px; }")
        manual_layout.addWidget(formula_label, 3, 0, 1, 2)
        
        manual_layout.setRowStretch(4, 1)
        
        # ===== Temperature Compensation Parameters =====
        self.temp_params_widget = QWidget()
        temp_layout = QGridLayout(self.temp_params_widget)
        temp_layout.setContentsMargins(10, 10, 10, 10)
        
        temp_info = QLabel("Compensates each selected column based on temperature deviation from reference:")
        temp_info.setWordWrap(True)
        temp_info.setStyleSheet("QLabel { color: #666; font-style: italic; }")
        temp_layout.addWidget(temp_info, 0, 0, 1, 2)
        
        temp_layout.addWidget(QLabel("Temperature Column:"), 1, 0)
        self.temp_column_combo = QComboBox()
        self.temp_column_combo.setMinimumWidth(200)
        temp_layout.addWidget(self.temp_column_combo, 1, 1)
        
        temp_layout.addWidget(QLabel("Temperature Coefficient (%/°C):"), 2, 0)
        self.temp_coeff_spin = QDoubleSpinBox()
        self.temp_coeff_spin.setRange(-10, 10)
        self.temp_coeff_spin.setValue(0.1)
        self.temp_coeff_spin.setDecimals(4)
        self.temp_coeff_spin.setSingleStep(0.01)
        self.temp_coeff_spin.setMinimumWidth(150)
        temp_layout.addWidget(self.temp_coeff_spin, 2, 1)
        
        temp_layout.addWidget(QLabel("Reference Temperature (°C):"), 3, 0)
        self.ref_temp_spin = QDoubleSpinBox()
        self.ref_temp_spin.setRange(-50, 50)
        self.ref_temp_spin.setValue(25.0)
        self.ref_temp_spin.setDecimals(2)
        self.ref_temp_spin.setMinimumWidth(150)
        temp_layout.addWidget(self.ref_temp_spin, 3, 1)
        
        self.btn_auto_calc_temp = QPushButton("Use Mean Temperature as Reference")
        self.btn_auto_calc_temp.clicked.connect(lambda: self.auto_calculate_params('temperature'))
        self.btn_auto_calc_temp.setStyleSheet("QPushButton { background-color: #9C27B0; color: white; }")
        temp_layout.addWidget(self.btn_auto_calc_temp, 4, 0, 1, 2)
        
        temp_formula = QLabel("Formula: y_cal = y_raw × [1 + coeff × (T - T_ref)]")
        temp_formula.setStyleSheet("QLabel { font-weight: bold; color: #01579b; margin-top: 10px; }")
        temp_layout.addWidget(temp_formula, 5, 0, 1, 2)
        
        temp_layout.setRowStretch(6, 1)
        
        # ===== Humidity Correction Parameters =====
        self.humid_params_widget = QWidget()
        humid_layout = QGridLayout(self.humid_params_widget)
        humid_layout.setContentsMargins(10, 10, 10, 10)
        
        humid_info = QLabel("Applies correction factor to all selected columns:")
        humid_info.setWordWrap(True)
        humid_info.setStyleSheet("QLabel { color: #666; font-style: italic; }")
        humid_layout.addWidget(humid_info, 0, 0, 1, 2)
        
        humid_layout.addWidget(QLabel("Humidity Column:"), 1, 0)
        self.humid_column_combo = QComboBox()
        self.humid_column_combo.setMinimumWidth(200)
        humid_layout.addWidget(self.humid_column_combo, 1, 1)
        
        humid_layout.addWidget(QLabel("Correction Factor:"), 2, 0)
        self.humid_factor_spin = QDoubleSpinBox()
        self.humid_factor_spin.setRange(0, 2)
        self.humid_factor_spin.setValue(1.0)
        self.humid_factor_spin.setDecimals(4)
        self.humid_factor_spin.setSingleStep(0.01)
        self.humid_factor_spin.setMinimumWidth(150)
        humid_layout.addWidget(self.humid_factor_spin, 2, 1)
        
        self.btn_auto_calc_humid = QPushButton("Get Humidity Statistics")
        self.btn_auto_calc_humid.clicked.connect(lambda: self.auto_calculate_params('humidity'))
        self.btn_auto_calc_humid.setStyleSheet("QPushButton { background-color: #9C27B0; color: white; }")
        humid_layout.addWidget(self.btn_auto_calc_humid, 3, 0, 1, 2)
        
        humid_formula = QLabel("Formula: y_calibrated = y_raw × correction_factor")
        humid_formula.setStyleSheet("QLabel { font-weight: bold; color: #01579b; margin-top: 10px; }")
        humid_layout.addWidget(humid_formula, 4, 0, 1, 2)
        
        humid_layout.setRowStretch(5, 1)
        
        # ===== Pressure/Altitude Correction Parameters =====
        self.pressure_params_widget = QWidget()
        pressure_layout = QGridLayout(self.pressure_params_widget)
        pressure_layout.setContentsMargins(10, 10, 10, 10)
        
        pressure_info = QLabel("Converts station pressure to sea level for all selected pressure columns:")
        pressure_info.setWordWrap(True)
        pressure_info.setStyleSheet("QLabel { color: #666; font-style: italic; }")
        pressure_layout.addWidget(pressure_info, 0, 0, 1, 2)
        
        pressure_layout.addWidget(QLabel("Station Altitude (m):"), 1, 0)
        self.altitude_spin = QDoubleSpinBox()
        self.altitude_spin.setRange(-500, 9000)
        self.altitude_spin.setValue(0.0)
        self.altitude_spin.setDecimals(1)
        self.altitude_spin.setMinimumWidth(150)
        pressure_layout.addWidget(self.altitude_spin, 1, 1)
        
        self.sea_level_check = QCheckBox("Convert to sea level pressure")
        self.sea_level_check.setChecked(True)
        pressure_layout.addWidget(self.sea_level_check, 2, 0, 1, 2)
        
        self.btn_auto_calc_pressure = QPushButton("Estimate Altitude from Pressure Data")
        self.btn_auto_calc_pressure.clicked.connect(lambda: self.auto_calculate_params('pressure'))
        self.btn_auto_calc_pressure.setStyleSheet("QPushButton { background-color: #9C27B0; color: white; }")
        pressure_layout.addWidget(self.btn_auto_calc_pressure, 3, 0, 1, 2)
        
        pressure_formula = QLabel("Formula: P_sea = P_station × (1 - 0.0065h/288.15)^(-5.255)")
        pressure_formula.setStyleSheet("QLabel { font-weight: bold; color: #01579b; margin-top: 10px; }")
        pressure_layout.addWidget(pressure_formula, 4, 0, 1, 2)
        
        pressure_layout.setRowStretch(5, 1)
        
        # ===== Wind Speed Height Correction Parameters =====
        self.wind_params_widget = QWidget()
        wind_layout = QGridLayout(self.wind_params_widget)
        wind_layout.setContentsMargins(10, 10, 10, 10)
        
        wind_info = QLabel("Adjusts wind speed from measurement height to target height:")
        wind_info.setWordWrap(True)
        wind_info.setStyleSheet("QLabel { color: #666; font-style: italic; }")
        wind_layout.addWidget(wind_info, 0, 0, 1, 2)
        
        wind_layout.addWidget(QLabel("Measurement Height (m):"), 1, 0)
        self.wind_height_spin = QDoubleSpinBox()
        self.wind_height_spin.setRange(0.1, 100)
        self.wind_height_spin.setValue(10.0)
        self.wind_height_spin.setDecimals(2)
        self.wind_height_spin.setMinimumWidth(150)
        wind_layout.addWidget(self.wind_height_spin, 1, 1)
        
        wind_layout.addWidget(QLabel("Target Height (m):"), 2, 0)
        self.target_height_spin = QDoubleSpinBox()
        self.target_height_spin.setRange(0.1, 100)
        self.target_height_spin.setValue(10.0)
        self.target_height_spin.setDecimals(2)
        self.target_height_spin.setMinimumWidth(150)
        wind_layout.addWidget(self.target_height_spin, 2, 1)
        
        wind_layout.addWidget(QLabel("Surface Roughness:"), 3, 0)
        self.roughness_combo = QComboBox()
        self.roughness_combo.addItems([
            'Open Water (z0 = 0.0002 m)',
            'Smooth Ground (z0 = 0.005 m)',
            'Grass (z0 = 0.03 m)',
            'Crops (z0 = 0.1 m)',
            'Forest (z0 = 0.5 m)'
        ])
        self.roughness_combo.setCurrentIndex(2)  # Default to grass
        self.roughness_combo.setMinimumWidth(250)
        wind_layout.addWidget(self.roughness_combo, 3, 1)
        
        wind_formula = QLabel("Formula: v2 = v1 × (h2/h1)^α")
        wind_formula.setStyleSheet("QLabel { font-weight: bold; color: #01579b; margin-top: 10px; }")
        wind_layout.addWidget(wind_formula, 4, 0, 1, 2)
        
        wind_layout.setRowStretch(5, 1)
        
        # Store all parameter widgets
        self.param_widgets = {
            'Linear Regression (y = mx + b)': self.linear_params_widget,
            'Manual Coefficients (y = mx + b)': self.manual_params_widget,
            'Temperature Compensation': self.temp_params_widget,
            'Humidity Correction': self.humid_params_widget,
            'Pressure/Altitude Correction': self.pressure_params_widget,
            'Wind Speed Height Correction': self.wind_params_widget
        }

    def on_target_selection_changed(self):
        """Enable/disable calculate button based on selection"""
        has_selection = len(self.calib_target_list.selectedItems()) > 0
        self.btn_calculate_calib.setEnabled(has_selection and self.processed_data is not None)

    def on_calibration_method_changed(self, method_text):
        """Handle calibration method selection change"""
        # Clear existing parameter widget
        while self.params_layout.count():
            child = self.params_layout.takeAt(0)
            if child.widget():
                child.widget().setParent(None)
        
        # Update description
        descriptions = {
            'Linear Regression (y = mx + b)': 
                "📊 Calculates INDIVIDUAL slope (m) and offset (b) for EACH selected column by comparing to the reference column. "
                "Each column will have its own unique calibration coefficients.",
            
            'Manual Coefficients (y = mx + b)': 
                "✏️ Applies the SAME manually entered slope and offset to ALL selected columns. "
                "All columns will be calibrated using identical coefficients.",
            
            'Temperature Compensation': 
                "🌡️ Applies temperature compensation to EACH selected column based on the temperature column. "
                "Corrects for temperature-dependent sensor drift.",
            
            'Humidity Correction': 
                "💧 Applies the SAME correction factor to ALL selected columns based on humidity. "
                "Useful for sensors affected by ambient moisture.",
            
            'Pressure/Altitude Correction': 
                "⛰️ Converts station pressure to sea level pressure for ALL selected columns. "
                "Each column will use the same altitude correction.",
            
            'Wind Speed Height Correction': 
                "🌬️ Adjusts wind speed for ALL selected columns from measurement height to target height. "
                "All columns will use the same height correction factor."
        }
        
        self.method_description.setText(descriptions.get(method_text, "Select a calibration method"))
        
        # Show appropriate parameter widget
        if method_text in self.param_widgets:
            self.params_layout.addWidget(self.param_widgets[method_text])
            self.param_widgets[method_text].setVisible(True)
        
        # Update reference column visibility
        needs_reference = method_text == 'Linear Regression (y = mx + b)'
        self.ref_column_combo.setEnabled(needs_reference)
        self.ref_column_label.setVisible(needs_reference)
        
        # Clear previous results
        self.calib_table.setRowCount(0)
        self.calib_summary_label.setText("Click 'Calculate Calibration Coefficients' to see results")
        self.btn_apply_calculated.setEnabled(False)

    def calculate_calibration_coefficients(self):
        """Calculate calibration coefficients for each selected column"""
        if self.processed_data is None:
            QMessageBox.warning(self, "Warning", "No data loaded")
            return
        
        selected_items = self.calib_target_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select columns to calibrate")
            return
        
        target_columns = [item.text() for item in selected_items]
        method = self.calib_method_combo.currentText()
        
        self.calculated_calibration = {}
        
        try:
            if method == 'Linear Regression (y = mx + b)':
                self.calculate_linear_regression(target_columns)
            elif method == 'Manual Coefficients (y = mx + b)':
                self.calculate_manual_coefficients(target_columns)
            elif method == 'Temperature Compensation':
                self.calculate_temperature_compensation(target_columns)
            elif method == 'Humidity Correction':
                self.calculate_humidity_correction(target_columns)
            elif method == 'Pressure/Altitude Correction':
                self.calculate_pressure_correction(target_columns)
            elif method == 'Wind Speed Height Correction':
                self.calculate_wind_correction(target_columns)
            
            # Display results in table
            self.display_calibration_table()
            
            # Enable apply button
            self.btn_apply_calculated.setEnabled(True)
            
            # Update summary
            self.calib_summary_label.setText(
                f"✓ Calibration calculated for {len(self.calculated_calibration)} column(s). "
                f"Review the table above and click 'Apply Calculated Calibration' to modify data."
            )
            self.calib_summary_label.setStyleSheet("QLabel { color: green; font-weight: bold; padding: 5px; }")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Calibration calculation failed: {str(e)}")
            import traceback
            traceback.print_exc()

    def calculate_linear_regression(self, target_columns):
        """Calculate linear regression for each column"""
        try:
            from scipy import stats
        except ImportError:
            raise ImportError("scipy is required for linear regression. Install with: pip install scipy")
        
        ref_column = self.ref_column_combo.currentText()
        force_zero = self.force_zero_check.isChecked()
        
        ref_data = self.processed_data[ref_column].dropna()
        
        for col in target_columns:
            col_data = self.processed_data[col].dropna()
            common_idx = ref_data.index.intersection(col_data.index)
            
            if len(common_idx) < 2:
                self.calculated_calibration[col] = {
                    'method': 'Linear Regression',
                    'slope': None,
                    'offset': None,
                    'r_squared': None,
                    'status': 'Insufficient data',
                    'formula': 'N/A'
                }
                continue
            
            # Sensor (target) is the predictor; reference is the dependent variable
            x = col_data.loc[common_idx].values   # target/sensor
            y = ref_data.loc[common_idx].values   # reference

            
            if force_zero:
                # Force through origin: y = m·x
                den = np.sum(x * x)
                if den == 0:
                    # no variance in x — cannot fit
                    self.calculated_calibration[col] = {
                        'method': 'Linear Regression',
                        'slope': None, 'offset': None, 'r_squared': None,
                        'status': 'Insufficient variance', 'formula': 'N/A'
                    }
                    continue
                slope = np.sum(x * y) / den
                offset = 0.0
                y_pred = slope * x
                sst = np.sum((y - np.mean(y))**2)
                r_squared = 1.0 - (np.sum((y - y_pred)**2) / sst if sst > 0 else 0.0)
            else:
                slope, offset, r_value, p_value, std_err = stats.linregress(x, y)
                r_squared = r_value ** 2
            
            self.calculated_calibration[col] = {
                'method': 'Linear Regression',
                'slope': slope,
                'offset': offset,
                'r_squared': r_squared,
                'n_points': len(common_idx),
                'status': 'Ready',
                'formula': f'y = {slope:.6f}x + {offset:.6f}'
            }

    def calculate_manual_coefficients(self, target_columns):
        """Apply manual coefficients to all columns"""
        slope = self.manual_gain_spin.value()
        offset = self.manual_offset_spin.value()
        
        for col in target_columns:
            self.calculated_calibration[col] = {
                'method': 'Manual',
                'slope': slope,
                'offset': offset,
                'status': 'Ready',
                'formula': f'y = {slope:.6f}x + {offset:.6f}'
            }

    def calculate_temperature_compensation(self, target_columns):
        """Calculate temperature compensation for each column"""
        temp_col = self.temp_column_combo.currentText()
        temp_coeff = self.temp_coeff_spin.value() / 100.0  # Convert to decimal
        ref_temp = self.ref_temp_spin.value()
        
        if not temp_col or temp_col not in self.processed_data.columns:
            raise ValueError("Temperature column not found")
        
        for col in target_columns:
            self.calculated_calibration[col] = {
                'method': 'Temperature Compensation',
                'temp_column': temp_col,
                'temp_coeff': temp_coeff,
                'ref_temp': ref_temp,
                'status': 'Ready',
                'formula': f'y = y_raw × [1 + {temp_coeff:.4f} × (T - {ref_temp})]'
            }

    def calculate_humidity_correction(self, target_columns):
        """Calculate humidity correction for each column"""
        humid_col = self.humid_column_combo.currentText()
        factor = self.humid_factor_spin.value()
        
        for col in target_columns:
            self.calculated_calibration[col] = {
                'method': 'Humidity Correction',
                'humid_column': humid_col,
                'factor': factor,
                'status': 'Ready',
                'formula': f'y = y_raw × {factor:.4f}'
            }

    def calculate_pressure_correction(self, target_columns):
        """Calculate pressure correction for each column"""
        altitude = self.altitude_spin.value()
        to_sea_level = self.sea_level_check.isChecked()
        
        # Standard atmosphere: P_sea = P_station * (1 - 0.0065*h/288.15)^(-5.255)
        correction_factor = (1 - 0.0065 * altitude / 288.15) ** (-5.255) if to_sea_level else 1.0
        
        for col in target_columns:
            self.calculated_calibration[col] = {
                'method': 'Pressure Correction',
                'altitude': altitude,
                'correction_factor': correction_factor,
                'status': 'Ready',
                'formula': f'y = y_raw × {correction_factor:.6f}'
            }

    def calculate_wind_correction(self, target_columns):
        """Calculate wind speed height correction for each column"""
        h1 = self.wind_height_spin.value()
        h2 = self.target_height_spin.value()
        
        # Get roughness length from combo
        roughness_text = self.roughness_combo.currentText()
        z0_values = {
            'Open Water (z0 = 0.0002 m)': 0.0002,
            'Smooth Ground (z0 = 0.005 m)': 0.005,
            'Grass (z0 = 0.03 m)': 0.03,
            'Crops (z0 = 0.1 m)': 0.1,
            'Forest (z0 = 0.5 m)': 0.5
        }
        z0 = z0_values.get(roughness_text, 0.03)
        
        # Power law exponent based on roughness
        alpha = 1.0 / 7.0  # Standard 1/7 power law for neutral conditions
        
        correction_factor = (h2 / h1) ** alpha
        
        for col in target_columns:
            self.calculated_calibration[col] = {
                'method': 'Wind Height Correction',
                'height_from': h1,
                'height_to': h2,
                'roughness': roughness_text,
                'correction_factor': correction_factor,
                'status': 'Ready',
                'formula': f'y = y_raw × {correction_factor:.6f}'
            }

    def display_calibration_table(self):
        """Display calibration results in table"""
        if not self.calculated_calibration:
            return
        
        # Configure table
        self.calib_table.setRowCount(len(self.calculated_calibration))
        
        # Determine columns based on calibration method
        first_entry = next(iter(self.calculated_calibration.values()))
        method = first_entry['method']
        
        if method == 'Linear Regression':
            headers = ['Column Name', 'Slope (m)', 'Offset (b)', 'R²', 'Data Points', 'Formula', 'Status']
            self.calib_table.setColumnCount(len(headers))
            self.calib_table.setHorizontalHeaderLabels(headers)
            
            for row, (col_name, calib) in enumerate(self.calculated_calibration.items()):
                self.calib_table.setItem(row, 0, QTableWidgetItem(col_name))
                self.calib_table.setItem(row, 1, QTableWidgetItem(f"{calib['slope']:.6f}" if calib['slope'] is not None else "N/A"))
                self.calib_table.setItem(row, 2, QTableWidgetItem(f"{calib['offset']:.6f}" if calib['offset'] else "N/A"))
                self.calib_table.setItem(row, 3, QTableWidgetItem(f"{calib['r_squared']:.4f}" if calib['r_squared'] else "N/A"))
                self.calib_table.setItem(row, 4, QTableWidgetItem(str(calib.get('n_points', 'N/A'))))
                self.calib_table.setItem(row, 5, QTableWidgetItem(calib['formula']))
                
                status_item = QTableWidgetItem(calib['status'])
                if calib['status'] == 'Ready':
                    status_item.setBackground(Qt.green)
                else:
                    status_item.setBackground(Qt.yellow)
                self.calib_table.setItem(row, 6, status_item)
        
        elif method in ['Manual', 'Humidity Correction', 'Pressure Correction', 'Wind Height Correction']:
            headers = ['Column Name', 'Correction Factor', 'Formula', 'Status']
            self.calib_table.setColumnCount(len(headers))
            self.calib_table.setHorizontalHeaderLabels(headers)
            
            for row, (col_name, calib) in enumerate(self.calculated_calibration.items()):
                self.calib_table.setItem(row, 0, QTableWidgetItem(col_name))
                
                if 'slope' in calib:
                    factor_text = f"m={calib['slope']:.6f}, b={calib['offset']:.6f}"
                elif 'correction_factor' in calib:
                    factor_text = f"{calib['correction_factor']:.6f}"
                elif 'factor' in calib:
                    factor_text = f"{calib['factor']:.6f}"
                else:
                    factor_text = "See formula"
                
                self.calib_table.setItem(row, 1, QTableWidgetItem(factor_text))
                self.calib_table.setItem(row, 2, QTableWidgetItem(calib['formula']))
                
                status_item = QTableWidgetItem(calib['status'])
                status_item.setBackground(Qt.green)
                self.calib_table.setItem(row, 3, status_item)
        
        elif method == 'Temperature Compensation':
            headers = ['Column Name', 'Temp Column', 'Coefficient (%/°C)', 'Ref Temp (°C)', 'Formula', 'Status']
            self.calib_table.setColumnCount(len(headers))
            self.calib_table.setHorizontalHeaderLabels(headers)
            
            for row, (col_name, calib) in enumerate(self.calculated_calibration.items()):
                self.calib_table.setItem(row, 0, QTableWidgetItem(col_name))
                self.calib_table.setItem(row, 1, QTableWidgetItem(calib['temp_column']))
                self.calib_table.setItem(row, 2, QTableWidgetItem(f"{calib['temp_coeff']*100:.4f}"))
                self.calib_table.setItem(row, 3, QTableWidgetItem(f"{calib['ref_temp']:.2f}"))
                self.calib_table.setItem(row, 4, QTableWidgetItem(calib['formula']))
                
                status_item = QTableWidgetItem(calib['status'])
                status_item.setBackground(Qt.green)
                self.calib_table.setItem(row, 5, status_item)
        
        # Resize columns to content
        self.calib_table.resizeColumnsToContents()

    def apply_calculated_calibration(self):
        """Apply the calculated calibration to the data"""
        if not self.calculated_calibration:
            QMessageBox.warning(self, "Warning", "No calibration calculated. Click 'Calculate Calibration Coefficients' first.")
            return
        
        # Count how many columns are ready to calibrate
        ready_columns = [col for col, calib in self.calculated_calibration.items() 
                         if calib['status'] == 'Ready']
        
        if not ready_columns:
            QMessageBox.warning(self, "Warning", "No columns ready for calibration")
            return
        
        # Confirm with user
        reply = QMessageBox.question(
            self, "Confirm Calibration",
            f"Apply calibration to {len(ready_columns)} column(s)?\n\n"
            f"Columns: {', '.join(ready_columns[:5])}"
            f"{' ...' if len(ready_columns) > 5 else ''}\n\n"
            "This will create new columns with '(CALd)' suffix containing calibrated values.",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        try:
            applied_count = 0
            
            for col_name, calib in self.calculated_calibration.items():
                if calib['status'] != 'Ready' or col_name not in self.processed_data.columns:
                    continue
                
                method = calib['method']
                
                if method in ['Linear Regression', 'Manual']:
                    # Apply y = mx + b
                    slope = calib['slope']
                    offset = calib['offset']
                    new_col = f"{col_name}(CALd)"
                    self.processed_data[new_col] = self.processed_data[col_name] * slope + offset
                    applied_count += 1
                
                elif method == 'Temperature Compensation':
                    # Apply temperature compensation
                    temp_col = calib['temp_column']
                    temp_coeff = calib['temp_coeff']
                    ref_temp = calib['ref_temp']
                    
                    if temp_col in self.processed_data.columns:
                        temp_diff = self.processed_data[temp_col] - ref_temp
                        correction = 1 + (temp_coeff * temp_diff)
                        new_col = f"{col_name}(CALd)"
                        self.processed_data[new_col] = self.processed_data[col_name] * correction
                        applied_count += 1
                
                elif method in ['Humidity Correction', 'Pressure Correction', 'Wind Height Correction']:
                    # Apply simple correction factor
                    factor = calib.get('correction_factor') or calib.get('factor')
                    if factor:
                        new_col = f"{col_name}(CALd)"
                        self.processed_data[new_col] = self.processed_data[col_name] * factor
                        applied_count += 1
            
            # Update data table
            self.update_data_table()
            
            # Update plot columns to include new calibrated columns
            self.update_plot_columns()
            
            # Store in calibration coefficients for export
            for col_name, calib in self.calculated_calibration.items():
                if calib['status'] == 'Ready':
                    self.calibration_coefficients[col_name] = calib
            
            # Enable export
            self.btn_export_calib.setEnabled(True)
            
            # Update status
            self.update_status(f"Calibration applied to {applied_count} columns")
            
            QMessageBox.information(
                self, "Success",
                f"Calibration successfully applied to {applied_count} column(s)!\n\n"
                "You can now:\n"
                "• Plot the calibrated columns in the Visualization tab\n"
                "• Export the data with calibrated values\n"
                "• Export calibration coefficients for future use\n"
                "• View the modified data in the Data View tab"
            )
            
            # Clear calculated calibration to prevent re-application
            self.calculated_calibration = {}
            self.calib_table.setRowCount(0)
            self.btn_apply_calculated.setEnabled(False)
            self.calib_summary_label.setText("Calibration applied successfully. Calculate new calibration or load a file to calibrate more columns.")
            self.calib_summary_label.setStyleSheet("QLabel { color: blue; font-style: italic; padding: 5px; }")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply calibration: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def create_statistics_tab(self):
        """Create the statistics tab"""
        stats_widget = QWidget()
        layout = QVBoxLayout(stats_widget)
        
        # Statistics text area
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setFont(QFont("Courier", 10))
        layout.addWidget(self.stats_text)
        
        # Calculate stats button
        self.btn_calc_stats = QPushButton("Calculate Statistics")
        self.btn_calc_stats.clicked.connect(self.calculate_statistics)
        self.btn_calc_stats.setEnabled(False)
        layout.addWidget(self.btn_calc_stats)
        
        self.tabs.addTab(stats_widget, "Statistics")
        
    def add_files(self):
        """Add CSV files to the processing list"""
        files, _ = QFileDialog.getOpenFileNames(self, "Select CSV Files", "", "CSV Files (*.csv)")
        for file in files:
            if file not in self.file_list:
                self.file_list.append(file)
                self.file_list_widget.addItem(os.path.basename(file))
        
        if len(files) > 0:
            self.update_status(f"Added {len(files)} files")
            self.btn_scan_dates.setEnabled(True)
            # Automatically scan for dates
            self.scan_file_dates()
        
    def clear_files(self):
        """Clear the file list"""
        self.file_list.clear()
        self.file_list_widget.clear()
        self.btn_scan_dates.setEnabled(False)
        self.detected_date_range_label.setText("Import files to see date range")
        self.update_status("File list cleared")
    
    def scan_file_dates(self):
        """Scan imported files to detect date range"""
        if not self.file_list:
            return
        
        self.update_status("Scanning files for date range...")
        QApplication.processEvents()  # Update UI
        
        min_date = None
        max_date = None
        datetime_col = self.datetime_col_input.text().strip() or None
        
        try:
            for file_path in self.file_list:
                try:
                    # Read just a sample to find datetime column if needed
                    df = pd.read_csv(file_path, nrows=100)
                    
                    # Auto-detect datetime column if not specified
                    if datetime_col is None:
                        for col in df.columns:
                            try:
                                sample = df[col].dropna().head(10)
                                if len(sample) > 0:
                                    with warnings.catch_warnings():
                                        warnings.filterwarnings('ignore', category=UserWarning)
                                        test_parse = pd.to_datetime(sample, errors='coerce', format='mixed')
                                        if test_parse.notna().sum() > 5:
                                            datetime_col = col
                                            break
                            except:
                                continue
                    
                    if datetime_col and datetime_col in df.columns:
                        # Parse the entire datetime column from the file
                        full_df = pd.read_csv(file_path)
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', category=UserWarning)
                            dates = pd.to_datetime(full_df[datetime_col], errors='coerce', format='mixed')
                        
                        # Remove NaT values
                        dates = dates.dropna()
                        
                        if len(dates) > 0:
                            file_min = dates.min()
                            file_max = dates.max()
                            
                            if pd.notna(file_min) and pd.notna(file_max):
                                if min_date is None or file_min < min_date:
                                    min_date = file_min
                                if max_date is None or file_max > max_date:
                                    max_date = file_max
                except Exception as e:
                    self.update_status(f"Warning: Could not scan {os.path.basename(file_path)}: {str(e)}")
                    continue
            
            if min_date and max_date and pd.notna(min_date) and pd.notna(max_date):
                # Update the display label
                self.detected_date_range_label.setText(
                    f"Start: {min_date.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"End: {max_date.strftime('%Y-%m-%d %H:%M:%S')}"
                )
                
                # Update the datetime editors in the processing tab
                # Convert pandas Timestamp to Python datetime first
                start_py = min_date.to_pydatetime()
                end_py = max_date.to_pydatetime()
                
                # Create QDateTime objects with proper components
                start_qdt = QDateTime(start_py.year, start_py.month, start_py.day,
                                     start_py.hour, start_py.minute, start_py.second)
                end_qdt = QDateTime(end_py.year, end_py.month, end_py.day,
                                   end_py.hour, end_py.minute, end_py.second)
                
                self.start_datetime.setDateTime(start_qdt)
                self.end_datetime.setDateTime(end_qdt)
                
                self.update_status(f"Date range detected: {min_date.date()} to {max_date.date()}")
            else:
                self.detected_date_range_label.setText("Could not detect date range")
                self.update_status("Warning: Could not detect date range in files")
                
        except Exception as e:
            self.detected_date_range_label.setText(f"Error scanning dates")
            self.update_status(f"Error scanning dates: {str(e)}")
            QMessageBox.warning(self, "Scan Error", f"Could not scan date range: {str(e)}")
    
    def toggle_timeframe_controls(self, checked):
        """Enable/disable timeframe controls"""
        self.start_datetime.setEnabled(checked)
        self.end_datetime.setEnabled(checked)
        
    def process_data(self):
        """Process the selected files"""
        if not self.file_list:
            QMessageBox.warning(self, "Warning", "Please select files to process")
            return
            
        # Disable buttons during processing
        self.btn_process.setEnabled(False)
        self.btn_export_csv.setEnabled(False)
        self.btn_export_excel.setEnabled(False)
        if self.btn_rename_column:
            self.btn_rename_column.setEnabled(False)
            self.btn_remove_columns.setEnabled(False)
            self.btn_reorder_columns.setEnabled(False)
        
        # Set up processor thread
        self.processor_thread.files = self.file_list
        self.processor_thread.resample_period = self.resample_combo.currentText()
        self.processor_thread.agg_method = self.agg_combo.currentText()
        self.processor_thread.remove_outliers = self.outlier_check.isChecked()
        self.processor_thread.outlier_std = self.outlier_spin.value()
        self.processor_thread.interpolate_missing = self.interpolate_check.isChecked()
        self.processor_thread.interpolate_method = self.interpolate_combo.currentText()
        
        # Set datetime parsing options
        datetime_col = self.datetime_col_input.text().strip()
        self.processor_thread.datetime_column = datetime_col if datetime_col else None
        self.processor_thread.datetime_format = self.datetime_format_combo.currentText()
        self.processor_thread.custom_format = self.custom_format_input.text().strip() if self.custom_format_input.isEnabled() else None
        
        # Set timeframe options
        self.processor_thread.apply_timeframe = self.apply_timeframe_check.isChecked()
        if self.processor_thread.apply_timeframe:
            self.processor_thread.start_time = pd.Timestamp(self.start_datetime.dateTime().toPyDateTime())
            self.processor_thread.end_time = pd.Timestamp(self.end_datetime.dateTime().toPyDateTime())
        else:
            self.processor_thread.start_time = None
            self.processor_thread.end_time = None
        
        # Start processing
        self.processor_thread.start()
        
    def processing_finished(self, df):
        """Handle processing completion"""
        self.processed_data = df
        self.btn_process.setEnabled(True)
        
        # Update data table
        self.update_data_table()
        
        # Enable export and column operation buttons
        self.btn_export_csv.setEnabled(True)
        self.btn_export_excel.setEnabled(True)
        self.btn_rename_column.setEnabled(True)
        self.btn_remove_columns.setEnabled(True)
        self.btn_reorder_columns.setEnabled(True)
        
        # Update plot column list
        self.update_plot_columns()
        
        # Update calibration controls
        self.update_calibration_controls()
        
        # Enable statistics button
        self.btn_calc_stats.setEnabled(True)
        
        # Switch to data view tab
        self.tabs.setCurrentIndex(2)
        
        # Check for data quality issues
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            total_values = len(df) * len(numeric_cols)
            nan_values = df[numeric_cols].isna().sum().sum()
            nan_percentage = (nan_values / total_values * 100) if total_values > 0 else 0
            
            msg = f"Data processing completed!\n\n"
            msg += f"Records: {len(df)}\n"
            msg += f"Numeric columns: {len(numeric_cols)}\n"
            msg += f"Missing values: {nan_percentage:.1f}%"
            
            if nan_percentage > 30:
                msg += f"\n\nNote: High percentage of missing values detected."
                msg += f"\nConsider using a larger resample period or check your data quality."
                QMessageBox.warning(self, "Processing Complete", msg)
            else:
                QMessageBox.information(self, "Success", msg)
        else:
            QMessageBox.information(self, "Success", "Data processing completed successfully!")
        
    def update_data_table(self):
        """Update the data table with processed data"""
        if self.processed_data is None:
            return
            
        # Set table dimensions
        self.data_table.setRowCount(len(self.processed_data))
        self.data_table.setColumnCount(len(self.processed_data.columns))
        
        # Set headers
        self.data_table.setHorizontalHeaderLabels(self.processed_data.columns.tolist())
        
        # Populate table (show first 1000 rows for performance)
        display_rows = min(1000, len(self.processed_data))
        for row in range(display_rows):
            for col in range(len(self.processed_data.columns)):
                value = self.processed_data.iloc[row, col]
                if pd.isna(value):
                    item = QTableWidgetItem("NaN")
                elif isinstance(value, (int, float)):
                    item = QTableWidgetItem(f"{value:.4f}")
                else:
                    item = QTableWidgetItem(str(value))
                self.data_table.setItem(row, col, item)
                
        if display_rows < len(self.processed_data):
            self.data_info_label.setText(f"Showing first {display_rows} of {len(self.processed_data)} rows | "
                                        f"{len(self.processed_data.columns)} columns")
        else:
            self.data_info_label.setText(f"Showing all {len(self.processed_data)} rows | "
                                        f"{len(self.processed_data.columns)} columns")
            
    def show_column_context_menu(self, position):
        """Show context menu for column operations"""
        if self.processed_data is None:
            return
            
        # Get the column index at the click position
        column = self.data_table.columnAt(position.x())
        if column < 0:
            return
            
        menu = QMenu(self)
        column_name = self.processed_data.columns[column]
        
        # Add menu actions
        rename_action = menu.addAction(f"Rename '{column_name}'")
        remove_action = menu.addAction(f"Remove '{column_name}'")
        menu.addSeparator()
        stats_action = menu.addAction(f"Show statistics for '{column_name}'")
        
        # Execute menu and handle selection
        action = menu.exec_(self.data_table.mapToGlobal(position))
        
        if action == rename_action:
            self.rename_single_column(column)
        elif action == remove_action:
            self.remove_single_column(column)
        elif action == stats_action:
            self.show_column_statistics(column)
            
    def rename_single_column(self, column_idx):
        """Rename a single column"""
        old_name = self.processed_data.columns[column_idx]
        new_name, ok = QInputDialog.getText(self, "Rename Column", 
                                           f"Enter new name for '{old_name}':",
                                           text=old_name)
        if ok and new_name and new_name != old_name:
            if new_name in self.processed_data.columns:
                QMessageBox.warning(self, "Warning", 
                                  f"Column name '{new_name}' already exists!")
                return
            
            # Rename in dataframe
            columns = list(self.processed_data.columns)
            columns[column_idx] = new_name
            self.processed_data.columns = columns
            
            # Update table
            self.update_data_table()
            self.update_plot_columns()
            self.update_status(f"Renamed column '{old_name}' to '{new_name}'")
            
    def remove_single_column(self, column_idx):
        """Remove a single column"""
        column_name = self.processed_data.columns[column_idx]
        
        # Check if this is the last column
        if len(self.processed_data.columns) == 1:
            QMessageBox.warning(self, "Warning", 
                              "Cannot remove the last remaining column!")
            return
        
        # Confirm deletion
        reply = QMessageBox.question(self, "Confirm Deletion",
                                    f"Are you sure you want to remove column '{column_name}'?",
                                    QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.processed_data = self.processed_data.drop(columns=[column_name])
            self.update_data_table()
            self.update_plot_columns()
            self.update_status(f"Removed column '{column_name}'")
            
    def show_column_statistics(self, column_idx):
        """Show statistics for a specific column"""
        column_name = self.processed_data.columns[column_idx]
        column_data = self.processed_data[column_name]
        
        # Check if numeric
        if not pd.api.types.is_numeric_dtype(column_data):
            stats_text = f"Column: {column_name}\nType: {column_data.dtype}\nUnique values: {column_data.nunique()}"
            if column_data.nunique() < 20:
                stats_text += f"\nValues: {', '.join(map(str, column_data.unique()[:20]))}"
        else:
            stats_text = f"""Column: {column_name}
Type: {column_data.dtype}
Count: {column_data.count()}
Missing: {column_data.isna().sum()}
Mean: {column_data.mean():.4f}
Std: {column_data.std():.4f}
Min: {column_data.min():.4f}
25%: {column_data.quantile(0.25):.4f}
Median: {column_data.median():.4f}
75%: {column_data.quantile(0.75):.4f}
Max: {column_data.max():.4f}"""
        
        QMessageBox.information(self, f"Statistics for '{column_name}'", stats_text)
            
    def rename_column(self):
        """Open dialog to rename columns"""
        if self.processed_data is None:
            return
            
        dialog = ColumnRenameDialog(self.processed_data.columns.tolist(), self)
        if dialog.exec_():
            new_names = dialog.get_new_names()
            if new_names != list(self.processed_data.columns):
                self.processed_data.columns = new_names
                self.update_data_table()
                self.update_plot_columns()
                self.update_status("Columns renamed successfully")
                
    def remove_columns(self):
        """Open dialog to remove multiple columns"""
        if self.processed_data is None:
            return
            
        dialog = ColumnRemoveDialog(self.processed_data.columns.tolist(), self)
        if dialog.exec_():
            columns_to_remove = dialog.get_columns_to_remove()
            if columns_to_remove:
                # Check if we're removing all columns
                if len(columns_to_remove) == len(self.processed_data.columns):
                    QMessageBox.warning(self, "Warning", 
                                      "Cannot remove all columns!")
                    return
                    
                self.processed_data = self.processed_data.drop(columns=columns_to_remove)
                self.update_data_table()
                self.update_plot_columns()
                self.update_status(f"Removed {len(columns_to_remove)} columns")
                
    def reorder_columns(self):
        """Open dialog to reorder columns"""
        if self.processed_data is None:
            return
            
        dialog = ColumnReorderDialog(self.processed_data.columns.tolist(), self)
        if dialog.exec_():
            new_order = dialog.get_column_order()
            self.processed_data = self.processed_data[new_order]
            self.update_data_table()
            self.update_plot_columns()
            self.update_status("Columns reordered successfully")
            
    def update_plot_columns(self):
        """Update the plot column selection list"""
        if self.processed_data is None:
            return
            
        self.plot_list.clear()
        numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            self.plot_list.addItem(col)
            
        self.btn_plot.setEnabled(True)
        self.btn_save_plot.setEnabled(True)
        
    def on_plot_type_changed(self, plot_type):
        """Handle plot type change"""
        if plot_type == 'Dual Y-Axis':
            self.dual_axis_group.setVisible(True)
            self.update_dual_axis_lists()
        else:
            self.dual_axis_group.setVisible(False)
            
    def update_dual_axis_lists(self):
        """Update the dual axis assignment lists"""
        if self.processed_data is None:
            return
            
        selected_items = self.plot_list.selectedItems()
        selected_columns = [item.text() for item in selected_items]
        
        self.left_axis_list.clear()
        self.right_axis_list.clear()
        
        # Default: first half to left, second half to right
        mid_point = len(selected_columns) // 2
        for i, col in enumerate(selected_columns):
            if i < mid_point or (i == mid_point and len(selected_columns) % 2 == 1):
                self.left_axis_list.addItem(col)
            else:
                self.right_axis_list.addItem(col)
                
    def update_plot(self):
        """Update the visualization plot"""
        if self.processed_data is None:
            return
            
        selected_items = self.plot_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select columns to plot")
            return
            
        columns = [item.text() for item in selected_items]
        
        # Find datetime column
        datetime_col = None
        for col in self.processed_data.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                datetime_col = col
                break
                
        if datetime_col is None:
            datetime_col = self.processed_data.columns[0]
            
        # Get plot type
        plot_type_map = {
            'Separate Subplots': 'subplots',
            'Single Plot': 'single',
            'Dual Y-Axis': 'dual'
        }
        plot_type = plot_type_map[self.plot_type_combo.currentText()]
        
        # Get axis assignments for dual plot
        left_columns = None
        right_columns = None
        if plot_type == 'dual':
            left_columns = [self.left_axis_list.item(i).text() 
                          for i in range(self.left_axis_list.count())]
            right_columns = [self.right_axis_list.item(i).text() 
                           for i in range(self.right_axis_list.count())]
            
        # Update plot
        self.plot_canvas.plot_data(
            self.processed_data, columns, datetime_col,
            plot_type=plot_type,
            normalize=self.normalize_check.isChecked(),
            show_grid=self.show_grid_check.isChecked(),
            show_legend=self.show_legend_check.isChecked(),
            rolling_avg=self.rolling_avg_check.isChecked(),
            rolling_window=self.rolling_window_spin.value(),
            left_columns=left_columns,
            right_columns=right_columns
        )
        
    def save_plot(self):
        """Save the current plot to file"""
        filename, _ = QFileDialog.getSaveFileName(self, "Save Plot", "", 
                                                 "PNG Files (*.png);;PDF Files (*.pdf)")
        if filename:
            self.plot_canvas.fig.savefig(filename, dpi=150, bbox_inches='tight')
            self.update_status(f"Plot saved to {filename}")

    def update_calibration_controls(self):
        """Update calibration controls when data is loaded"""
        if self.processed_data is None:
            return
        
        # Update column combo boxes
        numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns.tolist()
        
        self.ref_column_combo.clear()
        self.ref_column_combo.addItems(numeric_cols)
        
        self.temp_column_combo.clear()
        self.temp_column_combo.addItems(numeric_cols)
        
        self.humid_column_combo.clear()
        self.humid_column_combo.addItems(numeric_cols)
        
        # Update target columns list
        self.calib_target_list.clear()
        for col in numeric_cols:
            self.calib_target_list.addItem(col)
        
        # Enable apply loaded calibration if we have calibration coefficients
        if self.calibration_coefficients:
            self.btn_apply_loaded_calib.setEnabled(True)
            # Check which columns match
            matching_cols = [col for col in self.calibration_coefficients.keys() 
                           if col in numeric_cols]
            if matching_cols:
                self.calib_status_label.setText(
                    f"Calibration ready: {len(matching_cols)}/{len(self.calibration_coefficients)} "
                    f"columns match loaded data"
                )
                self.calib_status_label.setStyleSheet("QLabel { color: green; font-weight: bold; }")
            else:
                self.calib_status_label.setText(
                    f"Warning: No columns match between calibration and data"
                )
                self.calib_status_label.setStyleSheet("QLabel { color: orange; font-weight: bold; }")

    def auto_calculate_params(self, method):
        """Auto-calculate parameters for the selected calibration method"""
        if self.processed_data is None:
            QMessageBox.warning(self, "Warning", "No data loaded. Process data first.")
            return
        
        try:
            if method == 'temperature':
                # Calculate mean temperature as reference
                temp_col = self.temp_column_combo.currentText()
                if temp_col and temp_col in self.processed_data.columns:
                    mean_temp = self.processed_data[temp_col].mean()
                    self.ref_temp_spin.setValue(mean_temp)
                    
                    msg = f"Reference temperature set to mean value: {mean_temp:.2f}°C\n\n"
                    msg += "Typical temperature coefficients:\n"
                    msg += "• Electronic sensors: 0.1-0.3%/°C\n"
                    msg += "• Resistive sensors: 0.3-0.5%/°C\n"
                    msg += "• Pressure sensors: 0.01-0.05%/°C\n\n"
                    msg += "Adjust the coefficient based on your sensor specifications."
                    QMessageBox.information(self, "Auto-Calculate Results", msg)
                else:
                    QMessageBox.warning(self, "Warning", "No temperature column selected or found")
                    
            elif method == 'humidity':
                # Provide guidance on humidity correction
                humid_col = self.humid_column_combo.currentText()
                if humid_col and humid_col in self.processed_data.columns:
                    mean_humid = self.processed_data[humid_col].mean()
                    msg = f"Current mean humidity: {mean_humid:.1f}%\n\n"
                    msg += "Typical humidity correction factors:\n"
                    msg += "• Capacitive sensors: 1.0 (minimal correction)\n"
                    msg += "• Resistive sensors: 0.95-1.05\n\n"
                    msg += "Set correction factor based on calibration data."
                    QMessageBox.information(self, "Auto-Calculate Info", msg)
                else:
                    QMessageBox.warning(self, "Warning", "No humidity column selected")
                    
            elif method == 'pressure':
                # Estimate altitude from pressure if available
                # Standard atmosphere: P = P0 * (1 - 0.0065*h/288.15)^5.255
                # Solving for h: h = 288.15/0.0065 * (1 - (P/P0)^(1/5.255))
                pressure_cols = [col for col in self.processed_data.columns 
                               if 'pressure' in col.lower() or 'press' in col.lower()]
                
                if pressure_cols:
                    # Use first pressure column
                    mean_pressure = self.processed_data[pressure_cols[0]].mean()
                    # Assume sea level pressure = 1013.25 hPa
                    P0 = 1013.25
                    if 50 < mean_pressure < 1100:  # Valid pressure range in hPa
                        estimated_alt = 288.15 / 0.0065 * (1 - (mean_pressure / P0) ** (1/5.255))
                        self.altitude_spin.setValue(estimated_alt)
                        
                        msg = f"Estimated altitude from pressure data:\n\n"
                        msg += f"Mean pressure: {mean_pressure:.2f} hPa\n"
                        msg += f"Estimated altitude: {estimated_alt:.0f} m\n\n"
                        msg += "Note: This assumes standard atmospheric conditions.\n"
                        msg += "Verify with your actual station altitude."
                        QMessageBox.information(self, "Auto-Calculate Results", msg)
                    else:
                        QMessageBox.warning(self, "Warning", 
                                          f"Pressure value seems invalid: {mean_pressure:.2f}")
                else:
                    QMessageBox.information(self, "Info", 
                                          "No pressure columns found.\nManually enter station altitude.")
                    
            else:
                QMessageBox.information(self, "Info", 
                                      f"Auto-calculate not yet implemented for {method} method")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Auto-calculate failed: {str(e)}")
            
    def calculate_statistics(self):
        """Calculate and display statistics for the processed data"""
        if self.processed_data is None:
            return
            
        stats_text = "=" * 50 + "\n"
        stats_text += "METEOROLOGICAL DATA STATISTICS\n"
        stats_text += "=" * 50 + "\n\n"
        
        # Dataset info
        stats_text += f"Total Records: {len(self.processed_data)}\n"
        stats_text += f"Columns: {len(self.processed_data.columns)}\n"
        
        # Date range
        datetime_col = None
        for col in self.processed_data.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                datetime_col = col
                break
                
        if datetime_col:
            stats_text += f"\nDate Range:\n"
            stats_text += f"  Start: {self.processed_data[datetime_col].min()}\n"
            stats_text += f"  End: {self.processed_data[datetime_col].max()}\n"
            
        stats_text += "\n" + "-" * 50 + "\n"
        
        # Statistics for each numeric column
        numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            stats_text += f"\n{col}:\n"
            stats_text += f"  Count:     {self.processed_data[col].count()}\n"
            stats_text += f"  Mean:      {self.processed_data[col].mean():.4f}\n"
            stats_text += f"  Median:    {self.processed_data[col].median():.4f}\n"
            stats_text += f"  Std Dev:   {self.processed_data[col].std():.4f}\n"
            stats_text += f"  Min:       {self.processed_data[col].min():.4f}\n"
            stats_text += f"  Max:       {self.processed_data[col].max():.4f}\n"
            stats_text += f"  25%:       {self.processed_data[col].quantile(0.25):.4f}\n"
            stats_text += f"  75%:       {self.processed_data[col].quantile(0.75):.4f}\n"
            stats_text += f"  Missing:   {self.processed_data[col].isna().sum()}\n"
            
        self.stats_text.setText(stats_text)
        
    def export_csv(self):
        """Export processed data to CSV"""
        if self.processed_data is None:
            return
            
        filename, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv)")
        if filename:
            self.processed_data.to_csv(filename, index=False)
            QMessageBox.information(self, "Success", f"Data exported to {filename}")
            
    def export_excel(self):
        """Export processed data to Excel"""
        if self.processed_data is None:
            return
            
        filename, _ = QFileDialog.getSaveFileName(self, "Save Excel", "", "Excel Files (*.xlsx)")
        if filename:
            try:
                self.processed_data.to_excel(filename, index=False, engine='openpyxl')
                QMessageBox.information(self, "Success", f"Data exported to {filename}")
            except ImportError:
                QMessageBox.warning(self, "Warning", 
                                   "Please install openpyxl: pip install openpyxl")

    def export_calibration(self):
        """Export calibration coefficients to JSON file"""
        if not self.calibration_coefficients:
            QMessageBox.warning(self, "Warning", "No calibration coefficients to export")
            return
        
        filename, _ = QFileDialog.getSaveFileName(self, "Save Calibration", "", 
                                                 "JSON Files (*.json)")
        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(self.calibration_coefficients, f, indent=2)
                QMessageBox.information(self, "Success", 
                                      f"Calibration coefficients exported to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")
    
    def import_calibration_file(self):
        """Import calibration coefficients from JSON file"""
        filename, _ = QFileDialog.getOpenFileName(self, "Import Calibration", "",
                                                 "JSON Files (*.json)")
        if filename:
            try:
                with open(filename, 'r') as f:
                    imported_coeffs = json.load(f)
                
                # Validate the structure
                if not isinstance(imported_coeffs, dict):
                    raise ValueError("Invalid calibration file format")
                
                # Merge with existing calibrations
                for col_name, coeffs in imported_coeffs.items():
                    if not isinstance(coeffs, dict):
                        raise ValueError(f"Invalid coefficients for column {col_name}")
                    self.calibration_coefficients[col_name] = coeffs
                
                # Update status
                num_columns = len(imported_coeffs)
                self.calib_status_label.setText(
                    f"Calibration loaded: {num_columns} column(s) from {os.path.basename(filename)}"
                )
                self.calib_status_label.setStyleSheet("QLabel { color: green; font-weight: bold; }")
                
                # Enable buttons
                self.btn_view_edit_calib.setEnabled(True)
                self.btn_clear_calib.setEnabled(True)
                self.btn_export_calib.setEnabled(True)
                if self.processed_data is not None:
                    self.btn_apply_loaded_calib.setEnabled(True)
                
                QMessageBox.information(self, "Success", 
                                      f"Imported calibration for {num_columns} column(s)")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", 
                                   f"Failed to import calibration: {str(e)}")
    
    def view_edit_calibration(self):
        """Open dialog to view and edit calibration coefficients"""
        if not self.calibration_coefficients:
            QMessageBox.information(self, "No Calibration", 
                                  "No calibration coefficients to edit. Import a file or calculate calibration first.")
            return
        
        dialog = CalibrationEditorDialog(self.calibration_coefficients, self)
        if dialog.exec_():
            self.calibration_coefficients = dialog.get_calibration_coefficients()
            
            # Update status
            num_columns = len(self.calibration_coefficients)
            self.calib_status_label.setText(
                f"Calibration active: {num_columns} column(s)"
            )
            self.calib_status_label.setStyleSheet("QLabel { color: green; font-weight: bold; }")
            
            # Enable export button
            self.btn_export_calib.setEnabled(True)
            
            self.update_status(f"Calibration updated for {num_columns} column(s)")
    
    def clear_calibration(self):
        """Clear all calibration coefficients"""
        if not self.calibration_coefficients:
            return
        
        reply = QMessageBox.question(self, "Clear Calibration",
                                    "Clear all calibration coefficients?",
                                    QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.calibration_coefficients.clear()
            self.calib_status_label.setText("No calibration loaded")
            self.calib_status_label.setStyleSheet("QLabel { color: #666; font-style: italic; }")
            self.btn_view_edit_calib.setEnabled(False)
            self.btn_clear_calib.setEnabled(False)
            self.btn_export_calib.setEnabled(False)
            self.btn_apply_loaded_calib.setEnabled(False)
            self.update_status("Calibration cleared")
    
    def apply_loaded_calibration(self):
        """Apply the loaded/imported calibration coefficients to the data"""
        if self.processed_data is None:
            QMessageBox.warning(self, "Warning", "No data loaded")
            return
        
        if not self.calibration_coefficients:
            QMessageBox.warning(self, "Warning", "No calibration loaded. Import a file or calculate calibration first.")
            return
        
        # Find columns that match
        numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns.tolist()
        matching_cols = [col for col in self.calibration_coefficients.keys() 
                        if col in numeric_cols]
        
        if not matching_cols:
            QMessageBox.warning(self, "Warning", 
                              "No columns in the data match the calibration file")
            return
        
        # Show what will be calibrated
        reply = QMessageBox.question(self, "Apply Calibration",
                                    f"Apply calibration to {len(matching_cols)} column(s)?\n\n"
                                    f"Columns: {', '.join(matching_cols[:5])}"
                                    f"{' ...' if len(matching_cols) > 5 else ''}\n\n"
                                    "This will create new columns with '(CALd)' suffix containing calibrated values.",
                                    QMessageBox.Yes | QMessageBox.No)
        
        if reply != QMessageBox.Yes:
            return
        
        try:
            applied_count = 0
            
            for col in matching_cols:
                if col not in self.processed_data.columns:
                    continue
                
                coeffs = self.calibration_coefficients[col]
                
                # Apply different calibration types based on available coefficients
                if 'slope' in coeffs and 'offset' in coeffs:
                    # Linear calibration: y = slope*x + offset
                    slope = coeffs['slope']
                    offset = coeffs['offset']
                    new_col = f"{col}(CALd)"
                    self.processed_data[new_col] = self.processed_data[col] * slope + offset
                    applied_count += 1
                    
                elif 'gain' in coeffs and 'offset' in coeffs:
                    # Manual calibration: y = gain*x + offset
                    gain = coeffs['gain']
                    offset = coeffs['offset']
                    new_col = f"{col}(CALd)"
                    self.processed_data[new_col] = self.processed_data[col] * gain + offset
                    applied_count += 1
                    
                elif 'temp_coeff' in coeffs:
                    # Temperature compensation - would need temperature column
                    pass
                    
                else:
                    pass
            
            # Update displays
            self.update_data_table()
            
            # Update plot columns to include new calibrated columns
            self.update_plot_columns()
            
            self.update_status(f"Applied calibration to {applied_count} columns")
            
            # Make sure export is enabled
            self.btn_export_calib.setEnabled(True)
            
            QMessageBox.information(self, "Success", 
                                  f"Calibration applied to {applied_count} column(s)!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply calibration: {str(e)}")
                
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
        
    def update_status(self, message):
        """Update status label"""
        self.status_label.setText(message)
        
    def show_error(self, message):
        """Show error message"""
        self.btn_process.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Processing error: {message}")


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set application-wide font
    font = QFont()
    font.setPointSize(9)
    app.setFont(font)
    
    window = MeteoDataProcessor()
    window.showMaximized()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()