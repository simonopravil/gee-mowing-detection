"""
Utility functions for grassland mowing event detection using satellite time-series data.

This module provides functions for:
- Data loading and preprocessing (Sentinel-1 SAR and Sentinel-2 optical)
- Spatial cross-validation (proper train/test splitting)
- Feature engineering (with temporal awareness to prevent data leakage)
- Model training and evaluation
- Visualization

Author: Simon Opravil
Project: GMI Grassland Mowing Detection
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, max_error, make_scorer, f1_score)
from scipy.stats import pearsonr, linregress
from typing import Tuple, List, Dict, Optional, Union
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import optuna

def preprocess_points(points, group_col="auto"):
    """Preprocess input points DataFrame."""
    points = points.copy()
    points['date'] = pd.to_datetime(points['date'])
    if 'endDate' in points.columns:
        points['endDate'] = pd.to_datetime(points['endDate'])
    
    if group_col == "auto":
        points['id'] = points['PLOTID'].str[:2]
        group_col = 'id'
    
    return points, group_col

def load_and_concat_csv(paths, cols):
    """Load and concatenate CSV files with consistent PLOTID handling."""
    dfs = []
    for path in paths:
        df = pd.read_csv(path)

        # --- Handle plotID / PLOTID column mismatch ---
        if 'PLOTID' not in df.columns:
            if 'plotID' in df.columns:
                df = df.rename(columns={'plotID': 'PLOTID'})
            else:
                raise ValueError(f"No PLOTID or plotID column found in file: {path}")

        # --- Subset to requested columns (after rename) ---
        missing_cols = [c for c in cols if c not in df.columns]
        if missing_cols:
            print(f"⚠️ Missing columns in {path}: {missing_cols}")
        df = df[[c for c in cols if c in df.columns]]
        df = df.loc[:, ~df.columns.duplicated()]
        # --- Ensure Date is datetime ---
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        else:
            raise ValueError(f"Missing 'Date' column in file: {path}")

        dfs.append(df)

    if not dfs:
        raise ValueError("No CSV files loaded — check your file paths.")

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.sort_values(['PLOTID', 'Date']).reset_index(drop=True)
    return merged

def interpolate_s2(s2_df, s1_dates_df, bands):
    """Interpolate S2 data to S1 dates."""
    interp_list = []
    for pid, group in s1_dates_df.groupby('PLOTID'):
        s2_sub = s2_df[s2_df['PLOTID'] == pid]
        if s2_sub.empty:
            continue
        s2_sub = s2_sub.set_index('Date').sort_index()
        new_df = pd.DataFrame(index=group['Date'].unique())
        for band in bands:
            try:
                new_df[band] = np.interp(
                    new_df.index.astype(np.int64),
                    s2_sub.index.astype(np.int64),
                    s2_sub[band].values
                )
            except Exception:
                new_df[band] = np.nan
        new_df['Date'] = new_df.index
        new_df['PLOTID'] = pid
        interp_list.append(new_df.reset_index(drop=True))
    return pd.concat(interp_list, ignore_index=True) if interp_list else pd.DataFrame()

def apply_savgol_filter(df, bands, savgol_params):
    """Apply Savitzky-Golay filter to specified bands."""
    smooth = []
    for pid, group in df.groupby('PLOTID'):
        group = group.sort_values('Date').copy()
        for band in bands:
            vals = group[band].values
            if len(vals) >= savgol_params[0]:
                try:
                    group[band] = savgol_filter(vals, *savgol_params)
                except Exception:
                    group[band] = vals
            else:
                group[band] = vals
        smooth.append(group)
    return pd.concat(smooth, ignore_index=True)

def normalize_data(df, feature_cols, method='zscore'):
    """Normalize data by plotID and year."""
    df = df.copy()
    df['year'] = df['Date'].dt.year
    normalized = []
    
    for (pid, year), group in df.groupby(['PLOTID', 'year']):
        g = group.copy()
        for col in feature_cols:
            vals = g[col].values.astype(float)
            if method == 'zscore':
                mean, std = np.nanmean(vals), np.nanstd(vals)
                g[col] = (vals - mean) / std if std > 0 else 0.0
            elif method == 'minmax':
                min_, max_ = np.nanmin(vals), np.nanmax(vals)
                g[col] = (vals - min_) / (max_ - min_) if max_ > min_ else 0.0
        normalized.append(g)
    
    return pd.concat(normalized, ignore_index=True)

def label_management_events(merged, points):
    """Label mowing and grazing events."""
    merged['mowing'] = 0
    merged['grazing'] = 0
    
    for _, row in points.iterrows():
        pid, mtype, start = row['PLOTID'], row['type'], row['date']
        end = row.get('endDate', pd.NaT)
        mask = merged['PLOTID'] == pid
        
        if mtype == 'p':  # Grazing
            if pd.notna(end):
                date_mask = (merged['Date'] >= start) & (merged['Date'] <= end)
            else:
                date_mask = (merged['Date'] >= start) & (merged['Date'].dt.year == start.year)
            merged.loc[mask & date_mask, 'grazing'] = 1
        elif mtype == 'k':  # Mowing
            obs = merged[mask & (merged['Date'] >= start) & (merged['Date'].dt.year == start.year)].sort_values('Date')
            if not obs.empty:
                next_date = obs.iloc[0]['Date'].normalize()  # Normalize date
                merged.loc[mask & (merged['Date'].dt.normalize() == next_date), 'mowing'] = 1
    
    return merged

def prepare_sequences(points, use_s1 = True, s2_paths = None, s1_paths = None, coh_paths = None, s1_cols = None, s2_cols = None, coh_cols = None,
                          group_col="auto", apply_savgol=False, savgol_params=(5, 2), Normalize = False
                          ):
    """
    Prepare management sequences from S1 and S2 data.

    Args:
        points: DataFrame with plot information
        s1_paths, s2_paths: Lists of paths to S1/S2 CSV files
        s1_cols, s2_cols: Lists of columns to use from S1/S2 data
        group_col: Column for grouping (default: 'auto' for plotID prefix)
        allowed_orbit: Filter S1 data by specific orbit (optional)
        savgol_params: Parameters for Savitzky-Golay filter
        apply_savgol: Whether to apply Savitzky-Golay filter
        apply_normalization: Whether to apply normalization
    
    Returns:
        Processed DataFrame with labeled management events
    """
    
    # Load and concatenate data

    s2_df = load_and_concat_csv(s2_paths, s2_cols)
    s2_df = s2_df[s2_df[s2_cols[0]]>-9999]
    s2_df['bsi'] = (
        (s2_df["swir1"] + s2_df["red"]) - (s2_df["nir"] - s2_df["blue"])
    ) / (
        (s2_df["swir1"] + s2_df["red"]) + (s2_df["nir"] + s2_df["blue"])
    )
    
    if use_s1:
        s1_df = load_and_concat_csv(s1_paths, s1_cols)
        s1_df = s1_df[s1_df[s1_cols[0]]>-9999]

        coh_df = load_and_concat_csv(coh_paths, coh_cols)
        coh_df = coh_df[coh_df[coh_cols[0]]>-9999]
        # Create S1 date grid
        s1_grid = s1_df[['PLOTID', 'Date']].drop_duplicates()
        
        # Interpolate S2 data to S1 dates
        s2_bands = [col for col in s2_cols if col not in ['Date', 'PLOTID']]
        s2_interp = interpolate_s2(s2_df, s1_grid, s2_bands)
        
        # Merge S1 and interpolated S2 data
        s1_use = s1_df[['PLOTID', 'Date'] + [c for c in s1_cols if c not in ['Date', 'PLOTID']]]
        s1_s2 = pd.merge(s1_use, s2_interp, on=['PLOTID', 'Date'], how='inner')

        coh_use = coh_df[['PLOTID', 'Date'] + [c for c in coh_cols if c not in ['Date', 'PLOTID']]]
        merged = pd.merge(s1_s2, coh_use, on=['PLOTID', 'Date'], how='inner')

    else:
        merged = s2_df
    
    feature_cols = [c for c in merged.columns if c not in ['PLOTID', 'Date', 'year']]
    
    # Apply optional Savitzky-Golay filter
    if apply_savgol:
        merged = apply_savgol_filter(merged, feature_cols, savgol_params)
    
    if Normalize:
        merged = normalize_data(merged, feature_cols, method='minmax')
    
    if points is not None:
    # Label management events
        merged = label_management_events(merged, points)
    
    merged['id'] = merged['PLOTID'].str[:2]
    merged['year'] = merged['Date'].dt.year
    
    return merged.sort_values(['PLOTID', 'year', 'Date'])


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
def add_temporal_features(df: pd.DataFrame,
                          columns: List[str],
                          group_cols: List[str],
                          date_col: str = 'Date',
                          past_steps: int = 1,
                          compute_differences: bool = True) -> pd.DataFrame:
    """
    Add temporal features using ONLY PAST observations (no future information).

    CRITICAL: This function only looks backward in time to prevent data leakage.
    For predicting mowing at time t, we only use observations from t-3, t-2, t-1.

    Parameters
    ----------
    df : pd.DataFrame
        Input time-series data
    columns : List[str]
        Feature columns to create temporal windows for
    group_cols : List[str]
        Grouping columns (e.g., ['PLOTID', 'year'])
    date_col : str
        Name of date column
    past_steps : int
        Number of past time steps to include (e.g., 3 means t-3, t-2, t-1)
    compute_differences : bool
        Whether to compute differences between current and past values

    Returns
    -------
    pd.DataFrame
        Data with added temporal features
    """
    df = df.copy()
    grouped = df.groupby(group_cols, group_keys=False)

    new_cols = {}

    # Add past observations
    for step in range(1, past_steps + 1):
        for col in columns:
            new_cols[f'{col}_lag_{step}'] = grouped[col].shift(step)

    # Add differences (current - past)
    if compute_differences:
        for step in range(1, past_steps + 1):
            for col in columns:
                lag_col = f'{col}_lag_{step}'
                if lag_col in new_cols:
                    new_cols[f'{col}_diff_{step}'] = df[col] - new_cols[lag_col]

    # Concatenate all new columns at once (efficient, no fragmentation)
    result = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return result

# ============================================================================
# SPATIAL CROSS-VALIDATION
# ============================================================================

def cv_leave_one_out_split(df: pd.DataFrame, split_col: str = 'id'):
    """
    Create leave-one-region-out train/test splits.

    For datasets with few spatial regions (e.g., 3 regions), this function
    systematically holds out each region as test set while using all other
    regions for training.

    CRITICAL: This ensures proper spatial cross-validation where each region
    serves as test set exactly once, and NO region appears in both train and test.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with spatial grouping column
    id_col : str
        Name of spatial grouping column (e.g., 'id')

    Yields
    ------
    Tuple[str, pd.DataFrame, pd.DataFrame]
        Test region ID, training DataFrame, test DataFrame

    Example
    -------
    >>> for test_region, df_train, df_test in create_leave_one_region_out_splits(df, 'id'):
    >>>     print(f"Test region: {test_region}")
    >>>     # Train model on df_train, evaluate on df_test
    """
    unique_regions = sorted(df[split_col].unique())
    n_regions = len(unique_regions)

    print(f"Leave-One-Region-Out Cross-Validation:")
    print(f"  Total regions: {n_regions}")
    print(f"  Regions: {unique_regions}")
    print(f"  Will perform {n_regions} iterations (1 region held out each time)")
    print("=" * 80)

    for test_region in unique_regions:
        train_regions = [r for r in unique_regions if r != test_region]

        df_train = df[df[split_col].isin(train_regions)].reset_index(drop=True)
        df_test = df[df[split_col] == test_region].reset_index(drop=True)

        print(f"\nIteration: Test region = {test_region}")
        print(f"  Train regions: {train_regions}")
        print(f"  Train: {len(df_train)} samples from {len(train_regions)} regions")
        print(f"  Test:  {len(df_test)} samples from 1 region")

        yield test_region, df_train, df_test

# ============================================================================
# MODEL EVALUATION
# ============================================================================
def custom_f1_score(df: pd.DataFrame,
                    preds: np.ndarray,
                    use: str,
                    window_before: int = 3,
                    window_after: int = 20) -> float:
    """
    Calculate F1-score with temporal tolerance window for mowing event detection.

    Unlike standard F1, this metric considers a prediction as True Positive if it falls
    within a temporal window around the actual mowing date (±window_before/after days).
    This accounts for uncertainty in exact mowing dates and satellite revisit times.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: Date, and the target column specified by 'use'
    preds : np.ndarray
        Binary predictions (1 = mowing event, 0 = no mowing)
    use : str
        Column name containing true labels (e.g., 'mowing')
    window_before : int
        Days before reference date to consider for matching (default: 3)
    window_after : int
        Days after reference date to consider for matching (default: 20)

    Returns
    -------
    float
        F1-score accounting for temporal tolerance

    Notes
    -----
    - If both true and predicted events are 0, returns 1.0 (perfect TN)
    - Each true event can match at most one predicted event
    - Each predicted event can match at most one true event
    """
    df = df.copy()
    df['pred'] = preds
    df = df.sort_values('Date')

    true_mow = df[df[use] == 1]
    pred_mow = df[df['pred'] == 1]

    # Handle the all-0 special case (perfect TN)
    if len(true_mow) == 0 and len(pred_mow) == 0:
        return 1.0

    tp = 0
    used_preds = set()

    # Match each true event to closest predicted event within window
    for _, row in true_mow.iterrows():
        ref_date = row['Date']
        window_start = ref_date - pd.Timedelta(days=window_before)
        window_end = ref_date + pd.Timedelta(days=window_after)

        window_preds = pred_mow[
            (pred_mow['Date'] >= window_start) &
            (pred_mow['Date'] <= window_end)
        ]

        # Use first available prediction in window
        for idx in window_preds.index:
            if idx not in used_preds:
                tp += 1
                used_preds.add(idx)
                break

    fp = len(pred_mow) - len(used_preds)
    fn = len(true_mow) - tp

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return f1


def custom_error_matrix(df: pd.DataFrame,
                        preds: np.ndarray,
                        use: str,
                        window_before: int = 3,
                        window_after: int = 20) -> Dict[str, float]:
    """
    Calculate confusion matrix with temporal tolerance window for mowing events.

    Similar to custom_f1_score but returns full confusion matrix and derived metrics.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: Date, and the target column specified by 'use'
    preds : np.ndarray
        Binary predictions (1 = mowing event, 0 = no mowing)
    use : str
        Column name containing true labels (e.g., 'mowing')
    window_before : int
        Days before reference date for matching (default: 3)
    window_after : int
        Days after reference date for matching (default: 20)

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - TP, FP, FN, TN: Confusion matrix values
        - Precision, Recall, F1: Derived metrics
    """
    df = df.copy()
    df['pred'] = preds
    df = df.sort_values('Date')

    true_mow = df[df[use] == 1]
    pred_mow = df[df['pred'] == 1]

    tp = 0
    used_preds = set()
    matched_truths = set()

    # Match true events to predicted events
    for i, row in true_mow.iterrows():
        ref_date = row['Date']
        window_start = ref_date - pd.Timedelta(days=window_before)
        window_end = ref_date + pd.Timedelta(days=window_after)

        window_preds = pred_mow[
            (pred_mow['Date'] >= window_start) &
            (pred_mow['Date'] <= window_end)
        ]

        for idx in window_preds.index:
            if idx not in used_preds:
                tp += 1
                used_preds.add(idx)
                matched_truths.add(i)
                break

    fp = len(pred_mow) - len(used_preds)
    fn = len(true_mow) - len(matched_truths)
    tn = len(df) - (tp + fp + fn)

    return {
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'TN': tn,
        'Precision': tp / (tp + fp) if (tp + fp) else 0,
        'Recall': tp / (tp + fn) if (tp + fn) else 0,
        'F1': 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 0
    }

def calculate_mape_and_offset(predicted_mowings, reference_mowings):
    """
    Calculate MAPE and offset for mowing events.
    
    Parameters:
    predicted_mowings: List or array of predicted number of mowing events per parcel
    reference_mowings: List or array of reference number of mowing events per parcel
    
    Returns:
    mape: Mean Absolute Percentage Error across all parcels
    offsets: List of absolute differences in the number of mowing events per parcel
    """
    # Ensure inputs are numpy arrays
    predicted_mowings = np.array(predicted_mowings)
    reference_mowings = np.array(reference_mowings)
    
    # Initialize arrays to store MAPE and offsets
    mape_values = []
    offsets = []
    
    # Calculate MAPE and offset for each parcel
    for pred_count, ref_count in zip(predicted_mowings, reference_mowings):
        # MAPE calculation
        if ref_count == 0:
            if pred_count == 0:
                mape = 0.0  # Both predicted and reference are 0
            else:
                mape = 100.0  # Falsely predicted mowing
        else:
            absolute_error = abs(pred_count - ref_count)
            mape = (absolute_error / ref_count) * 100.0
        mape_values.append(mape)
        
        # Offset calculation (absolute difference in number of events)
        offset = abs(pred_count - ref_count)
        offsets.append(offset)
    
    # Calculate mean MAPE across all parcels
    mape = np.mean(mape_values)/100
    offset = np.mean(offsets)
    
    return mape, offset


def get_date_stats(df: pd.DataFrame,
                   df_point: pd.DataFrame,
                   pred: np.ndarray,
                   plotID_col: str) -> Dict[str, float]:
    """
    Calculate statistical metrics comparing predicted and actual mowing dates.

    This function evaluates the accuracy of predicted mowing DATES (day-of-year),
    not just binary event detection. Useful for assessing temporal precision.

    Parameters
    ----------
    df : pd.DataFrame
        Mowing data with columns: [plotID_col, 'year', 'doy', 'mowing']
    df_point : pd.DataFrame
        Reference point data with columns: [plotID_col, 'year', 'date', 'type']
    pred : np.ndarray
        Binary predictions (1 = mowing event, 0 = no mowing)
    plotID_col : str
        Name of plot ID column (e.g., 'PLOTID' or 'plotID')

    Returns
    -------
    Dict[str, float]
        Dictionary containing date prediction metrics:
        - RMSE: Root mean squared error (days)
        - MAE: Mean absolute error (days)
        - R²: Coefficient of determination
        - Max Error: Maximum absolute error (days)
        - Mean Deviance: Average signed error (bias)
        - Correlation Coefficient: Pearson correlation
        - Slope: Linear regression slope
        - 50th/95th Percentile AE: Median and 95th percentile absolute errors

    Notes
    -----
    - Returns NaN values if insufficient data for comparison
    - Only considers mowing events (type='k') from df_point
    - Matches events by plot, year, and mowing sequence number
    """
    # Input validation
    required_cols_df = [plotID_col, 'year', 'doy', 'mowing']
    required_cols_point = [plotID_col, 'year', 'date', 'type']

    if not all(col in df.columns for col in required_cols_df):
        raise ValueError(f"df missing required columns: {required_cols_df}")
    if not all(col in df_point.columns for col in required_cols_point):
        raise ValueError(f"df_point missing required columns: {required_cols_point}")
    if len(pred) != len(df):
        raise ValueError("Length of pred must match length of df")

    # Preprocess point data (reference mowing dates)
    df_point = df_point[[plotID_col, 'year', 'date', 'type']].copy()
    df_point = df_point[df_point['type'] == 'k']  # Only mowing events

    if df_point.empty:
        return _return_nan_metrics()

    df_point['doy'] = df_point['date'].dt.dayofyear
    df_point['year'] = df_point['date'].dt.year.astype(int)
    df_point['mowing_event'] = df_point.groupby([plotID_col, 'year']).cumcount()

    # Process predictions
    df = df.copy()
    df['pred'] = pred
    pred_mows = (df[df['pred'] == 1][[plotID_col, 'year', 'doy', 'pred']]
                 .drop_duplicates(subset=[plotID_col, 'year', 'doy']))
    pred_mows['mowing_event'] = pred_mows.groupby([plotID_col, 'year']).cumcount()

    # Process actual mowing events from df
    actual_mows = (df[df['mowing'] == 1][[plotID_col, 'year', 'doy', 'mowing']]
                  .drop_duplicates(subset=[plotID_col, 'year', 'doy']))
    actual_mows['mowing_event'] = actual_mows.groupby([plotID_col, 'year']).cumcount()

    # Merge predicted and actual dates
    comparison = pred_mows.merge(
        actual_mows,
        how='outer',
        on=[plotID_col, 'year', 'mowing_event']
    ).dropna()

    comparison = comparison[(comparison['doy_x'] > 0) & (comparison['doy_y'] > 0)]

    # Check for insufficient data
    if len(comparison) < 2:
        return _return_nan_metrics()

    # Calculate metrics
    y_true = comparison['doy_y']
    y_pred = comparison['doy_x']
    abs_error = np.abs(y_pred - y_true)
    residuals = y_pred - y_true

    # Safe calculation of slope and correlation
    if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
        slope = linregress(y_true, y_pred)[0]
        corr = pearsonr(y_true, y_pred)[0]
    else:
        slope = np.nan
        corr = np.nan

    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R²': r2_score(y_true, y_pred),
        'Max Error': max_error(y_true, y_pred),
        'Mean Deviance': np.mean(residuals),
        'Correlation Coefficient': corr,
        'Slope': slope,
        '50th Percentile AE': np.percentile(abs_error, 50),
        '95th Percentile AE': np.percentile(abs_error, 95)
    }

    return comparison, metrics


def _return_nan_metrics() -> Dict[str, float]:
    """Helper function to return NaN metrics when insufficient data."""
    return {
        'RMSE': np.nan,
        'MAE': np.nan,
        'R²': np.nan,
        'Max Error': np.nan,
        'Mean Deviance': np.nan,
        'Correlation Coefficient': np.nan,
        'Slope': np.nan,
        '50th Percentile AE': np.nan,
        '95th Percentile AE': np.nan
    }


def plot_scatter_doys(df: pd.DataFrame,
                      df_point: pd.DataFrame,
                      pred: np.ndarray,
                      plotID_col: str = 'PLOTID',
                      tolerance_before: int = 3,
                      tolerance_after: int = 20,
                      save_path: str = 'ML_bestModel_DOY_scatter') -> None:
    """
    Create scatter plot comparing predicted vs actual mowing day-of-year (DOY).

    Visualizes temporal accuracy of mowing date predictions. Points are colored by
    mowing event number (1st, 2nd, 3rd cut) and shaped by accuracy (circle=correct,
    x=incorrect within tolerance window).

    Parameters
    ----------
    df : pd.DataFrame
        Mowing data with columns: [plotID_col, 'year', 'doy', 'mowing']
    df_point : pd.DataFrame
        Reference point data with columns: [plotID_col, 'year', 'date', 'type']
    pred : np.ndarray
        Binary predictions (1 = mowing event, 0 = no mowing)
    plotID_col : str
        Name of plot ID column (default: 'PLOTID')
    tolerance_before : int
        Days before predicted DOY for TP classification (default: 3)
    tolerance_after : int
        Days after predicted DOY for TP classification (default: 20)
    save_path : str
        Base filename for saving plot (saves as .pdf and .png)

    Notes
    -----
    - Saves two files: {save_path}.pdf and {save_path}.png
    - 1:1 reference line shows perfect prediction
    - Dashed lines show tolerance window boundaries
    """
    # Preprocess point data
    df_point = df_point[[plotID_col, 'year', 'date', 'type']].copy()
    df_point = df_point[df_point['type'] == 'k']
    df_point['doy'] = df_point['date'].dt.dayofyear
    df_point['year'] = df_point['date'].dt.year.astype(int)
    df_point['mowing_event'] = df_point.groupby([plotID_col, 'year']).cumcount()

    # Process predictions
    df = df.copy()
    df['pred'] = pred
    pred_mows = (df[df['pred'] == 1][[plotID_col, 'year', 'doy', 'pred']]
                 .drop_duplicates(subset=[plotID_col, 'year', 'doy']))
    pred_mows['mowing_event'] = pred_mows.groupby([plotID_col, 'year']).cumcount()

    # Process actual mowing events
    actual_mows = (df[df['mowing'] == 1][[plotID_col, 'year', 'doy', 'mowing']]
                  .drop_duplicates(subset=[plotID_col, 'year', 'doy']))
    actual_mows['mowing_event'] = actual_mows.groupby([plotID_col, 'year']).cumcount()

    # Merge and prepare comparison data
    comparison = pred_mows.merge(
        actual_mows,
        how='outer',
        on=[plotID_col, 'year', 'mowing_event']
    ).dropna()
    comparison['mowing_event'] = comparison['mowing_event'].astype(int)
    df_sorted = comparison.sort_values(by='mowing_event')

    # Classify predictions within tolerance window
    def classify_prediction(row):
        if pd.notna(row['doy_y']) and pd.notna(row['doy_x']):
            return 'TP' if (row['doy_x'] - tolerance_before) <= row['doy_y'] <= (row['doy_x'] + tolerance_after) else 'FP'
        return 'FP'

    df_sorted['prediction'] = df_sorted.apply(classify_prediction, axis=1)

    # Initialize plot
    plt.figure(figsize=(6, 6))
    sns.set_theme(style='white')

    # Plot reference lines
    doy_limits = [df_sorted[['doy_x', 'doy_y']].min().min() - 5,
                 df_sorted[['doy_x', 'doy_y']].max().max() + 5]
    plt.plot(doy_limits, doy_limits, color='black', linestyle='-',
            linewidth=1, label='1:1 line')
    plt.plot(doy_limits, [x + tolerance_before for x in doy_limits],
            color='black', linestyle='--', linewidth=1,
            label=f'+{tolerance_before} days')
    plt.plot(doy_limits, [x - tolerance_after for x in doy_limits],
            color='black', linestyle='--', linewidth=1,
            label=f'-{tolerance_after} days')

    # Plot scatter points
    palette = sns.color_palette("YlGnBu", df_sorted['mowing_event'].nunique())
    for _, row in df_sorted.iterrows():
        plt.scatter(
            row['doy_y'], row['doy_x'],
            color=palette[row['mowing_event']],
            edgecolor=None,
            s=80,
            marker='o' if row['prediction'] == 'TP' else 'X',
            linewidth=1.2
        )

    # Configure plot
    plt.xlabel('Reference DOY', fontsize=12)
    plt.ylabel('Predicted DOY', fontsize=12)
    plt.xlim(110, 310)
    plt.ylim(110, 310)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Create custom legend
    unique_events = sorted(df_sorted['mowing_event'].dropna().unique())
    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                  markerfacecolor=palette[int(cut)], markeredgecolor='black',
                  markersize=8, label=f'{int(cut)+1}. mowing')
        for cut in unique_events
    ]
    handles.extend([
        plt.Line2D([0], [0], marker='x', color='black',
                  label='False', markersize=8),
        plt.Line2D([0], [0], marker='o', color='black',
                  markerfacecolor='black', label='True', markersize=8)
    ])

    plt.tight_layout()
    plt.savefig(f"{save_path}.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Scatter plot saved as: {save_path}.pdf and {save_path}.png")
# ============================================================================
# Evaluation grid
# ============================================================================

def create_feature_combinations() -> pd.DataFrame:
    """
    Create grid of all logical combinations of input feature groups.
    
    Feature groups:
    - optical: Blue, green, red, NIR, red-edge, SWIR bands
    - vi: Vegetation indices (NDVI, diff_max)
    - vi_texture: NDVI texture features
    - radar: Sentinel-1 backscatter (VV, VH, RVI, RND)
    - r_texture: Radar texture features
    - coh: Coherence bands
    
    For S2-only combinations (optical, vi, vi_texture), creates two versions:
    - use_s1=False (S2 features only)
    - use_s1=True (S2 features + S1 temporal grid)
    
    Returns
    -------
    pd.DataFrame
        Each row represents one feature combination experiment
    """
    groups = {
        'optical': ['blue', 'green', 'red', 'nir', 're1', 're2', 're3', 'swir1', 'swir2'],
        'vi': ['ndvi', 'diff_max', 'bsi'],
        'vi_texture': ['ndvi_texture_sd_3', 'ndvi_texture_sd_5'],
        'radar': ['VH', 'VV', 'rvi', 'rnd'],
        'r_texture': [
            'VH_asm', 'VH_corr', 'VH_ent',
            'VV_asm', 'VV_corr', 'VV_ent',
            'rvi_asm', 'rvi_corr', 'rvi_ent'
        ],
        'coh': ['b1', 'b2']
    }
    
    group_names = list(groups.keys())
    s1_groups = {'radar', 'r_texture', 'coh'}
    s2_groups = {'optical', 'vi', 'vi_texture'}
    configs = []
    
    # Generate all combinations
    for r in range(1, len(group_names) + 1):
        for combo in itertools.combinations(group_names, r):
            input_name = "+".join(combo)
            input_vars = list(itertools.chain.from_iterable(groups[g] for g in combo))
            
            # Identify S1 vs S2 variables
            s1_variables = list(itertools.chain.from_iterable(
                groups[g] for g in combo if g in s1_groups
            ))
            
            s2_variables = list(itertools.chain.from_iterable(
                groups[g] for g in combo if g in s2_groups
            ))
            
            # Check if this is S2-only combination (no S1 feature groups)
            has_s1_features = any(g in s1_groups for g in combo)
            has_s2_features = any(g in s2_groups for g in combo)
            
            if has_s2_features and not has_s1_features:
                # S2-only combination: create two versions
                
                # Version 1: S2 features only (no S1 temporal grid)
                configs.append({
                    'input_name': input_name,
                    'input_vars': input_vars.copy(),
                    's1_variables': [],
                    's2_variables': s2_variables.copy(),
                    'use_s1': False,
                    'use_s2': True
                })
                
                # Version 2: S2 features + S1 temporal grid
                configs.append({
                    'input_name': f"{input_name}+s1_grid",
                    'input_vars': input_vars.copy(),
                    's1_variables': [],
                    's2_variables': s2_variables.copy(),
                    'use_s1': True,
                    'use_s2': True
                })
            else:
                # S1-only or mixed S1+S2: create single version
                configs.append({
                    'input_name': input_name,
                    'input_vars': input_vars,
                    's1_variables': s1_variables,
                    's2_variables': s2_variables,
                    'use_s1': len(s1_variables) > 0,
                    'use_s2': len(s2_variables) > 0
                })
    
    return pd.DataFrame(configs).sort_values('input_name').reset_index(drop=True)

def create_optuna_objective(X_train, y_train, groups, cv_folds=2):
    """
    Create Optuna objective function with spatial cross-validation.
    """
    
    def objective(trial):
        # Define hyperparameters
        n_estimators = trial.suggest_int('n_estimators', 200, 600, step=100)
        max_features_option = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        max_leaf_nodes = trial.suggest_categorical('max_leaf_nodes', [None, 50, 100, 200, 500])
        bootstrap = trial.suggest_categorical('bootstrap', [True, False])
        
        if bootstrap:
            max_samples = trial.suggest_float('max_samples', 0.5, 0.8, step=0.1)
        else:
            max_samples = None
        
        params = {
            'n_estimators': n_estimators,
            'max_features': max_features_option,
            'min_samples_leaf': min_samples_leaf,
            'max_leaf_nodes': max_leaf_nodes,
            'bootstrap': bootstrap,
            'n_jobs': -1
        }
        
        if bootstrap and max_samples is not None:
            params['max_samples'] = max_samples
        
        model = RandomForestClassifier(**params)
        
        # CORRECTED: Pass splitter object, not generator
        from sklearn.model_selection import cross_val_score, GroupKFold
        from sklearn.metrics import make_scorer, f1_score
        
        gkf = GroupKFold(n_splits=cv_folds)
        
        # Method 1: Pass list of (train, test) splits
        splits = list(gkf.split(X_train, y_train, groups))
        scores = cross_val_score(
            model, X_train, y_train,
            cv=splits,  # ✅ Pass list of splits
            scoring=make_scorer(f1_score),
            n_jobs=1
        )
        
        return scores.mean()
    
    return objective


def get_temporal_feature_names(base_features: List[str],
                                past_steps: int = 1) -> List[str]:
    """
    Generate list of temporal feature names created by add_temporal_features().

    Parameters
    ----------
    base_features : List[str]
        Original feature names
    past_steps : int
        Number of lagged time steps
    include_differences : bool
        Whether difference features are included

    Returns
    -------
    List[str]
        Complete list of temporal feature names
    """
    temporal_features = []

    for feature in base_features:
        for step in range(1, past_steps + 1):
            temporal_features.append(f'{feature}_diff_{step}')

    return temporal_features
