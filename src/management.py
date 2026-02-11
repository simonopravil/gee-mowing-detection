import pandas as pd
import numpy as np

def add_time_shifts(df, column, shifts=2, group_cols=['plotID', 'year']):
    """
    Adds shifted versions of a column within groups.

    Parameters:
        df (pd.DataFrame): Input DataFrame with a datetime column 'Date'.
        column (str): Name of the column to shift (e.g., 'ndvi').
        shifts (int): Number of time steps to shift before and after.
        group_cols (list): Columns to group by before shifting.
    
    Returns:
        pd.DataFrame: DataFrame with added shifted columns.
    """
    df = df.copy()

    # If shifting dayofyear, make sure the column exists
    if column == 'doy' and 'doy' not in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df['doy'] = df['Date'].dt.dayofyear
        column = 'doy'

    # Sort values within groups
    df = df.sort_values(by=group_cols + ['Date'])

    # Perform shifts (corrected direction labels)
    grouped = df.groupby(group_cols)
    for i in range(1, shifts + 1):
        df[f'{column}_-{i}'] = grouped[column].shift(i)   # Past
        df[f'{column}_+{i}'] = grouped[column].shift(-i)  # Future

    return df

def compute_lessthanprevious_df(df, roll_band='ndvi_roll', sum_band='ndvi'):
    df = df.sort_values('Date')
    df['roll_diff'] = df[roll_band] - df[roll_band].shift(1)
    df['diff'] = df[sum_band] - df[sum_band].shift(1)
    df['doydiff'] = df['doy'] - df['doy'].shift(1)
    df['lsp'] = df['roll_diff'] <= 0.01
    return df

def cumsum_negative_diff_1d(diff, lsp, mowing):
    result = np.full(len(diff), np.nan, dtype=np.float32)
    total = 0.0
    for i in range(len(diff)):
        if lsp[i] and mowing[i] != 1:
            if not np.isnan(diff[i]) and diff[i] < 0:
                total += diff[i]
            result[i] = total
        else:
            total = 0.0
            result[i] = np.nan
    return result

def cumsum_swir_hits_1d(swir, lsp, mowing):
    result = np.full(len(swir), np.nan, dtype=np.float32)
    total = 0
    for i in range(len(swir)):
        if lsp[i] and mowing[i] != 1:
            if swir[i] == 1:
                total += 1
            result[i] = total
        else:
            total = 0
            result[i] = np.nan
    return result

def cumsum_doydiff_1d(doydiff, lsp, mowing):
    result = np.full(len(doydiff), np.nan, dtype=np.float32)
    total = 0.0
    for i in range(len(doydiff)):
        if lsp[i] and mowing[i] != 1:
            if not np.isnan(doydiff[i]):
                total += doydiff[i]
            result[i] = total
        else:
            total = 0.0
            result[i] = np.nan
    return result

def apply_all_cumsums(group):
    group = compute_lessthanprevious_df(group)
    group['cumsum'] = cumsum_negative_diff_1d(group['diff'].values,
                                              group['lsp'].values,
                                              group['mowing_pred'].values)
    group['swirCondition_cumsum'] = cumsum_swir_hits_1d(group['swirCondition'].values,
                                                        group['lsp'].values,
                                                        group['mowing_pred'].values)
    group['duration'] = cumsum_doydiff_1d(group['doydiff'].values,
                                          group['lsp'].values,
                                          group['mowing_pred'].values)
    return group

def compute_mowing(df, diff_thresh, max_thresh, frac_thresh, doy_thresh):
    diff = df['ndvi'] - df['ndvi_-1']
    residuals = df['ndvi'] - df['spatial_max']
    previous_diff = df['ndvi_-1'] - df['ndvi_-2']
    previous = df['ndvi_-1']
    next_ = df['ndvi_+1']
    fraction = next_ * 100 / previous
    doy_condition = df['doy_+1'] - df['doy']
    
    # fraction condition
    frac_condition = ~((fraction >= frac_thresh) & (doy_condition <= doy_thresh))

    # Main condition
    condition = (
        (diff <= diff_thresh) &
        (residuals <= max_thresh) &
        (previous_diff >= diff_thresh) &
        (frac_condition)
    )
    return condition.astype(int)

def remove_close_mowing(group, mowing_col):
    # Get sorted mowing dates (indices and days of year)
    mowings = group.loc[group[mowing_col] == 1].sort_values('doy')
    last_kept_day = -999  # Initialize to a value far outside any realistic DOY

    for idx, row in mowings.iterrows():
        current_day = row['doy']
        # Only keep if it's not too close to the last kept mowing
        if current_day - last_kept_day >= 12:
            last_kept_day = current_day  # Update last kept mowing day
        else:
            group.at[idx, mowing_col] = 0  # Remove this mowing

    return group

def compute_grazing(ds, cumm_thresh, max_thresh):
    """
    Grazing detection.
    Assumes relevant variables exist in ds:
    - ndvi, ndvi_roll, ndvi_roll_-1/2, ndvi_roll_+1/2
    - swir2, cumsum, spatial_max
    - ndvi_-1, ndvi_-2
    """
    current = ds['ndvi_roll']
    next_ = ds['ndvi_roll_+1'] - 0.01
    previous_diff = ds['ndvi_-1'] - ds['ndvi_-2']
    residuals = ds['ndvi'] - ds['spatial_max']
    cumsum = ds['cumsum']
    swir_con = ds['swirCondition_cumsum']

    # Logical condition for grazing
    condition = (
      (cumsum <= cumm_thresh) &
      (residuals <= max_thresh) &
      (previous_diff >= -0.1) &
      ((next_ > current) & (swir_con < 2))
      )

    return condition.astype(int)

def remove_close_grazing(group):
        mowing_days = group.loc[group['mowing_pred'] == 1, 'doy'].values
        if len(mowing_days) == 0:
            return group

        # For each grazing row, check if it's within 15 days of any mowing date
        grazing_idx = group.loc[group['grazing_pred'] == 1].index
        for idx in grazing_idx:
            day = group.at[idx, 'doy']
            if any(abs(day - md) <= 20 for md in mowing_days):
                group.at[idx, 'grazing_pred'] = 0
        return group

# def grazing_interval(df):
#     df['grazing_pred_start'] = (df['doy'] - df['duration']) * df['grazing_pred']
#     df['grazing_pred_end'] = (df['doy'] * df['grazing_pred'])

#     df['grazing_interval'] = 0

#     # Group by plotID and year
#     for (plotID, year), group in df.groupby(['plotID', 'year']):
#         for idx, row in group.iterrows():
#             if row['grazing_pred'] == 1:
#                 end_doy = row['doy']
#                 duration = row['duration']
#                 start_doy = end_doy - duration
#                 start_doy_pred = row['grazing_pred_start']
#                 end_doy_pred = row['grazing_pred_end']

#                 # Select correct sub-dataframe
#                 mask = (df['plotID'] == plotID) & (df['year'] == year) & (df['doy'] >= start_doy) & (df['doy'] <= end_doy)
#                 df.loc[mask, 'grazing_interval'] = 1
#                 df.loc[mask, 'grazing_pred_start'] = start_doy_pred
#                 df.loc[mask, 'grazing_pred_end'] = end_doy_pred
#     return df



def grazing_interval(df, threshold):
    """
    Mark grazing intervals only for the first grazing_pred==1 event in each (plotID,year)
    where the running total of grazing_pred ≤ threshold.
    """
    df2 = df.copy()
    
    
    # Initialize columns
    df2['grazing_interval'] = 0
    df2['grazing_pred_start'] = np.nan
    df2['grazing_pred_end'] = np.nan

    # Iterate over each group
    for (pid, yr), grp in df2.groupby(['plotID', 'year']):
        # Filter the relevant rows from the full df2 (with cumsum)
        group_df = df2.loc[grp.index]

        # Find first event where grazing_pred==1 and cumulative count <= threshold
        valid_events = group_df[(group_df['grazing_pred'] == 1) & (group_df['cumsum'] <= threshold)]

        if valid_events.empty:
            continue

        first_idx = valid_events.index[0]
        end_doy = df2.at[first_idx, 'doy']
        duration = df2.at[first_idx, 'duration']
        start_doy = end_doy - duration

        # Mark interval
        mask_interval = (
            (df2['plotID'] == pid) &
            (df2['year'] == yr) &
            (df2['doy'] >= start_doy) &
            (df2['doy'] <= end_doy)
        )

        df2.loc[mask_interval, 'grazing_interval'] = 1
        df2.loc[mask_interval, 'grazing_pred_start'] = start_doy
        df2.loc[mask_interval, 'grazing_pred_end'] = end_doy

    return df2