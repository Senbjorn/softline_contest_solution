import pandas as pd
import datetime as dt
import numpy as np

def copy_missing_days(full_df):
    """
    Detect dates for which historical data is missing and fill corresponding
    rows with information from the previous year, for time series' continuity.
    
    Args:
        full_df - pd.DataFrame where some dates are missing
    
    Returns:
        full_df - pd.DataFrame where missing rows are filled with information
            from the previous year
    """
    # find dates that are missing in the dataset
    current_day, last_day = full_df.create_date.iloc[0], full_df.create_date.iloc[-1]
    all_dates = []
    while current_day <= last_day:
        all_dates.append(current_day)
        current_day += dt.timedelta(days=1)
    missing_dates = list(set(all_dates) - set(full_df.create_date.unique()))
    
    # obtain dataes that were year ago
    dates_replace = []
    for day in missing_dates:
        dates_replace.append(day - dt.timedelta(days=365))
    
    # put information from a year ago insted of missing rows
    replacement = full_df[full_df.create_date.isin(dates_replace)]
    for i in range(len(missing_dates)):
        replacement.create_date[replacement.create_date == dates_replace[i]] = missing_dates[i]
    
    full_df = full_df.append(replacement, ignore_index=True)
    full_df = full_df.sort_values(by="create_date")
    
    return full_df

def get_target_days(result_order):
    """
    Get list of days for which forecast should be made
    
    Args:
        result_order - pd.DataFrame that specifies days for which forecast
            should be made (target days)
    
    Returns:
        target_days - list of target days
    """
    
    target_days = []
    for start_end in result_order.dates.unique():
        start, end = start_end.split(',')
        start, end = dt.datetime.strptime(start[1:], '%Y-%m-%d'), dt.datetime.strptime(end[:-1], '%Y-%m-%d')
        delta = end - start
        
        for i in range(delta.days + 1):
            day = start + dt.timedelta(days=i)
            target_days.append(day)
    
    return target_days

def add_nans(agg_data, result_order, n_target_values=128):
    """
    Add rows with NaNs for the days for which forecast should be made.
    
    Args:
        agg_data - pd.DataFrame with aggregated historical data
        result_order - pd.DataFrame that specifies days for which forecast
            should be made (target days)
        missing_dates - list of days for which historical information is missing
        n_target_values - int, number of categories/markets/vendors for which
            revenue should be predicted
    
    Returns:
        full_df - pd.DataFrame similar to agg_data, but with additional rows for
            target days filled with NaNs
    """
    # if there were no purchases within some category/market/vendor during some day,
    # fill the corresponding values with zeros
    agg_data['full_discount_price'] = agg_data['full_discount_price'].fillna(0)
    agg_data['full_price'] = agg_data['full_price'].fillna(0)
    
    # get list of days for which forecast should be made
    target_days = get_target_days(result_order)
    
    # create dataframe with the same columns as in agg_data and with rows that
    # correspond to target days
    agg_data_nan = agg_data[:n_target_values * len(target_days)].copy()
    agg_data_nan.loc[:, agg_data.columns[4:]] = np.nan
    agg_data_nan.loc[:, "create_date"] = np.repeat(target_days, n_target_values)
    
    # merge dataframes
    full_df = agg_data.append(agg_data_nan, ignore_index=True)
    
    full_df.create_date = pd.to_datetime(full_df.create_date).dt.date
    
    # fill missing rows with data from previous year, for time series' continuity.
    full_df = copy_missing_days(full_df)
    return full_df

def create_column_names(mars, cats, vens):
    """
    Create column names where each column corresponds to a particular time series,
    e.g. to a particular market ID
    
    Args:
        mars - list of markets' IDs
        cats - list of categories' IDs
        vens - list of vendors' IDs
    
    Returns:
        col_names - dict with keys that are column names
    """
    col_names = {}
    for discount in ['full_discount_price', 'full_price']:

        for mar in mars:
            col_names[(discount, mar, np.nan, np.nan)] = []
        
        for cat in cats:
            col_names[(discount, np.nan, cat, np.nan)] = []
        
        for ven in vens:
            col_names[(discount, np.nan, np.nan, ven)] = []
        
        col_names[(discount, np.nan, np.nan, np.nan)] = []

    return col_names
    
def dataset_to_timeseries(target_ds):
    """
    Transform a dataset with aggregated purchases to a dataset where each column
    represents a separate time series that should be forecasted
    
    Args:
        target_ds - pd.DataFrame with continuous aggregated historical data
    
    Returns:
        time_series - pd.DataFrame where each column is a separate time series
    """
    # obtain market/category/vendor IDs of those markets/categories/vendors for which
    # forecast should be made
    mar_ids = [x for x in target_ds.market_id.unique() if (x != None) and (not np.isnan(x))]
    cat_ids = [x for x in target_ds.category_id.unique() if (x != None) and (not np.isnan(x))]
    ven_ids = [x for x in target_ds.vendor_id.unique() if (x != None) and (not np.isnan(x))]
    
    # create column names where each column corresponds to a particular time series,
    # e.g. to a particular market ID
    col_names = create_column_names(mar_ids, cat_ids, ven_ids)
    
    ts_full = []
    i = 0
    for col in col_names:
        # create time series that corresponds to column name col
        ts = pd.DataFrame()
        disc, mar, cat, ven = col
        if np.isnan(mar):
            expr1 = target_ds.market_id.isnull()
        else:
            expr1 = (target_ds.market_id==mar)
        if np.isnan(cat):
            expr2 = target_ds.category_id.isnull()
        else:
            expr2 = (target_ds.category_id==cat)
        if np.isnan(ven):
            expr3 = target_ds.vendor_id.isnull()
        else:
            expr3 = (target_ds.vendor_id==ven)

        expr = expr1&expr2&expr3
        
        if disc == 'full_discount_price':            
            ts = target_ds[expr][['full_discount_price', 'create_date']]
            ts.rename(columns={'full_discount_price': f'{mar:.0f}-{cat:.0f}-{ven:.0f}-{disc}'}, inplace=True)
        else:
            ts = target_ds[expr][['full_price', 'create_date']]
            ts.rename(columns={'full_price': f'{mar:.0f}-{cat:.0f}-{ven:.0f}-{disc}'}, inplace=True)
        
        ts.set_index('create_date', inplace=True)
        ts_full.append(ts)
    
    # concatenate time series
    time_series = pd.concat(ts_full, axis=1)
    return time_series