import pandas as pd
import datetime
import time
import numpy as np
from tqdm import tqdm

import warnings
warnings.filterwarnings("default", category=FutureWarning)


bound_dict ={'market_id': [104448, 103297, 100610, 104705, 104837, 106629, 106376, 104335, 103952, 106130, 276, 103961, 105498, 105502, 105375, 105503, 299, 107313, 310, 104374, 314, 100031, 104768, 106563, 102725, 103244, 104524, 105036, 105935, 106703, 105701, 105702, 103529, 105071, 103664, 105072, 105970, 104572], 
             'vendor_id': [1799, 264, 3847, 139, 103436, 17, 1937, 19, 4370, 27, 28, 30, 31, 34, 36, 38, 104106, 44, 301, 49, 59, 3134, 104008, 330, 1740, 2763, 599, 728, 1624, 2648, 2654, 735, 3806, 97, 610, 4321, 1638, 1639, 3686, 1257, 1645, 2291, 104573], 
             'category_id': [128, 6147, 6150, 8, 9, 140, 13, 142, 16, 6161, 146, 149, 6166, 24, 25, 27, 6171, 31, 32, 161, 39, 43, 49, 6066, 51, 6068, 6195, 56, 58, 187, 60, 61, 6207, 194, 67, 80, 208, 217, 6123, 112, 113, 115, 116, 117, 240, 247]}
agg_columns = ['market_id', 'category_id', 'vendor_id']
pred_columns = ['full_discount_price', 'full_price']


def get_preprocessed_data(dataset_path):
    """
    Reads an original dataset, removes outliers
    and drops unnecessary columns and rows.

    Args:
        dataset_path - a path to a file with original dataset
    Returns:
        preprocessed dataset
    """

    total_steps = 8
    step = 1
    print('Read original dataset', f'({step}/{total_steps})')
    step += 1
    df = pd.read_csv(dataset_path, index_col=False)
                
    ## Drop unnecessary columns
    print('Drop unnecessary columns', f'({step}/{total_steps})')
    step += 1
    unnecessary_columns = [
        'order_item_id',
        'region',
        'refund_date',
        'subscription_type',
        'nds',
        'nds_percent',
        'price_group_id',
        'currency_name',
        'ratio2rub',
        'ratio2ue',
        'ratio2rub_official',
        'ratio2ue_official',
        'zone',
        'locale',
        'country_id',
        'update_date',
        'payment_date',
        'delivery_date',
        'discount_percent',
        'group_id',
        'currency_id',
        'delivery_date',
        'shop_id',
        'price',
        'boxweight',
        'full_price_in_order_currency',
        'dostavlen',
        'payment_is_auto',
        'payment_is_error',
        'Unnamed: 0',
    ]
    unnecessary_columns = sorted(unnecessary_columns)
    print('Columns to be dropped:', *unnecessary_columns, sep='\n\t')
    df.drop(labels=unnecessary_columns, axis=1, inplace=True)

    ## Change columns datatypes
    print('Change columns datatypes', f'({step}/{total_steps})')
    step += 1
    df['create_date']=pd.to_datetime(df.create_date).dt.date

    ## Drop rare status and STATUS column
    print('Drop rare status and STATUS column', f'({step}/{total_steps})')
    step += 1
    df.drop(df[df.STATUS == "D"].index, inplace=True)
    df.drop('STATUS', axis=1, inplace=True)

    ## Drop outliers in data
    print('Drop outliers in data', f'({step}/{total_steps})')
    step += 1
    df = df[(df.full_price < 10000) & (df.price_with_discount < 1e5) & (df.price_wo_discount < 1e5) & (df.discount_price < 10000)]
    
    ## Drop price_with_discount, price_wo_discount columns and discount_price
    print('Drop price_with_discount, price_wo_discount columns and discount_price', f'({step}/{total_steps})')
    step += 1
    df.drop(labels=['price_with_discount', 'price_wo_discount', 'discount_price'], axis=1, inplace=True)
    
    
    ## Change column types
    print('Change column types', f'({step}/{total_steps})')
    step += 1
    df.vendor_id = df.vendor_id.fillna(0).astype(int)
    df['category_id'] = df.category_id.fillna(0).astype(int)
    df['full_price'] = df.full_price.astype(int)
    df['full_discount_price'] = df.full_discount_price.astype(int)

    ## Drop if full_price is zero
    print('Drop if full_price is zero', f'({step}/{total_steps})')
    step += 1
    df = df[df.full_price != 0.]

    return df


def sort_columns(df):
    """
    Makes a specific order of columns in a dataset.

    Args:
        df - a dataset
    Returns:
        The same dataset, but its order of columns is changed
    """

    columns = sorted(df.columns)
    columns.remove("category_id")
    columns.remove("market_id")
    columns.remove("vendor_id")
    columns.remove("create_date")
    columns = ["create_date", "category_id", "market_id", "vendor_id"] + columns
    return df[columns]


def get_target_vals(df):
    """
    Groups by agg_columns and computes mean for each group.
    Adds sum of full_price, full_discount_price and quantity for each group.
    Creates special values that allows to infer by which column a data was groupped.

    Args:
        df - a dataset
    Returns:
        A dataset which is a result of mergeing the dataset groupped by different columns
    """

    s = pd.Series()
    for column in agg_columns:
        agg_group = df.groupby([column])
        agg_df = agg_group.mean()
        agg_df["count"] = agg_group.size()
        agg_df[["full_price_sum", "full_discount_price_sum", "quantity_sum"]] = agg_group[["full_price", 
                                                                                       "full_discount_price",
                                                                                       "quantity"]].sum()
        agg_df = agg_df.loc[bound_dict[column]].reset_index()
        for other_agg in set(agg_columns)-{column}:
            agg_df[other_agg] = np.nan

        s = s.append(agg_df, ignore_index=True)


    full_stats = df.mean()
    full_stats["count"] = df.index.size
    full_stats["full_price_sum"] = df["full_price"].sum()
    full_stats["full_discount_price_sum"] = df["full_discount_price"].sum()
    full_stats["quantity_sum"] = df["quantity"].sum()
    
    full_stats[agg_columns] = np.nan
    s = s.append(full_stats, ignore_index=True)
    s.drop(0,axis=1, inplace=True)
    return s.reset_index(drop=True)


def rename_target_columns(df):
    """
    Renames target columns.

    Args:
        df - a dataset
    Returns:
        The dataset with renamed columns.
    """


    # full_price
    df['full_price_mean'] = df['full_price']
    df['full_price'] = df['full_price_sum']
    df.drop('full_price_sum', axis=1, inplace=True)

    # full_discount_price
    df['full_discount_price_mean'] = df['full_discount_price']
    df['full_discount_price'] = df['full_discount_price_sum']
    df.drop('full_discount_price_sum', axis=1, inplace=True)

    # quantity
    df['quantity_mean'] = df['quantity']
    df['quantity'] = df['quantity_sum']
    df.drop('quantity_sum', axis=1, inplace=True)

    return df

def get_aggregated_data(df):
    """
    Creates a dataset that contains a data aggregated by market_id, vendor_id and category_id
    for each day in a dataset.

    Args:
        df - a dataset
    Returns:
        Aggregated version of the dataset that contains all target values
    """

    new_df = pd.DataFrame()
    cd = df.create_date.values

    if hasattr(tqdm, "_instances"):  # prevents many printing issues
        tqdm._instances.clear()

    for date in tqdm(df.create_date.unique()):
        tmp_df = get_target_vals(df[cd == date])
        tmp_df['create_date'] = date
        new_df = new_df.append(tmp_df, ignore_index=True, sort=True)
    new_df = sort_columns(new_df)
    new_df = rename_target_columns(new_df)
    new_df.drop(labels=['count', 'full_price_mean', 'full_discount_price_mean', 'quantity_mean', 'quantity'], axis=1, inplace=True)
    return new_df
    