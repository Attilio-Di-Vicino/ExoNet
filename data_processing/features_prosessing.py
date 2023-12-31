import pandas as pd
from utils.mission import Mission
from utils.util import print_count_nan

def rows_id_nan(number_of_nan_rows, X_train, threshold=1):
    """Identifies rows in a DataFrame where the number of NaN
    values is greater than or equal
    to a specified threshold.

    Args:
        number_of_nan_rows (list): List containing the count of NaN values for each row.
        X_train (DataFrame): Original features.
        threshold (int): Threshold for the number of NaN values. Default is 0.

    Returns:
        list: List of row indices with NaN values greater than or equal to the threshold.
    """
    id_rows = []
    count = 0
    for id_row, nan_count in enumerate(number_of_nan_rows):
        if nan_count >= threshold:
            id_rows.append(id_row)
            count += 1
    print(f'Total rows >= of T={threshold}: {count} out of a total of {X_train.shape[0]}'
        f' By eliminating them you obtain {X_train.shape[0] - count} observations')
    return id_rows

def remove_non_numeric_columns(df):
    """Removes non-numeric columns from a DataFrame.

    Args:
        df (DataFrame): The input DataFrame.

    Returns:
        DataFrame: A new DataFrame with only numeric columns.
        List: non-numeric columns removed
    """
    non_numeric_columns = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
    df = df.drop(columns=non_numeric_columns)
    return df, non_numeric_columns

def features_to_preserve(mission=Mission.KEPLER):
    """Returns the features to be preserved and the corresponding label column name
    based on the specified mission.

    Args:
        mission (Mission): The mission name.

    Returns:
        tuple: (features_to_preserve, label_column)
        - features_to_preserve (list): List of features to be preserved.
        - label_column (str): Name of the label column.
        - -1 if the mission name is invalid.
    """
    if mission == Mission.KEPLER:
        features_to_preserve_ = [
            'koi_kepmag', 
            'koi_period', 'koi_period_err1', 'koi_duration', 'koi_duration_err1', 
            'koi_ror', 'koi_ror_err1', 'koi_insol',
            'koi_teq', 'koi_steff', 'koi_steff_err1', 'koi_slogg',
            'koi_slogg_err1', 'koi_srad', 'koi_srad_err1',
        ]
        label_column = 'koi_disposition'
    elif mission == Mission.TESS:
        features_to_preserve_ = [
            'TESS Mag',
            'Period (days)', 'Period (days) err',
            'Duration (hours)', 'Duration (hours) err',
            'Planet Radius (R_Earth)', 'Planet Radius (R_Earth) err',
            'Planet Insolation (Earth Flux)', 'Planet Equil Temp (K)',
            'Stellar Eff Temp (K)', 'Stellar Eff Temp (K) err',
            'Stellar log(g) (cm/s^2)', 'Stellar log(g) (cm/s^2) err',
            'Stellar Radius (R_Sun)', 'Stellar Radius (R_Sun) err'
        ]
        label_column = 'TFOPWG Disposition'
    elif mission == Mission.K2:
        features_to_preserve_ = [
            'sy_kepmag', 'pl_orbper', 'pl_orbpererr1', 'pl_trandur', 'pl_trandurerr1',
            'pl_rade', 'pl_radeerr1', 'pl_insol', 'pl_eqt', 'st_teff', 'st_tefferr1',
            'st_logg', 'st_loggerr1', 'st_rad', 'st_raderr1'
        ]
        label_column = 'disposition'
    elif mission == Mission.CTOI:
        features_to_preserve_ = [
            "TESS Mag",
            "Period (days)", "Period (days) Error", 
            "Duration (hrs)", "Duration (hrs) Error",
            "Planet Radius (R_Earth)", "Planet Radius (R_Earth) Error",
            'Insolation (Earth Flux)', "Equilibrium Temp (K)",
            'Stellar Eff Temp (K)', 'Stellar Eff Temp (K) err' 
            'Stellar log(g) (cm/s^2)', 'Stellar log(g) (cm/s^2) err',
            "Stellar Radius (R_Sun)", "Stellar Radius (R_Sun) err",
        ]
        label_column = "User Disposition"
    else:
        print("Warning: Invalid mission name.")
        return -1

    return features_to_preserve_, label_column
    
def remove_feature(data, list_to_preserve):
    """Remove unwanted features from the data.

    Args:
        data (DataFrame): The observation data.
        list_to_preserve (list): List of features to preserve.

    Returns:
        DataFrame: Modified observation data.
    """
    unwanted_features = [col for col in data.columns if col not in list_to_preserve]
    data = data.drop(unwanted_features, axis=1)
    return data
    
def rename_columns(data, mission=Mission.KEPLER):
    """Rename columns in Kepler, K2, or CTOI data.

    Args:
        data (DataFrame): Observations of Kepler, K2, or CTOI.
        mission (Mission): The mission name.

    Returns:
        DataFrame: Renamed observations.
    """
    common_kepler_tess = {
        'koi_disposition': 'TFOPWG Disposition',
        'koi_kepmag': 'TESS Mag',
        'koi_period': 'Period (days)',
        'koi_period_err1': 'Period (days) err',
        'koi_duration': 'Duration (hours)',
        'koi_duration_err1': 'Duration (hours) err',
        'koi_ror': 'Planet Radius (R_Earth)',
        'koi_ror_err1': 'Planet Radius (R_Earth) err',
        'koi_insol': 'Planet Insolation (Earth Flux)',
        'koi_teq': 'Planet Equil Temp (K)',
        'koi_steff': 'Stellar Eff Temp (K)',
        'koi_steff_err1': 'Stellar Eff Temp (K) err',
        'koi_slogg': 'Stellar log(g) (cm/s^2)',
        'koi_slogg_err1': 'Stellar log(g) (cm/s^2) err',
        'koi_srad': 'Stellar Radius (R_Sun)',
        'koi_srad_err1': 'Stellar Radius (R_Sun) err',
    }
    common_k2_tess = {
        'disposition': 'TFOPWG Disposition',
        'sy_kepmag': 'TESS Mag',
        'pl_orbper': 'Period (days)',
        'pl_orbpererr1': 'Period (days) err',
        'pl_trandur': 'Duration (hours)',
        'pl_trandurerr1': 'Duration (hours) err',
        'pl_rade': 'Planet Radius (R_Earth)',
        'pl_radeerr1': 'Planet Radius (R_Earth) err',
        'pl_insol': 'Planet Insolation (Earth Flux)',
        'pl_eqt': 'Planet Equil Temp (K)',
        'st_teff': 'Stellar Eff Temp (K)',
        'st_tefferr1': 'Stellar Eff Temp (K) err',
        'st_logg': 'Stellar log(g) (cm/s^2)',
        'st_loggerr1': 'Stellar log(g) (cm/s^2) err',
        'st_rad': 'Stellar Radius (R_Sun)',
        'st_raderr1': 'Stellar Radius (R_Sun) err',
    }
    common_ctoi_toi = {
        'User Disposition': 'TFOPWG Disposition',
        'TESS Mag': 'TESS Mag',
        'Period (days)': 'Period (days)',
        'Period (days) Error': 'Period (days) err',
        'Duration (hrs)': 'Duration (hours)',
        'Duration (hrs) Error': 'Duration (hours) err',
        'Planet Radius (R_Earth)': 'Planet Radius (R_Earth)',
        'Planet Radius (R_Earth) Error': 'Planet Radius (R_Earth) err',
        'Insolation (Earth Flux)': 'Planet Insolation (Earth Flux)',
        'Equilibrium Temp (K)': 'Planet Equil Temp (K)',
        'Stellar Eff Temp (K)': 'Stellar Eff Temp (K)',
        'Stellar Eff Temp (K) err': 'Stellar Eff Temp (K) err',
        'Stellar log(g) (cm/s^2)': 'Stellar log(g) (cm/s^2)',
        'Stellar log(g) (cm/s^2) err': 'Stellar log(g) (cm/s^2) err',
        'Stellar Radius (R_Sun)': 'Stellar Radius (R_Sun)',
        'Stellar Radius (R_Sun) err': 'Stellar Radius (R_Sun) err',
    }
    common_mappings = {
        Mission.KEPLER: common_kepler_tess,
        Mission.K2: common_k2_tess,
        Mission.CTOI: common_ctoi_toi,
    }

    if mission in common_mappings:
        data.rename(columns=common_mappings[mission], inplace=True)
    else:
        print("Warning: Invalid mission name")
        return -1
    return data

def feature_processing(kepler_data, ctoi_data, k2_data, tess_data):
    """Process features for Kepler and TESS data.

    Args:
        kepler_data (DataFrame): Observations of Kepler.
        ctoi_data (DataFrame): Observations of CTOI.
        k2_data (DataFrame): Observations of K2.
        tess_data (DataFrame): Observations of TESS.

    Returns:
        tuple: (X_train, y_train)
        - X_train (DataFrame): Combined observations.
        - y_train (Series): Combined labels.
    """
    # Save value of label
    list_of_kepler, kepler_label_index = features_to_preserve(mission=Mission.KEPLER)
    list_of_tess, tess_label_index = features_to_preserve(mission=Mission.TESS)
    list_of_k2, k2_label_index = features_to_preserve(mission=Mission.K2)
    list_of_ctoi, ctoi_label_index = features_to_preserve(mission=Mission.CTOI)

    kepler_label = kepler_data[kepler_label_index]
    tess_label = tess_data[tess_label_index]
    k2_label = k2_data[k2_label_index]
    ctoi_label = ctoi_data[ctoi_label_index]

    # Drop unwanted features
    kepler_data = remove_feature(data=kepler_data, list_to_preserve=list_of_kepler)
    tess_data = remove_feature(data=tess_data, list_to_preserve=list_of_tess)
    k2_data = remove_feature(data=k2_data, list_to_preserve=list_of_k2)
    ctoi_data = remove_feature(data=ctoi_data, list_to_preserve=list_of_ctoi)

    # Print not a number
    print('\nAfter the processing:')
    print_count_nan(data=tess_data, name='TESS   ')
    print_count_nan(data=kepler_data, name='Kepler ')
    print_count_nan(data=k2_data, name='K2     ')
    print_count_nan(data=ctoi_data, name='CTOI   ')

    # Rename columns in kepler_data
    kepler_data = rename_columns(data=kepler_data, mission=Mission.KEPLER)
    k2_data = rename_columns(data=k2_data, mission=Mission.K2)
    ctoi_data = rename_columns(data=ctoi_data, mission=Mission.CTOI)

    # Combine dataframes
    X_train = pd.concat([kepler_data, k2_data, ctoi_data, tess_data], ignore_index=True)
    y_train = pd.concat([kepler_label, k2_label, ctoi_label, tess_label], ignore_index=True)
    return X_train, y_train

def feature_processing_kepler_tess(kepler_data, tess_data):
    """Process features for Kepler and TESS data.

    Args:
        kepler_data (DataFrame): Observations of Kepler.
        tess_data (DataFrame): Observations of TESS.

    Returns:
        tuple: (X_train, y_train)
        - X_train (DataFrame): Combined observations.
        - y_train (Series): Combined labels.
    """
    # Save value of label
    list_of_kepler, kepler_label_index = features_to_preserve_kepler_tess(mission=Mission.KEPLER)
    list_of_tess, tess_label_index = features_to_preserve_kepler_tess(mission=Mission.TESS)

    kepler_label = kepler_data[kepler_label_index]
    tess_label = tess_data[tess_label_index]

    # Drop unwanted features
    kepler_data = remove_feature(data=kepler_data, list_to_preserve=list_of_kepler)
    tess_data = remove_feature(data=tess_data, list_to_preserve=list_of_tess)

    # Print not a number
    print('\nAfter the processing:')
    print_count_nan(data=tess_data, name='TESS   ')
    print_count_nan(data=kepler_data, name='Kepler ')

    # Rename columns in kepler_data
    kepler_data = rename_columns_kepler_tess(data=kepler_data, mission=Mission.KEPLER)

    # Combine dataframes
    X_train = pd.concat([kepler_data, tess_data], ignore_index=True)
    y_train = pd.concat([kepler_label, tess_label], ignore_index=True)
    return X_train, y_train

def rename_columns_kepler_tess(data, mission=Mission.KEPLER):
    """Rename columns in Kepler, K2, or CTOI data.

    Args:
        data (DataFrame): Observations of Kepler, K2, or CTOI.
        mission (Mission): The mission name.

    Returns:
        DataFrame: Renamed observations.
    """
    common_kepler_tess = {
        'koi_disposition': 'TFOPWG Disposition', # Disposition
        'koi_period': 'Period (days)', # The interval between consecutive planetary transits
        'koi_period_err1': 'Period (days) err',
        'koi_time0': 'Epoch (BJD)', # The time corresponding to the center of the first detected transit in Barycentric Julian Day (BJD)
        'koi_time0_err1': 'Epoch (BJD) err',
        'koi_duration': 'Duration (hours)', # The duration of the observed transits
        'koi_duration_err1': 'Duration (hours) err',
        'koi_depth': 'Depth (ppm)', # The fraction of stellar flux lost at the minimum of the planetary transit
        'koi_depth_err1': 'Depth (ppm) err',
        'koi_prad': 'Planet Radius (R_Earth)', # Planet Radius (R_Earth)
        'koi_ror_err1': 'Planet Radius (R_Earth) err',
        'koi_insol': 'Planet Insolation (Earth Flux)', # Insolation flux is another way to give the equilibrium temperature
        'koi_teq': 'Planet Equil Temp (K)', # Equilibrium Temperature (Kelvin)
        'koi_steff': 'Stellar Eff Temp (K)', # Stellar Effective Temperature (Kelvin)
        'koi_steff_err1': 'Stellar Eff Temp (K) err',
        'koi_slogg': 'Stellar log(g) (cm/s^2)', # Stellar Surface Gravity (log10(cm s^-2)
        'koi_slogg_err1': 'Stellar log(g) (cm/s^2) err',
        'koi_srad': 'Stellar Radius (R_Sun)', # Stellar Radius (solar radii)
        'koi_srad_err1': 'Stellar Radius (R_Sun) err',
        'koi_smet': 'Stellar Metallicity', # Stellar Metallicity
        'koi_smet_err1': 'Stellar Metallicity err',
        'koi_srad': 'Stellar Mass (M_Sun)', # Stellar Mass
        'koi_srad_err1': 'Stellar Mass (M_Sun) err',
        ################# I PROSSIMI 4 SONO CORRETTI ? #################
        'koi_kepmag': 'TESS Mag',
        'koi_model_snr': 'Planet SNR', # Transit depth normalized by the mean uncertainty in the flux during the transits
        'ra': 'RA', # Right Ascension of the planetary system in decimal degrees
        'dec': 'Dec', # Declination of the planetary system in sexagesimal
    }
    common_mappings = {
        Mission.KEPLER: common_kepler_tess,
    }
    if mission in common_mappings:
        data.rename(columns=common_mappings[mission], inplace=True)
    else:
        print("Warning: Invalid mission name")
        return -1
    return data

def features_to_preserve_kepler_tess(mission=Mission.KEPLER):
    """Returns the features to be preserved and the corresponding label column name
    based on the specified mission.

    Args:
        mission (Mission): The mission name.

    Returns:
        tuple: (features_to_preserve, label_column)
        - features_to_preserve (list): List of features to be preserved.
        - label_column (str): Name of the label column.
        - -1 if the mission name is invalid.
    """
    if mission == Mission.KEPLER:
        features_to_preserve_ = [
            'koi_period', 'koi_period_err1', 'koi_time0',
            'koi_time0_err1', 'koi_duration', 'koi_duration_err1',
            'koi_depth', 'koi_depth_err1', 'koi_prad', 'koi_ror_err1',
            'koi_insol', 'koi_teq', 'koi_steff', 'koi_steff_err1',
            'koi_slogg', 'koi_slogg_err1', 'koi_srad', 'koi_srad_err1',
            'koi_smet', 'koi_smet_err1', 'koi_srad', 'koi_srad_err1',
            'koi_kepmag', 'koi_model_snr', 'ra', 'dec',
        ]
        label_column = 'koi_disposition'
    elif mission == Mission.TESS:
        features_to_preserve_ = [
            'Period (days)', 'Period (days) err', 'Epoch (BJD)', 
            'Epoch (BJD) err', 'Duration (hours)', 'Duration (hours) err', 'Depth (ppm)', 
            'Depth (ppm) err', 'Planet Radius (R_Earth)', 'Planet Radius (R_Earth) err',
            'Planet Insolation (Earth Flux)', 'Planet Equil Temp (K)', 'Stellar Eff Temp (K)', 
            'Stellar Eff Temp (K) err', 'Stellar log(g) (cm/s^2)', 'Stellar log(g) (cm/s^2) err',
            'Stellar Radius (R_Sun)', 'Stellar Radius (R_Sun) err', 'Stellar Metallicity',
            'Stellar Metallicity err', 'Stellar Mass (M_Sun)', 'Stellar Mass (M_Sun) err',
            'TESS Mag', 'Planet SNR', 'RA', 'Dec', 
        ]
        label_column = 'TFOPWG Disposition'
    else:
        print("Warning: Invalid mission name.")
        return -1
    return features_to_preserve_, label_column

def remove_nan_label(X_train, y_train):
    """Remove rows with NaN labels in y_train and correspondingly from X_train.

    Args:
        X_train (DataFrame): Observations.
        y_train (Series): Labels.

    Returns:
        tuple: (X_train, y_train)
        - X_train (DataFrame): Edited observations.
        - y_train (Series): Edited labels.
    """
    # Identify row indices that contain NaN values in the label column
    nan_indices = y_train[y_train.isna()].index
    # Remove matching rows
    X_train = X_train.drop(nan_indices).reset_index(drop=True)
    y_train = y_train.drop(nan_indices).reset_index(drop=True)
    return X_train, y_train