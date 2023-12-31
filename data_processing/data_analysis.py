import numpy as np

def compute_all_columns_nan(data, name='X_train'):
    """Compute columns that contain only not a number values.

    Args:
        data (DataFrame): Dataset.
        dataset_name (str, optional): Name of the dataset. Defaults to 'X_train'.

    Returns:
        list: Columns that contain only not a number values.
    """
    nan_columns_name = [col for col in data.columns if all(np.isnan(value) for value in data[col])]
    print(f'Columns containing only not a number in {name}: {nan_columns_name}')
    return nan_columns_name

def print_nan_numbers_for_features(data, number_of_nan_columns, name='X_train'):
    """Print the number of not a number values for each feature in the dataset.

    Args:
        data (DataFrame): Dataset.
        number_of_nan_columns (dict): Dictionary containing the count of NaN values for each feature.
        name (str, optional): Name of the dataset. Defaults to 'X_train'.

    Returns:
        None
    """
    print(f'Number of Observations: {data.shape[0]}')
    print('NaN for each feature')

    for index, (feature, nan_count) in enumerate(number_of_nan_columns.items(), start=1):
        index_str = str(index).rjust(3)
        print(f"{index_str}: {feature.ljust(30, '-')}> {nan_count}")

    total_nan_count = data.isna().sum().sum()
    print(f'Total number of not a number in {name}: {total_nan_count}')

def threshold_delete_nan(number_of_nan_columns, nan_columns_name, threshold):
    """Delete features with not a number values exceeding the given threshold.

    Args:
        number_of_nan_columns (dict): Dictionary containing the count of NaN values for each feature.
        nan_columns_name (list): List of columns to be deleted.
        threshold (int): Threshold value for not a number count.

    Returns:
        list: Updated list of columns to be deleted.
    """
    for feature, nan_count in number_of_nan_columns.items():
        if nan_count > threshold:
            nan_columns_name.append(feature)
    return nan_columns_name