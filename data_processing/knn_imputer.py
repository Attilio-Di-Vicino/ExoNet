from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from utils.util import print_count_nan

def get_q_approximation(q):
    """Compute an approximation of q to produce odd numbers
    from 0 to the square root of N.

    Args:
        q (int): Integer.

    Returns:
        int: Approximation of q.
    """
    return 2 * q + 1

def compute_values_of_k(X_train):
    """Calculate the possible values of k.

    Args:
        X_train (DataFrame): Observations.

    Returns:
        list: Vector of values of k.
    """
    square_root = math.isqrt(len(X_train))
    q_max = (square_root - 1) // 2
    values_of_k = [get_q_approximation(i) for i in range(q_max)]
    return values_of_k

def k_nearest_neighbors_imputer(X_train, index_of_k=None):
    """KNN imputer is a method often used for handling missing values. It calculates the similarity
    between the known examples and the observation with the missing value.
    Select the k most similar known samples and use their values to calculate an
    estimated value for the missing data. Similarity can be measured using different metrics,
    such as Euclidean distance or Manhattan distance. In this case, Euclidean.

    Args:
        X_train (DataFrame): Observations.
        index_of_k (int, optional): Index for vector values of k. Defaults to None.

    Returns:
        DataFrame: Imputed observations.
    """
    total_nan = print_count_nan(data=X_train, name='X_train')
    if total_nan == 0:
        print('There are no NaN')
        return X_train
    # Calculation of the possible values ​​of k with approximation
    values_of_k = compute_values_of_k(X_train=X_train)
    # Set k
    k = values_of_k[-1] if index_of_k is None else values_of_k[index_of_k]
    # Creating the imputer object
    # ‘distance’: weight points by the inverse of their distance.
    # In this case, closer neighbors of a query point will have a greater
    # influence than neighbors which are further away.
    imputer = KNNImputer(n_neighbors=k, weights='distance')
    # Fit and transform data
    X_train_imputed = imputer.fit_transform(X_train)
    # DataFrame imputed
    X_train_imputed = pd.DataFrame(X_train_imputed, columns=X_train.columns)
    return X_train_imputed

def compute_mean_std_knn(X_train, k_values):
    """Computes mean and standard deviation before and after imputation using KNN.

    Args:
        X_train (DataFrame): Original features.
        k_values (list): List of k values for KNN imputation.

    Returns:
        dict: Statistics before imputation.
        dict: Statistics after imputation.
    """
    stats_before_imputation = {}
    stats_after_imputation = {}
    for k_value in k_values:
        stats_before_imputation[k_value] = {
            'mean': np.mean(X_train),
            'std': np.std(X_train)
        }
        knn_imputer = KNNImputer(n_neighbors=k_value, weights='distance')
        X_train_imputed = knn_imputer.fit_transform(X_train)
        X_train_imputed = pd.DataFrame(X_train_imputed, columns=X_train.columns)
        stats_after_imputation[k_value] = {
            'mean': np.mean(X_train_imputed),
            'std': np.std(X_train_imputed)
        }
    return stats_before_imputation, stats_after_imputation

def print_and_compute_mean_std(X_train, k_values, stats_before_imputation, stats_after_imputation):
    """Prints and computes mean and standard deviation differences before and after imputation
    for different k values.

    Args:
        X_train (DataFrame): Original features.
        k_values (list): List of k values for KNN imputation.
        stats_before_imputation (dict): Statistics before imputation.
        stats_after_imputation (dict): Statistics after imputation.

    Returns:
        dict: Sum of mean differences for each k.
        dict: Sum of std differences for each k.
    """
    sum_difference_mean = {}
    sum_difference_std = {}

    for k_value in k_values:
        sum_difference_mean[k_value] = 0
        sum_difference_std[k_value] = 0
        print('\n')
        print(f"----BEFORE-AND-AFTER-IMPUTATION-K={k_value}----")
        print('-------------------Mean------------------------')
        for index in X_train.columns:
            before_mean = stats_before_imputation[k_value]["mean"][index]
            after_mean = stats_after_imputation[k_value]["mean"][index]
            difference = abs(before_mean - after_mean)
            sum_difference_mean[k_value] += difference
            print(f'{index.ljust(30, "-")}> Before: {before_mean:.2f} After: {after_mean:.2f} Difference: {difference:.2f}')
        print(f'Sum of Difference: {sum_difference_mean[k_value]:.2f}')

        print('-------------Standard Deviation----------------')
        for index in X_train.columns:
            before_std = stats_before_imputation[k_value]["std"][index]
            after_std = stats_after_imputation[k_value]["std"][index]
            difference = abs(before_std - after_std)
            sum_difference_std[k_value] += difference
            print(f'{index.ljust(30, "-")}> Before: {before_std:.2f} After: {after_std:.2f} Difference: {difference:.2f}')
        print(f'Sum of Difference: {sum_difference_std[k_value]:.2f}')
    return sum_difference_mean, sum_difference_std

def plot_diff_mean_std(k_values, sum_difference_mean, sum_difference_std):
    """Plots the sum of mean and std differences between before and after imputation
    for different k values.

    Args:
        k_values (list): List of k values for KNN imputation.
        sum_difference_mean (dict): Sum of mean differences for each k.
        sum_difference_std (dict): Sum of std differences for each k.

    Returns:
        None
    """
    sum_difference_mean_values = [sum_difference_mean[k] for k in k_values]
    sum_difference_std_values = [sum_difference_std[k] for k in k_values]
    _, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_values, sum_difference_mean_values, marker='o', label='Sum of Mean Differences')
    ax.plot(k_values, sum_difference_std_values, marker='o', label='Sum of Std Differences')
    ax.set_xlabel('K Values')
    ax.set_ylabel('Sum of Differences')
    ax.set_title('Sum of Differences between Before and After Imputation for Different K Values')
    ax.legend()
    plt.show()

def compute_best_value_of_k(k_values, sum_difference_mean, sum_difference_std):
    """Computes the best value of k based on the total difference.

    Args:
        k_values (list): List of k values for KNN imputation.
        sum_difference_mean (dict): Sum of mean differences for each k.
        sum_difference_std (dict): Sum of std differences for each k.

    Returns:
        int: Best value of k.
    """
    best_k = None
    min_total_difference = float('inf')
    for k_value in k_values:
        total_difference = sum_difference_mean[k_value] + sum_difference_std[k_value]
        if total_difference < min_total_difference:
            min_total_difference = total_difference
            best_k = k_value
    return best_k

def plot_mean_best_k(X_train, best_k, stats_before_imputation, stats_after_imputation):
    """Plots the mean of features before and after imputation for the best k.

    Args:
        X_train (DataFrame): Original features.
        best_k (int): Best value of k.
        stats_before_imputation (dict): Statistics before imputation.
        stats_after_imputation (dict): Statistics after imputation.

    Returns:
        None
    """
    mean_before_imputation = stats_before_imputation[best_k]['mean'] 
    mean_after_imputation = stats_after_imputation[best_k]['mean'] 
    _, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    index = np.arange(len(X_train.columns))
    _ = ax.bar(index, mean_before_imputation, bar_width, label='Before Imputation')
    _ = ax.bar(index + bar_width, mean_after_imputation, bar_width, label=f'After Imputation (k={best_k})')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(X_train.columns, rotation=90)
    ax.set_ylabel('Mean')
    ax.set_title(f'Mean of Features Before and After Imputation (k={best_k})')
    ax.legend()
    plt.show()

def plot_std_best_k(X_train, best_k, stats_before_imputation, stats_after_imputation):
    """Plots the standard deviation of features before and after imputation for the best k.

    Args:
        X_train (DataFrame): Original features.
        best_k (int): Best value of k.
        stats_before_imputation (dict): Statistics before imputation.
        stats_after_imputation (dict): Statistics after imputation.

    Returns:
        None
    """
    std_before_imputation = stats_before_imputation[best_k]['std'] 
    std_after_imputation = stats_after_imputation[best_k]['std'] 
    _, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    index = np.arange(len(X_train.columns))
    _ = ax.bar(index, std_before_imputation, bar_width, label='Before Imputation')
    _ = ax.bar(index + bar_width, std_after_imputation, bar_width, label=f'After Imputation (k={best_k})')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(X_train.columns, rotation=90)
    ax.set_ylabel('Standard Deviation')
    ax.set_title(f'Standard Deviation of Features Before and After Imputation (k={best_k})')
    ax.legend()
    plt.show()