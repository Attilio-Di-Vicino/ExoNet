import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from utils.color import Color

def plot_top_7_difference(X_train):
    """Visualize the top seven features with the greatest difference between maximum and minimum values.

    Args:
        X_train (DataFrame): Observations.

    Returns:
        None
    """
    # Compute the difference between the maximum and minimum for each column
    diff_values = X_train.max() - X_train.min()
    # Select the top 7 columns with the largest differences
    top_7_diff = diff_values.nlargest(7)
    # Plot the top 7 features
    plt.figure(figsize=(10, 6))
    plt.bar(top_7_diff.index, top_7_diff.values, edgecolor=Color.SPACE.value,
                    color=Color.SEA.value)
    plt.xlabel('Features')
    plt.ylabel('Difference')
    plt.title('Top 7 Features with the Greatest Difference')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha="right", ticks=top_7_diff.index)
    plt.show()

def data_scaling_normalization(X_train, with_mean=True, with_std=True, print_mean_scale=False):
    """Standardize features by removing the mean and scaling to unit variance.

    The standard score of a sample x is calculated as:
        z = (x - u) / s
    where u is the mean of the training samples or zero if with_mean=False,
    and s is the standard deviation of the training samples or one if with_std=False.

    Args:
        X_train (DataFrame): Observations.

    Returns:
        DataFrame: X_train normalized.
    """
    scaler = StandardScaler(with_mean=with_mean, with_std=with_std).fit(X_train)
    if print_mean_scale:
        print('Mean: ', scaler.mean_)
        print('Scale:', scaler.scale_)
    X_train_normalized = scaler.transform(X_train)
    # Create a pandas DataFrame from the normalized data
    X_train_normalized_df = pd.DataFrame(X_train_normalized, columns=X_train.columns)
    return X_train_normalized_df
