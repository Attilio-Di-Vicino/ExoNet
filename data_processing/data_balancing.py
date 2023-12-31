from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import numpy as np

def smote_oversampling(X_train, y_train, random_state):
    """Applies SMOTE (Synthetic Minority Over-sampling Technique) to balance the class distribution.

    Args:
        X_train (DataFrame): Features.
        y_train (Series): Target labels of the training set.
        random_state (int): Random seed.

    Returns:
        DataFrame: Balanced features.
        Series: Balanced labels.
    """
    smote = SMOTE(random_state=random_state)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    return X_train_balanced, y_train_balanced

def compute_mean_std(X_train, X_train_balanced):
    """Computes mean and standard deviation before and after balancing.

    Args:
        X_train (DataFrame): Original features.
        X_train_balanced (DataFrame): Balanced features.

    Returns:
        dict: Statistics before balancing.
        dict: Statistics after balancing.
    """
    stats_before_balancing = {
        'mean': np.mean(X_train),
        'std': np.std(X_train)
    }
    stats_after_balancing = {
            'mean': np.mean(X_train_balanced),
            'std': np.std(X_train_balanced)
    }
    return stats_before_balancing, stats_after_balancing

def plot_mean(X_train, stats_before_balancing, stats_after_balancing):
    """Plots the mean of features before and after balancing.

    Args:
        X_train (DataFrame): Original features.
        stats_before_balancing (dict): Statistics before balancing.
        stats_after_balancing (dict): Statistics after balancing.

    Returns:
        None
    """
    mean_before = abs(stats_before_balancing['mean']) 
    mean_after = abs(stats_after_balancing['mean']) 
    _, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    index = np.arange(len(X_train.columns))
    _ = ax.bar(index, mean_before, bar_width, label='Before balancing')
    _ = ax.bar(index + bar_width, mean_after, bar_width, label='After balancing')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(X_train.columns, rotation=90)
    ax.set_ylabel('Mean')
    ax.set_title(f'Mean of Features Before and After balancing')
    ax.legend()
    plt.show()

def plot_std(X_train, stats_before_balancing, stats_after_balancing):
    """Plots the standard deviation of features before and after balancing.

    Args:
        X_train (DataFrame): Original features.
        stats_before_balancing (dict): Statistics before balancing.
        stats_after_balancing (dict): Statistics after balancing.

    Returns:
        None
    """
    std_before = stats_before_balancing['std'] 
    std_after = stats_after_balancing['std']
    _, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    index = np.arange(len(X_train.columns))
    _ = ax.bar(index, std_before, bar_width, label='Before balancing')
    _ = ax.bar(index + bar_width, std_after, bar_width, label='After balancing')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(X_train.columns, rotation=90)
    ax.set_ylabel('Standard Devation')
    ax.set_title(f'Standard Devation of Features Before and After balancing')
    ax.legend()
    plt.show()