import matplotlib.pyplot as plt
import pandas as pd

from utils.color import Color
from utils.plot_style import PlotStyle


def plot_class_distribution(y_train):
    """View the distribution of classes:
    'CONFIRMED/CANDIDATE': 1, 'FALSE POSITIVE': 0.

    Args:
        y_train (Series): Labels.

    Returns:
        class_count_1 (int): Number of obsrvation labeled 1
        class_count_0 (int): Number of observations labeled 0
    """
    class_count_1 = y_train.sum().item()
    class_count_0 = len(y_train) - class_count_1
    plt.figure(figsize=(8, 6))
    plt.bar(['CONFIRMED/CANDIDATE', 'FALSE POSITIVE'], [class_count_1, class_count_0], 
            color=[Color.SEA.value, Color.SKY.value])
    plt.xlabel('Classes')
    plt.ylabel('Count')
    plt.title('Distribution of classes in the dataset')
    plt.show()
    return class_count_1, class_count_0

def plot_feature_importances(X_train, feature_importances, max_features=None, fig_size=(8, 24),
                              decimal_places=4):
    """Plot the Feature Importances.

    Args:
        X_train (DataFrame): DataFrame containing the features.
        feature_importances (array-like): Feature Importances.

    Returns:
        None
    """
    if len(feature_importances) != len(X_train.columns):
        raise ValueError("The length of feature_importances must be"
                          "equal to the number of columns in X_train")
    df_importances = pd.DataFrame({'Features': X_train.columns, 'Importances': feature_importances})
    df_importances = df_importances.sort_values(by='Importances', ascending=True)
    # Limits the number of features if max_features is specified
    if max_features is not None:
        df_importances = df_importances.tail(max_features)
    colors = [Color.SEA.value if i % 2 == 0 else Color.SKY.value 
              for i in range(len(df_importances))]
    plt.figure(figsize=fig_size)
    bars = plt.barh(df_importances['Features'], df_importances['Importances'], color=colors)
    plt.ylabel('Features')
    plt.xlabel('Importances')
    plt.title('Feature Importances')
    plt.grid(axis='x', linestyle='-', alpha=0.7)
    for bar, color, _, importance in zip(bars, colors, df_importances['Features'], 
                                         df_importances['Importances']):
        bar.set_color(color)
        plt.text(bar.get_width() + 0.0001, bar.get_y() + bar.get_height() / 2,
                 f'{importance:.{decimal_places}}',
                 ha='left', va='center', color=color)
    plt.subplots_adjust(right=1.5)
    plt.show()

def plot_scatter_feature_distributions(X_train, plot_style=PlotStyle.POINT):
    """Visualizza la distribuzione dei valori per una singola feature in X_train.

    Args:
        X_train (Series): Serie contenente i valori della feature.

    Returns:
        None
    """
    plt.figure(figsize=(14, 6))
    if plot_style == PlotStyle.BAR:
        plt.bar(X_train, range(len(X_train)), edgecolor=Color.SPACE.value,
                color=Color.SEA.value)
    else:
        plt.scatter(X_train, range(len(X_train)), edgecolor=Color.SPACE.value,
                    color=Color.SEA.value)
    plt.grid(axis='x', linestyle='-', alpha=0.7)
    plt.grid(axis='y', linestyle='-', alpha=0.7)
    plt.title(f'Distribution of {X_train.name}')
    plt.xlabel(f'Index')
    plt.ylabel('Value')
    plt.show()

def plot_hist_feature_distributions(X_train):
    """Plot the distributions of values for each feature in X_train.

    Args:
        X_train (DataFrame): DataFrame containing the features.

    Returns:
        None
    """
    X_train.hist(figsize=(50, 50), bins=30, edgecolor=Color.SPACE.value,
                 color=Color.SEA.value)
    plt.suptitle('Distribution of features', y=0.1)
    plt.show()

def plot_hist_feature_distributions_0_1(X_train_0, X_train_1):
    """Plot the distributions of values for each feature
    in X_train_0 and X_train_1 overlaid.

    Args:
        X_train_0 (DataFrame): DataFrame containing the features for class 0.
        X_train_1 (DataFrame): DataFrame containing the features for class 1.

    Returns:
        None
    """
    plt.figure(figsize=(50, 50))
    for col in X_train_0.columns:
        
        plt.hist(X_train_0[col], bins=30, alpha=0.5, 
                 edgecolor=Color.SPACE.value, color=Color.SPACE.value, label='False Planet')
        
        plt.hist(X_train_1[col], bins=30, alpha=0.5, 
                 edgecolor=Color.SUN.value, color=Color.SUN.value, label='True Planet')
        plt.suptitle(f'Distribution of feature: {col}', y=0.92)
        plt.legend()
        plt.show()

def compute_train_0_1(X_train, y_train):
    """Split the training set into two subsets based on binary labels (0 and 1).

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.

    Returns:
        X_train_0 (pd.DataFrame): Subset of training features with label 0.
        X_train_1 (pd.DataFrame): Subset of training features with label 1.
    """
    X_train_0 = []
    X_train_1 = []
    for index, row in X_train.iterrows():
        if y_train.iloc[index] == 0:
            X_train_0.append(row)
        else:
            X_train_1.append(row)
    X_train_0 = pd.DataFrame(X_train_0, columns=X_train.columns)
    X_train_1 = pd.DataFrame(X_train_1, columns=X_train.columns)
    return X_train_0, X_train_1