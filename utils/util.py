import seaborn as sns
import matplotlib.pyplot as plt

def print_count_nan(data, name='dataset'):
    """Count not a number values, total elements,
    and the percentage of NaN in the data.

    Args:
        data (DataFrame): The dataset.
        name (str): The name of the dataset.

    Returns:
        Prints the calculated values.
    """
    nan_count = data.isna().sum()
    total_elements = data.size
    percentage_nan = (nan_count.sum() / total_elements) * 100
    print(f'Number of not a numbers values in {name}: {nan_count.sum()}'
          f' out of {total_elements}: {percentage_nan:.2f}%')

def print_feature_importance(X_train, feature_importances_):
    """Print feature importances with formatting.

    Args:
        feature_importances_ (list): List of feature importances.

    Returns:
        None.
    """
    feature_importances = {}
    for i in range(len(X_train.columns)):
        feature_importances[X_train.columns[i]] = feature_importances_[i]
    feature_importances = dict(sorted(feature_importances.items(),
                                      key=lambda x: x[1], reverse=True))
    index = 1
    for key, value in feature_importances.items():     
        # Create padding for index
        space = ' ' if index < 10 else ''
        # Print the formatted line
        print(f"{index}:{space}{key.ljust(30, '-')}> {value}")
        index += 1

def plot_confusion_matrix(cm):
    """Plot Confusion Matrix computed.

    Args:
        cm (array): np Array of values.

    Returns:
        None.
    """
    class_labels = ['False Planet', 'True Planet']
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()