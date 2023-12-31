from scipy.spatial import distance
import warnings
import MissingInputError

class DistanceMetrics:
    
    @staticmethod
    def euclidean_distance(x, y):
        """Compute the Euclidean distance between two 1-D arrays."""
        DistanceMetrics.__check_input__(x, y)
        DistanceMetrics.__check_len__(x, y)
        return distance.euclidean(x, y)

    @staticmethod
    def manhattan_distance(x, y):
        """Compute the City Block (Manhattan) distance."""
        DistanceMetrics.__check_input__(x, y)
        DistanceMetrics.__check_len__(x, y)
        return distance.cityblock(x, y)

    @staticmethod
    def cosine_distance(x, y):
        """Compute the Cosine distance between 1-D arrays."""
        DistanceMetrics.__check_input__(x, y)
        DistanceMetrics.__check_len__(x, y)
        return distance.cosine(x, y)

    @staticmethod
    def chebyshev_distance(x, y):
        """Compute the Chebyshev distance."""
        DistanceMetrics.__check_input__(x, y)
        DistanceMetrics.__check_len__(x, y)
        return distance.chebyshev(x, y)

    @staticmethod
    def minkowski_distance(x, y):
        """Compute the Minkowski distance between two 1-D arrays."""
        DistanceMetrics.__check_input__(x, y)
        DistanceMetrics.__check_len__(x, y)
        return distance.minkowski(x, y)
    
    @staticmethod
    def __check_input__(x, y):
        """Check if input values x and y are not None.

        Args:
            x: First input value.
            y: Second input value.

        Raises:
            MissingInputError (raise): If either x or y is None.
        """
        if x in None or y is None:
            raise MissingInputError("x or y cannot be None.")
    
    @staticmethod
    def __check_len__(x, y):
        """Check if the lengths of arrays x and y are equal.

        Args:
            x: First array.
            y: Second array.

        Warns:
            UserWarning: If the lengths of x and y are not equal.
        """
        if len(x) != len(y):
            warnings.warn('Array sizes are not equal'
                          ' for distance calculation.', UserWarning)