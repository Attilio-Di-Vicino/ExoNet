from scipy.spatial import distance

class DistanceMetrics:
    
    @staticmethod
    def euclidean_distance(x, y):
        """Compute the Euclidean distance between two 1-D arrays."""
        return distance.euclidean(x, y)

    @staticmethod
    def manhattan_distance(x, y):
        """Compute the City Block (Manhattan) distance."""
        return distance.cityblock(x, y)

    @staticmethod
    def cosine_distance(x, y):
        """Compute the Cosine distance between 1-D arrays."""
        return distance.cosine(x, y)

    @staticmethod
    def chebyshev_distance(x, y):
        """Compute the Chebyshev distance."""
        return distance.chebyshev(x, y)

    @staticmethod
    def minkowski_distance(x, y):
        """Compute the Minkowski distance between two 1-D arrays."""
        return distance.minkowski(x, y)