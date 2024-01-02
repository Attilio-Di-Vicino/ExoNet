import numpy as np
import pandas as pd
import time
import warnings
from joblib import Parallel, delayed
from collections import Counter
from numpy.ma.core import ceil
from utils.ditance_metrics import DistanceMetrics
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

class SimpleSOM:
    def __init__(self, X_train, y_train, W=None, max_iter=int(7.5*10e3), 
                 num_rows=10, num_cols=10, max_learning_rate=0.5, max_distance=None, 
                 random_seed=42, distance_bmu=DistanceMetrics.euclidean_distance,
                 distance_nr=DistanceMetrics.manhattan_distance, 
                 W_PLANET=1, W_FALSE_PLANET=1, K=None, n_jobs=1):
        """Initialization of the SimpleSOM class.

        Args:
            X_train (numpy.ndarray): Array of training features.
            y_train (numpy.ndarray): Array of training labels.
            W (numpy.ndarray): Initial weights for the SOM map.
            max_iter (int): Maximum number of iterations.
            num_rows (int): Number of rows in the SOM map.
            num_cols (int): Number of columns in the SOM map.
            max_learning_rate (float): Maximum learning rate.
            max_distance (int): Maximum distance for neighborhood calculation.
            random_seed (int): Random seed for reproducibility.
            distance_bmu (function): Distance function for BMU calculation.
            distance_nr (function): Distance function for neighborhood calculation.
            W_PLANET (int): Weight assigned to the 'True Planet' class during initialization.
            W_FALSE_PLANET (int): Weight assigned to the 'False Planet' class during initialization.
            K (int): Number of neighbors to consider during K-NN prediction.
            n_jobs (int): Number of parallel jobs for training SOM models.
                        Default is 1.
        """
        if len(X_train) != len(y_train):
            raise ValueError("The lengths of X_train and labels must be equal.")
        if max_iter < (X_train.shape[0] * 0.7):
            warnings.warn(
                (
                    f"Warning: max_iter ({max_iter}) is less than the"
                    f" number of training examples ({X_train.shape[0]})"
                    " the SOM could be trained with only a few observations."
                ),
                UserWarning,
            )

        self.X_train = X_train
        self.y_train = y_train
        self.num_features = X_train.shape[1]
        self.max_iter = max_iter
        self.num_rows = num_rows 
        self.num_cols = num_cols
        self.max_learning_rate = max_learning_rate
        self.max_distance = max(self.num_rows, self.num_cols) \
            if max_distance is None else max_distance
        self.random_seed = random_seed
        self.distance_bmu = distance_bmu
        self.distance_nr = distance_nr
        self.W_PLANET = W_PLANET
        self.W_FALSE_PLANET = W_FALSE_PLANET
        self.K = K
        self.n_jobs = n_jobs
        self.__QE = None
        self.__maps = None
        self.__label_map = None
        self.__label_map_kernel_weighed = None
        self.__label_map_kernel_occurrences_weighed = None
        self.__BMUs = None
        self.__BMU_labels = None
        self.__BMU_label_combinations = None
        self.__coord_label = None
        self.W = W if W is not None else self._w_som_initialization()

    # Getter
    def get_qe(self):
        return self.__QE
    
    def get_maps(self):
        return self.__maps
        
    def get_label_map(self):
        return self.__label_map
    
    def get_label_map_kw(self):
        return self.__label_map_kernel_weighed

    def get_label_map_kcw(self):
        return self.__label_map_kernel_occurrences_weighed

    def get_bmus(self):
        return self.__BMUs

    def get_bmu_label(self):
        return self.__BMU_labels
    
    def get_bmu_label_combinations(self):
        return self.__BMU_label_combinations
    
    def get_coord_label(self):
        return self.__coord_label

    def _w_som_initialization(self):
        """Initialize the weights of the SOM map.
        If weights are not provided as input, they are chosen randomly, 
        taking 50% of the total neurons for each class.
        
        Args:
            None.

        Returns:
            W (numpy.ndarray): initial weights of map.
        """
        np.random.seed(self.random_seed)
        tot_elem = (self.num_rows * self.num_cols)
        N = tot_elem // 2 # About 50%
        M = tot_elem - N  # About 50%
        index_class_0 = np.random.choice(np.where(self.y_train == 0)[0], N, replace=False)
        index_class_1 = np.random.choice(np.where(self.y_train == 1)[0], M, replace=False)
        final_index = np.concatenate([index_class_0, index_class_1])
        return self.X_train[final_index].reshape(self.num_rows,
                                                 self.num_cols, self.num_features)

    def winning_neuron(self, data):
        """Calculate the neuron that has the shortest distance (e.g. euclidean ecc.).
        
        Args:
            data (numpy.ndarray): Single observation.

        Returns:
            shortest_distance (float): distance between data end shortest neuron.
            winner (array): coordinates of best maching unit (BMU).
        """
        winner = [0,0]
        shortest_distance = np.sqrt(self.num_features) # initialise with max distance
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                distance = self.distance_bmu(self.W[row][col], data)
                if distance < shortest_distance: 
                    shortest_distance = distance
                    winner = [row,col]
        return shortest_distance, winner

    def decay(self, step, max_steps, max_learning_rate, max_distance):
        """Decay of learning rate and neighborhood radius.
        
        Args:
            step (int): iteration.
            max_steps (int): max number of iteration.
            max_learing_rate (float): current learning rate.
            max_distance (int): current neighborhood radius.

        Returns:
            learning_rate (float): new learning rate.
            neighbourhood_range (int): new neighbourhood radius.
        """
        coefficient = 1.0 - (np.float64(step) / max_steps)
        learning_rate = coefficient * max_learning_rate
        neighbourhood_range = ceil(coefficient * max_distance)
        return learning_rate, neighbourhood_range
    
    def __help_fit(self, title, step, qe, lr, nr, time):
        """Helper fit method."""
        print(f"{title}: ", "{:05d}".format(step),
                " | QE: ", "{:10f}".format(qe),
                " | LR: ", "{:10f}".format(lr),
                " | NR: ", "{:10f}".format(nr),
                " | Time: ", "{:10f}".format(time))

    def fit(self, X_data, verbose=0):
        """Map training method.
        
        Args:
            X_data (numpy.ndarray): observation.
            verbode (int): default = 0, if it's > 1 print
                            the progress each 100 observation.

        Returns:
            params (dict): dict that contains:
                QE (int): final quantization error.
                errors (numpy.ndarray): all quantization errors.
                lr (numpy.ndarray): all learing rate.
                nr (numpy.ndarray): all neighbourhood radius
        """
        params = {
            'QE': 0,
            'errors': np.zeros(self.max_iter),
            'lr': np.zeros(self.max_iter),
            'nr': np.zeros(self.max_iter)
        }
        start_time = time.time()
        for epoch in range(self.max_iter):
            t1 = time.time()
            params['QE'] = 0
            learning_rate, neighbourhood_range = self.decay(epoch, 
                                                            self.max_iter,
                                                            self.max_learning_rate,
                                                            self.max_distance)
            params['lr'][epoch] = learning_rate
            params['nr'][epoch] = neighbourhood_range
            t = np.random.randint(0, high=X_data.shape[0]) # random index of traing data
            shortest_distance, winner = self.winning_neuron(X_data[t])
            params['QE'] += shortest_distance
            params['QE'] /= X_data.shape[0]
            params['errors'][epoch] = params['QE']
            for row in range(self.num_rows):
                for col in range(self.num_cols):
                    if self.distance_nr([row,col],winner) <= neighbourhood_range:
                        self.W[row][col] += learning_rate*(X_data[t]-self.W[row][col])
            t1 = time.time() - t1
            if ((epoch + 1) % 1000 == 0 or epoch == 0) and verbose > 1:
                end_time = t1 if epoch == 0 else t1 * 1000
                self.__help_fit("Iteration", epoch+1, params['QE'],
                                params['lr'][epoch], params['nr'][epoch], end_time)
        end_time = time.time() - start_time
        self.__QE = params['QE']
        self.__help_fit("SOM training completed", epoch+1, params['QE'],
                        params['lr'][epoch], params['nr'][epoch], end_time)
        return params
    
    def collecting_labels(self, X_train, y_train):
        """Collect all the labels associated with the neuron.
        
        Args:
            X_train (numpy.ndarray): observation.
            y_train (numpy.ndarray): labels.

        Returns:
            maps (numpy.ndarray): array of label associated with the neuron.
        """
        label_data = y_train
        if self.__maps is None:
            self.__maps = np.empty(shape=(self.num_rows, self.num_cols), dtype=object)
            for row in range(self.num_rows):
                for col in range(self.num_cols):
                    self.__maps[row][col] = [] # empty list to store the label
            for t in range(X_train.shape[0]):
                if self.__BMUs is None:
                    _, BMU = self.winning_neuron(X_train[t])
                else:
                    BMU = self.__BMUs[t]
                # label of winning neuron
                self.__maps[BMU[0]][BMU[1]].append(label_data[t])
        return self.__maps
    
    def construct_label_map(self, X_train, y_train):
        """Construction of the label map which will have the same dimensions as the SOM.
        The construction of this label map is done simply by considering the collection
        of labels that each neuron has, and each neuron takes on the label
        of the majority class.
        
        Args:
            X_train (numpy.ndarray): observation.
            y_train (numpy.ndarray): labels.

        Returns:
            label_map (numpy.ndarray): label map of size (num_rows, num_cols, num_features).
        """
        if self.__label_map is None:
            self.__label_map = np.zeros(shape=(self.num_rows, 
                                        self.num_cols),dtype=np.int64)
            maps = self.collecting_labels(X_train=X_train, y_train=y_train)
            for row in range(self.num_rows):
                for col in range(self.num_cols):
                    label_list = maps[row][col]
                    if len(label_list)==0:
                        label = 0
                    else:
                        label = max(label_list, key=label_list.count)
                    self.__label_map[row, col] = label
        return self.__label_map
    
    def construct_label_map_weighed(self, X_train, y_train):
        """Construction of the label map which will have the same dimensions as the SOM.
        The construction of this label map is done not only on the basis of the collection
        of labels for each neuron, but also on its surroundings,
        and with a weight for each class, assigned during the initialization
        phase of the SOM, in case you want to give a different weight to the classes.
        
        Args:
            X_train (numpy.ndarray): observation.
            y_train (numpy.ndarray): labels.

        Returns:
            label_map (numpy.ndarray): label map of size (num_rows, num_cols, num_features).
        """
        if self.__label_map_kernel_weighed is not None:
            return self.__label_map_kernel_weighed
        if self.__label_map is None:
            self.__label_map = self.construct_label_map(X_train=X_train, y_train=y_train)
        rows, cols = self.__label_map.shape
        new_label_map_w = np.zeros((rows, cols), dtype=int)
        KERNEL_SIZE = 3
        for row in range(rows):
            for col in range(cols):
                count_1 = 0
                count_0 = 0
                for y in range(KERNEL_SIZE):
                    for x in range(KERNEL_SIZE):
                        new_y = row + (y - 1) # Kernel 3x3 - range(-1,2)
                        new_x = col + (x - 1)
                        if 0 <= new_x < cols and 0 <= new_y < rows:
                            if self.__label_map[new_y][new_x] == 1:
                                count_1 += 1 * self.W_PLANET
                            else:
                                count_0 += 1 * self.W_FALSE_PLANET
                if count_1 > count_0:
                    new_label_map_w[row][col] = 1
                else:
                    new_label_map_w[row][col] = 0
        self.__label_map_kernel_weighed = new_label_map_w
        return new_label_map_w
    
    def construct_label_map_occ_weighed(self, X_train, y_train):
        """Construction of the label map which will have the same dimensions as the SOM.
        The construction of this label map is done not only on the basis of the collection
        of labels for each neuron, but also on its surroundings, on a weight for each class,
        assigned during the initialization phase of the SOM, in case one wants to give a
        different weight to the classes, and finally also based on the occurrences of each BMU,
        this is because a neuron should have a different weight if it is
        activated several times.
        
        Args:
            X_train (numpy.ndarray): observation.
            y_train (numpy.ndarray): labels.

        Returns:
            label_map (numpy.ndarray): label map of size (num_rows, num_cols, num_features).
        """
        BMU_counts = self.bmu_occurrences()
        if self.__label_map_kernel_occurrences_weighed is not None:
            return self.__label_map_kernel_occurrences_weighed
        if self.__label_map is None:
            self.__label_map = self.construct_label_map(X_train=X_train, y_train=y_train)
        rows, cols = self.__label_map.shape
        new_label_map_w = np.zeros((rows, cols), dtype=int)
        KERNEL_SIZE = 3
        for row in range(rows):
            for col in range(cols):
                count_1 = 0
                count_0 = 0
                for y in range(KERNEL_SIZE):
                    for x in range(KERNEL_SIZE):
                        new_y = row + (y - 1) # Kernel 3x3 - range(-1,2)
                        new_x = col + (x - 1)
                        if 0 <= new_x < cols and 0 <= new_y < rows:
                            if self.__label_map[new_y][new_x] == 1:
                                count_1 += BMU_counts[(new_y,new_x)] * self.W_PLANET
                            else:
                                count_0 += BMU_counts[(new_y,new_x)] * self.W_FALSE_PLANET
                if count_1 > count_0:
                    new_label_map_w[row][col] = 1
                else:
                    new_label_map_w[row][col] = 0
        self.__label_map_kernel_occurrences_weighed = new_label_map_w
        return new_label_map_w
    
    def _compute_labels(self, X_test, label_map, K):
        """Calculate the labels of the new data, placing the i-esima observation on the SOM,
        and considering a neighborhood of the K closest neighbors, for which a statistic
        is carried out, and therefore the majority class will be the predicted label.
        
        Args:
            X_test (numpy.ndarray): observation.
            label_map (numpy.ndarray): label map of size (num_rows, num_cols, num_features).

        Returns:
            winner_labels (numpy.ndarray): predicted labels.
        """
        winner_labels = []
        flattened_som_W = self.W.reshape(-1, self.W.shape[-1])
        nn_model = NearestNeighbors(n_neighbors=K, n_jobs=self.n_jobs)
        nn_model.fit(flattened_som_W)

        for t in range(X_test.shape[0]):
            _, winner = self.winning_neuron(X_test[t])
            row = winner[0]
            col = winner[1]
        
            _, indices = nn_model.kneighbors(self.W[row][col].reshape(1, -1))
            label_neighbors = []
            for k in range(K):
                index_rows = indices[0][k] // self.num_rows
                index_cols = indices[0][k] % self.num_cols
                label_neighbors.append(label_map[index_rows][index_cols])
            predicted = max(label_neighbors, key=label_neighbors.count)
            winner_labels.append(predicted)
        return np.array(winner_labels)
    
    def predict(self, X_train, y_train, X_test, K):
        """Calculate the labels of the new data, using the label map based on
        the collection of labels of each neuron.
        
        Args:
            X_train (numpy.ndarray): train observation.
            y_train (numpy.ndarray): train labels.
            X_test (numpy.ndarray): new observation.
            K (int): neighborhood of the K nearest neighbors, preferably odd.

        Returns:
            winner_labels (numpy.ndarray): predicted labels.
        """
        if K is None:
            K = self.K
        label_map = self.construct_label_map(X_train=X_train, y_train=y_train)
        winner_labels = self._compute_labels(X_test=X_test, label_map=label_map, K=K)
        return np.array(winner_labels)
    
    def predict_kw(self, X_train, y_train, X_test, K):
        """Calculate the labels of the new data, using the label map based on
        the collection of labels of each neuron, an around and a weight to the classes.
        
        Args:
            X_train (numpy.ndarray): train observation.
            y_train (numpy.ndarray): train labels.
            X_test (numpy.ndarray): new observation.
            K (int): neighborhood of the K nearest neighbors, preferably odd.

        Returns:
            winner_labels (numpy.ndarray): predicted labels.
        """
        if K is None:
            K = self.K
        label_map = self.construct_label_map_weighed(X_train, y_train)
        winner_labels = self._compute_labels(X_test=X_test, label_map=label_map, K=K)
        return np.array(winner_labels)
    
    def predict_kcw(self, X_train, y_train, X_test, K):
        """Calculate the labels of the new data, using the label map based on
        the collection of labels of each neuron, a surrounding, a weight to the classes,
        and the occurrences of the BMUs.
        
        Args:
            X_train (numpy.ndarray): train observation.
            y_train (numpy.ndarray): train labels.
            X_test (numpy.ndarray): new observation.
            K (int): neighborhood of the K nearest neighbors, preferably odd.

        Returns:
            winner_labels (numpy.ndarray): predicted labels.
        """
        if K is None:
            K = self.K
        label_map = self.construct_label_map_occ_weighed(X_train, y_train)
        winner_labels = self._compute_labels(X_test=X_test, label_map=label_map, K=K)
        return np.array(winner_labels)
    
    def predict_knn(self, X_test, K):
        """Calculate the labels of the new data,
        using the KNeighborsClassifier.
        
        Args:
            X_test (numpy.ndarray): new observation.
            K (int): neighborhood of the K nearest neighbors, preferably odd.

        Returns:
            winner_labels (numpy.ndarray): predicted labels.
        """
        if self.W is None:
            raise ValueError("Error, do the fit first.")
        if K is None:
            K = self.K
        if self.__label_map_kernel_weighed is None:
            self.__label_map_kernel_weighed = self.construct_label_map_weighed(self.X_train,
                                                                            self.y_train)
        flattened_som_W = self.W.reshape(-1, self.W.shape[-1])
        label = self.__label_map_kernel_weighed.reshape(-1,)
        knn_classifier = KNeighborsClassifier(n_neighbors=K)
        knn_classifier.fit(flattened_som_W, label)
        return knn_classifier.predict(X_test)
    
    def __parallel_bmu_and_labels__(self, n):
        """Compute the BMU neuron."""
        _, BMU = self.winning_neuron(self.X_train[n])
        return BMU, self.y_train[n]

    def calculate_bmu_and_labels(self):
        """Calculate the BMU for each observation and save the label.

        Args:
            None.

        Returns:
            BMUs (list): list of BMU for each observation.
            BMU_labels (list): list of labels for each BMU.
        """
        if len(self.X_train) != len(self.y_train):
            raise ValueError("The lengths of X_train and labels must be equal.")
        if self.__BMUs is None or self.__BMU_labels is None:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self.__parallel_bmu_and_labels__)(n) for n in range(len(self.X_train))
            )

            BMUs, BMU_labels = zip(*results)

            self.__BMUs = list(BMUs)
            self.__BMU_labels = list(BMU_labels)
            self.__BMU_label_combinations = list(zip(self.__BMUs, self.__BMU_labels))

        return self.__BMUs, self.__BMU_labels

    def bmu_occurrences(self):
        """Calculate the occurrences for each neuron.

        Args:
            None.

        Returns:
            counts (collections.Counter): list of coordinate with occurrences.
        """
        if self.__BMUs is None:
            self.__BMUs, self.__BMU_labels = self.calculate_bmu_and_labels()
        flattened_BMUs = [tuple(BMU) for BMU in self.__BMUs]
        counts = Counter(flattened_BMUs)
        return counts

    def compute_coordinates_label(self):
        """Calculate all label for each neuron.

        Args:
            None.

        Returns:
            counts (dict): dict of coordinates with labels.
        """
        if self.__coord_label is None:
            if self.__BMU_label_combinations is None:
                _, _, = self.calculate_bmu_and_labels()
            coord_label = {}
            for index in range(len(self.__BMU_label_combinations)):
                x = self.__BMU_label_combinations[index][0][0]
                y = self.__BMU_label_combinations[index][0][1]
                label = []
                for i in range(len(self.__BMU_label_combinations)):
                    if self.__BMU_label_combinations[i][0][0] == x \
                        and self.__BMU_label_combinations[i][0][1] == y:
                        label.append(self.__BMU_label_combinations[index][1][0])
                coord_label[(x, y)] = label
            self.__coord_label = coord_label
        return coord_label

    def print_bmu_label(self):
        """Print coordinates and all labels associated.

        Args:
            None.

        Returns:
            None.
        """
        if self.__coord_label is None:
            self.__coord_label = self.compute_coordinates_label()
        for key, values in self.__coord_label.items():
            value_counts = pd.Series(values).value_counts()
            count_1 = value_counts.get(1, 0)
            count_0 = value_counts.get(0, 0)
            coo = str(key)
            print(f'{coo.ljust(8)}-> Planet: {count_1:3}, False Planet: {count_0:3}')

    def compute_occurrences_plot(self):
        """Compute occurrences for plot.

        Args:
            None.

        Returns:
            None.
        """
        coords, counts = zip(*self.bmu_occurrences().items())
        x_coords, y_coords = zip(*coords)
        counts_occ = np.array(counts)
        return x_coords, y_coords, counts_occ