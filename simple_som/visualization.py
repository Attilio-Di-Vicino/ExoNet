import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import colors
from itertools import product
from sklearn.preprocessing import MinMaxScaler
from matplotlib.lines import Line2D
from utils.color import Color


class View(object):
    def __init__(self, width, height, title, show_axis=True, packed=True,
                 text_size=2.8, show_text=True, col_size=6, *args, **kwargs):
        self.width = width
        self.height = height
        self.title = title
        self.show_axis = show_axis
        self.packed = packed
        self.text_size = text_size
        self.show_text = show_text
        self.col_size = col_size

    def prepare(self, *args, **kwargs):
        raise NotImplementedError()

    def save(self, filename):
        raise NotImplementedError()

    def show(self, *args, **kwrags):
        raise NotImplementedError()
    
    # CALCULATION OF RECTANGULAR U-MATRIX FROM W
    @staticmethod
    def make_u_rect(som):

        U = np.zeros([som.W.shape[0]*2-1, som.W.shape[1]*2-1], dtype=np.float64)

        # YELLOW CELLS
        for i in range(som.W.shape[0]): # across columns
            k=1
            for j in range(som.W.shape[1]-1):
                U[2*i, k]= np.linalg.norm(som.W[i,j]-som.W[i,j+1], ord=2)
                k += 2

        for j in range(som.W.shape[1]): # down rows
            k=1
            for i in range(som.W.shape[0]-1):
                U[k,2*j] = np.linalg.norm(som.W[i,j]-som.W[i+1,j], ord=2)
                k+=2

        # ORANGE AND BLUE CELLS - average of cells top, bottom, left, right.
        for (i,j) in product(range(U.shape[0]), range(U.shape[1])):
            if U[i,j] !=0: continue
            all_vals = np.concatenate((
                U[(i-1 if i>0 else i): (i+2 if i<=U.shape[0]-1 else i), j],
                U[i, (j-1 if j>0 else j): (j+2 if j<=U.shape[1]-1 else j)]))
            U[i,j] = all_vals[all_vals!=0].mean()
        
        # Normalizing in [0-1] range for better visualization.
        scaler = MinMaxScaler()
        return scaler.fit_transform(U)
    
    def collapse_hex_u(som, U):

        new_U = np.zeros([int((U.shape[0]+1)/2), U.shape[1]])
        # Moving up values in every alternate column
        for j in range(1, U.shape[1], 2):
                for i in range(U.shape[0]-1):
                    U[i,j]=U[i+1,j]

        # Removing extra rows
        for i in range(new_U.shape[0]): new_U[i,:] = U[2*i,:]

        return new_U
    
    # CALCULATION OF HEXAGONAL U-MATRIX FROM W
    @staticmethod
    def make_u_hex(som): 
    
        # Creating arrays with extra rows to accommodate hexagonal shape.
        U_temp =  np.zeros([4*som.W.shape[0]-1, 2*som.W.shape[1]-1, som.W.shape[2]])
        U = np.zeros([4*som.W.shape[0]-1, 2*som.W.shape[1]-1])
        """
        The U matrix is mapped to a numpy array as shown below.
        U_temp holds neuron at postion 1 in place of {1} for easy computation
        of {1,2}, {2,3} etc. in U. {1}, {2}, {3}, ... are computed later.
        [
        [    (1),        0,       0,       0,    (3)],
        [      0,    (1,2),       0,   (2,3),      0],
        [   (1,4),       0,     (2),       0,  (3,6)],
        [      0,    (2,4),       0 ,  (2,6),      0],
        [    (4),        0,    (2,5),      0,    (6)],
        [      0,    (4,5),        0,  (5,6),      0],
        [   (4,7),       0,      (5),      0,  (6,9)],
        [      0,    (2,4),        0,  (5,9),      0],
        [    (7),        0,    (5,8),      0,    (9)],
        [      0,    (7,8),        0,  (8,9),      0],
        [      0,        0,      (8),      0,      0]
        ]
        """

        # Creating a temporary array placing neuron vectors in 
        # place of orange cells.
        k=0
        indices = []
        for i in range(som.W.shape[0]):
            l=0
            for j in range(som.W.shape[1]):
                U_temp[k+2 if l%4!=0 else k,l,:] = som.W[i,j,:]
                indices.append((k+2 if l%4!=0 else k,l))
                l+=2
            k += 4
        
        # Finding distances for YELLOW cells.
        for (i,j),(k,l) in product(indices, indices):
            if abs(i-k)==2 and abs(j-l)==2:      # Along diagonals
                U[int((i+k)/2), int((j+l)/2)] = np.linalg.norm(U_temp[i,j,:]-U_temp[k,l,:], ord=2)
            
            if abs(i-k)==4 and abs(j-l)==0:       # In vertical direction
                U[int((i+k)/2), int((j+l)/2)] = np.linalg.norm(U_temp[i,j,:]-U_temp[k,l,:], ord=2)
        
        # Calculating ORANGE cells as mean of immediate surrounding cells.
        for (i,j) in indices:
            all_vals = U[(i-2 if i-1>0 else i): (i+3 if i+2<U.shape[0]-1 else i), 
                (j-1 if j>0 else j): (j+2 if j<=U.shape[1]-1 else j)]
            U[i,j] = np.average(all_vals[all_vals!=0])
        
        # To remove extra rows introduced in above function.
        new_U = View.collapse_hex_u(som, U)

        # Normalizing in [0-1] range for better visualization.
        scaler = MinMaxScaler()
        return scaler.fit_transform(new_U)
    
    def draw_hex_grid(som, ax, U):

        def make_hex(ax, x, y, colour):
            if x%2==1: y=y+0.5
            xs = np.array([x-0.333,x+0.333,x+0.667,x+0.333,x-0.333,x-0.667])
            ys = np.array([  y+0.5,  y+0.5,      y,  y-0.5,  y-0.5,      y])
            ax.fill(xs, ys, facecolor = colour)
        
        ax.invert_yaxis()
        cmap = matplotlib.cm.get_cmap('jet')

        for (i,j) in product(range(U.shape[0]), range(U.shape[1])):
            if U[i,j]==0: rgba='white'
            else: rgba = cmap(U[i,j])
            make_hex(ax, i, j, rgba)
            
        ax.set_title("U-Matrix (Hexagonal)")

    @staticmethod
    def make_u_slide(component_names, som):
        num_cols = 4
        num_components = len(component_names)
        num_rows = (num_components // num_cols) + 1
        for index, name in enumerate(component_names, start=1):
            w = som.W[:, :, index - 1] 
            plt.subplot(num_rows, num_cols, index)
            plt.imshow(w)
            plt.title(f'{name}')
            plt.colorbar()
        plt.tight_layout()

    @staticmethod
    def plot_bmu_occurrences_with_planet_noplanet(ax, coord_label):
        x_coordinates, y_coordinates, counts_planet, counts_false_planet = [], [], [], []
        for key, values in coord_label.items():
            x, y = key
            x_coordinates.append(x)
            y_coordinates.append(y)
            value_counts = pd.Series(values).value_counts()
            count_1 = value_counts.get(1, 0)
            count_0 = value_counts.get(0, 0)
            counts_planet.append(count_1)
            counts_false_planet.append(count_0)

        colors = np.array(['green' if count_planet > count_false_planet \
                        else 'blue' for count_planet, count_false_planet \
                            in zip(counts_planet, counts_false_planet)])

        plt.scatter(x_coordinates, y_coordinates, c=colors, 
                    s=np.maximum(counts_planet, counts_false_planet), alpha=0.5)
        plt.title('BMU Occurrences - True Planet vs. False Planet')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        legend_labels = ["True Planet", "False Planet"]
        legend_elements = [Line2D([0], [0], marker='o', color='w', label=label,
                                markerfacecolor=color,
                                markersize=6) for color, label in zip(['green', 'blue'],
                                                                        legend_labels)]
        ax.legend(handles=legend_elements, loc="upper right")
        plt.gca().invert_yaxis()

    @staticmethod
    def plot_dispersion_observation_on_som(X_train, BMUs, BMU_labels,
                                       MIN_OFFSET=-0.5, MAX_OFFSET=0.5, verbose=0):
        if X_train is None or BMUs is None or BMU_labels is None:
                raise ValueError("X_train, BMUs and BMU_labels they cannot be None.")
        _, ax = plt.subplots(figsize=(10, 8), dpi=80)
        occupied_coordinates = set()
        # Calculate the coordinates of the single observation
        for i in range(len(X_train)):
            if i % 1000 == 0 and verbose > 1:
                print(f'Observation number {i} out of {len(X_train)}')
            BMU = tuple(BMUs[i])  # Convert the list to a tuple
            while BMU in occupied_coordinates:
                new_x = BMU[0] + np.random.uniform(MIN_OFFSET, MAX_OFFSET)
                new_y = BMU[1] + np.random.uniform(MIN_OFFSET, MAX_OFFSET)
                BMU = (new_x, new_y)
            # Update coordinates as busy
            occupied_coordinates.add(BMU)
            # Set the color based on the label
            color = 'green' if BMU_labels[i] == 1 else 'blue'
            _ = ax.scatter(*BMU, c=color, s=8)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, color='gray')
        ax.set_title('Dispersion of True Planet vs. False Planet on the SOM')
        legend_labels = ["True Planet", "False Planet"]
        legend_elements = [Line2D([0], [0], marker='o', color='w', label=label,
                                markerfacecolor=color,
                                markersize=8) for color, label in zip(['green', 'blue'], legend_labels)]
        ax.legend(handles=legend_elements, loc="upper right")
        plt.gca().invert_yaxis()
        plt.show()

    @staticmethod
    def plot_class_dispersion_observation_on_som(X_train, BMUs, BMU_labels,
                                        MIN_OFFSET=-0.5, MAX_OFFSET=0.5, verbose=0, classes=1):
        if X_train is None or BMUs is None or BMU_labels is None:
            raise ValueError("X_train, BMUs and BMU_labels cannot be None.")
        _, ax = plt.subplots(figsize=(10, 8), dpi=80)
        occupied_coordinates = set()
        # Calculate the coordinates of the single observation
        for i in range(len(X_train)):
            if i % 1000 == 0 and verbose > 1:
                print(f'Observation number {i} out of {len(X_train)}')
            # Check if the observation belongs to class 1 or 0
            if BMU_labels[i] == classes:
                BMU = tuple(BMUs[i])  # Convert the list to a tuple
                while BMU in occupied_coordinates:
                    new_x = BMU[0] + np.random.uniform(MIN_OFFSET, MAX_OFFSET)
                    new_y = BMU[1] + np.random.uniform(MIN_OFFSET, MAX_OFFSET)
                    BMU = (new_x, new_y)
                # Update coordinates as busy
                occupied_coordinates.add(BMU)
                # Set the color based on the label
                color = 'green' if BMU_labels[i] == 1 else 'blue'
                _ = ax.scatter(*BMU, c=color, s=8)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, color='gray')
        ax.set_title('Dispersion of True Planet on the SOM')
        if classes:
            legend_labels = ["True Planet"]
            c = ['green']
        else:
            legend_labels = ["False Planet"]
            c = ['blue']
        legend_elements = [Line2D([0], [0], marker='o', color='w', label=label,
                                markerfacecolor=color,
                                markersize=8) for color, label in zip(c, legend_labels)]
        ax.legend(handles=legend_elements, loc="upper right")
        plt.gca().invert_yaxis()
        plt.show()

    @staticmethod
    def plot_label_map(label_map, title, figsize=(8, 8)):
        cmap = colors.ListedColormap([Color.SEA.value, Color.EARTH.value])
        fig = plt.figure(figsize=figsize)
        plt.imshow(label_map, cmap=cmap)
        plt.title(title)
        legend_labels = ["True Planet", "False Planet"]
        legend_elements = [Line2D([0], [0], marker='o', color='w', label=label,
                                markerfacecolor=color,
                                markersize=8) for color, label \
                                    in zip([Color.EARTH.value, 
                                            Color.SEA.value], legend_labels)]
        fig.legend(handles=legend_elements, loc="upper right")
        plt.show()