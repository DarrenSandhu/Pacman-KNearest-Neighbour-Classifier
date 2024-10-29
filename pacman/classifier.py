# classifier.py
# Lin Li/26-dec-2021
#
# Use the skeleton below for the classifier and insert your code here.
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Implmenting a k-nearest neighbor classifier
class Classifier:
    
    # Initialise the classifier with the data and target, as well as the number of neighbors and the weights for the different features
    def __init__(self):
        self.data = []
        self.target = []
        self.k = None
        self.model = None

        # Weights for the different features
        self.ghost_weight = 8
        self.visible_ghost_weight = 16
        self.food_weight = 0.5
        self.walls_weight = 0.5


    # Function to reset the classifier
    def reset(self):
        self.data = []
        self.target = []
        self.model = None


    # Function to fit the classifier to the data and target
    def fit(self, data, target):
        self.adjust_weights_of_train_data(data) # Set the weights of the training data
        self.data = data

        self.target = target


    # Function to calculate the hamming distnace between two binary vectors
    def hamming_distance(self, binary_vector_1, binary_vector_2):
        return sum(bit1 != bit2 for bit1, bit2 in zip(binary_vector_1, binary_vector_2))
    

    # Function to calculate the euclidean distance between two vectors
    def euclidean_distance(self, x, y):
            x = np.array(x)
            y = np.array(y)
            return np.sqrt(np.sum((x - y) ** 2)) # This loops through the elements of the vectors and calculates the sum of the squares of the differences.
    

    # Function to get the number of ghosts which is calculated by subtracting 1 from the length of the data and then subtracting 8 from the result and dividing by 8.
    def get_number_of_ghosts(self, data):
        return int(((len(data) - 1) - 8) / 8)
    

    # Function to get the upper bound for the ghosts which is calculated by adding the number of ghosts multiplied by 8 to 8.
    def get_upper_bound_of_ghosts(self, data):
        return 8 + self.get_number_of_ghosts(data) * 8
    

    # Function to adjust the weights of the training data
    def adjust_weights_of_train_data(self, data):
        for i in range(len(data)):
            self.adjust_weights_of_data(data[i])


    # Function to adjust the weights of the test data using the weights specified in the constructor
    def adjust_weights_of_data(self, data):
        ghost_upper_bound = self.get_upper_bound_of_ghosts(data)
        
        # Adjust the weights of food data by looping through where the food features is and multiplying the value by the food weight
        for i in range(4, 8):
            if data[i] == 1:
                data[i] = data[i] * self.food_weight

        # Adjust the weights of the ghosts data by looping through where the ghosts features is and multiplying the value by the ghost weight
        for i in range(8, ghost_upper_bound):
            if data[i] == 1:
                data[i] = data[i] * self.ghost_weight
        
        # Adjust the weights of the visible ghosts data by multiplying the value by the visible ghost weight
        if (data[-1] == 1):
            data[-1] = data[-1] * self.visible_ghost_weight   


    # Function to use the k-nearest neighbors classifier to predict the best move
    def k_nearest_neighbors(self, data):
        print("Data:", data)
        self.adjust_weights_of_data(data) 
        print("Adjusted Data:", data)

        distances = np.array([self.euclidean_distance(data, x) for x in self.data]) # Calculate the distance between the data and the each instance in the training data and store the results in an array

        self.k = self.get_best_k() # Get the best k for the k-nearest neighbors classifier

        k_neighbors_indices = np.argsort(distances)[:self.k] # Get the indices of the k nearest neighbors
        print("Nearest neighbors indices:", k_neighbors_indices)

        k_neighbours_targets = [self.target[i] for i in k_neighbors_indices] # Get the targets of the k nearest neighbors ie the best moves
        print("Nearest neighbors targets:", k_neighbours_targets)

        counts = np.bincount(k_neighbours_targets) # Count the number of times each target occurs in the k nearest neighbors
        predicted_good_move = np.argmax(counts) # Get the target that occurs the most in the k nearest neighbors
        print("Predicted good move:", predicted_good_move)

        
        print()
        print()

        return predicted_good_move
    

    # Function to calculate the best k for the k-nearest neighbors classifier
    def get_best_k(self):
        best_k = 0
        best_accuracy = 0
        self.adjust_weights_of_train_data(self.data)
        for k in range(1, 11):
            self.k = k
            X_train, X_test, y_train, y_test = train_test_split(self.data, self.target, test_size=0.2, random_state=0)
            self.model = KNeighborsClassifier(n_neighbors=k)
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            accuracy = np.mean(y_pred == y_test)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_k = k
        return best_k


    # Function to predict the best move using the k-nearest neighbors classifier
    def predict(self, data, legal=None):
        return self.k_nearest_neighbors(data)
    


    ############################################################################################################
    ############################################################################################################
    # The code below is for the k-means clustering algorithm. It is not used in the final implementation of the classifier
    ## Code to implement the k-means clustering algorithm
    # def fit(self, data, target):
        #     self.data = data
        #     self.target = target
        #     converged = False
        #     average = [0] * len(self.data[1])

        #     k_clusters = [[] for _ in range(self.k)]
        #     for i in range(self.k):
        #         cluster = []
        #         for j in range(len(self.data[1])):
        #             cluster.append(np.random.randint(0, 2))
        #         k_clusters[i].append(cluster)

        #     for cluster in k_clusters:
        #         for data in cluster:
        #             print(data)
        #         print()

        #     while not converged:
        #         previous_average = average

        #         for i in range(len(self.data)):
        #             nearest = float('inf')
        #             cluster_index = 0
        #             for cluster in k_clusters:
        #                 distance = self.euclidean_distance(self.data[i], cluster[0])
        #                 if distance < nearest:
        #                     nearest = distance
        #                     cluster_index = k_clusters.index(cluster)
        #             k_clusters[cluster_index].append(self.data[i])

        #         for cluster in k_clusters:
        #             print()
        #             sum = [0] * len(cluster[0])

        #             for data in cluster:
        #                 for i in range(len(data)):
        #                     sum[i] += data[i]
        #             # print(sum)
        #             average = [x / len(cluster) for x in sum]
        #             # print(average)
                    
        #             cluster.clear()
        #             cluster.append(average)

        #         if (np.abs(np.array(average) - np.array(previous_average)) < self.error).all():
        #             converged = True

        #     # for cluster in k_clusters:
        #     #     for data in cluster:
        #     #         print(data)
        #     #     print()
            
        #     new_cluster = k_clusters
        #     for i in range(len(self.data)):
        #             nearest = float('inf')
        #             cluster_index = 0
        #             for cluster in k_clusters:
        #                 distance = self.euclidean_distance(self.data[i], cluster[0])
        #                 if distance < nearest:
        #                     nearest = distance
        #                     cluster_index = new_cluster.index(cluster)
        #             new_cluster[cluster_index].append(self.data[i])

        #     for cluster in new_cluster:
        #         for data in cluster:
        #             print(data)
        #         print()
        
        
