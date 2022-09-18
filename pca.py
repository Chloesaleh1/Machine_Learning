import numpy as np
from matplotlib import pyplot as plt

class PCA(object):

    def __init__(self):
        self.U = None
        self.S = None
        self.V = None

    def fit(self, X): # 5 points
        """
        Decompose dataset into principal components by finding the singular value decomposition of the centered dataset X
        You may use the numpy.linalg.svd function
        Don't return anything. You can directly set self.U, self.S and self.V declared in __init__ with
        corresponding values from PCA. See the docstrings below for the expected shapes of U, S, and V

        Args:
            X: (N,D) numpy array corresponding to a dataset

        Return:
            None

        Set:
            self.U: (N, min(N,D)) numpy array
            self.S: (min(N,D), ) numpy array
            self.V: (min(N,D), D) numpy array (no need to transpose what is returned from np.linalg.svd)
        """
        mu = np.mean(X, axis = 0, keepdims = True)
        X = X - mu
        U, S, V = np.linalg.svd(X, full_matrices = False)
        self.U = U
        self.S = S
        self.V = V



    def transform(self, data, K=2): # 2 pts
        """
        Transform data to reduce the number of features such that final data (X_new) has K features (columns)
        Utilize self.U, self.S and self.V that were set in fit() method.

        Args:
            data: (N,D) numpy array corresponding to a dataset
            K: int value for number of columns to be kept

        Return:
            X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data
        """
        mu = np.mean(data, axis = 0, keepdims = True)
        data = data - mu
        self.fit(data)
        X_new = np.dot(data, self.V.T[:, :K])
        return X_new




    def transform_rv(self, data, retained_variance=0.99): # 3 pts
        """
        Transform data to reduce the number of features such that the retained variance given by retained_variance is kept
        in X_new with K features
        Utilize self.U, self.S and self.V that were set in fit() method.

        Args:
            data: (N,D) numpy array corresponding to a dataset
            retained_variance: float value for amount of variance to be retained

        Return:
            X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data, where K is the number of columns
                   to be kept to ensure retained variance value is retained_variance
        """
        mu = np.mean(data, axis = 0, keepdims = True)
        data = data - mu
        self.fit(data)
        var = np.cumsum(self.S ** 2)
        var = var / var[-1]

        N = data.shape[0]

        for i in range(N):
            if var[i] > retained_variance:
                break

        X_new = np.dot(data, self.V.T[:, :i + 1])
        return X_new




    def get_V(self):
        """ Getter function for value of V """

        return self.V


    def visualize(self, X, y, fig=None):   # 5 pts
        """
        Use your PCA implementation to reduce the dataset to only 2 features. You'll need to run PCA on the dataset and then transform it so that the new dataset only has 2 features.
        Create a scatter plot of the reduced data set and differentiate points that have different true labels using color.
        Hint: To create the scatter plot, it might be easier to loop through the labels (Plot all points in class '0', and then class '1')
        Hint: To reproduce the scatter plot in the expected outputs, use the colors 'blue', 'magenta', and 'red' for classes '0', '1', '2' respectively.

        Args:
            xtrain: (N,D) numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: (N,) numpy array, the true labels

        Return: None
        """
        N = X.shape[0]

#do dimensionality reduction
        obj = PCA()
        x_new = obj.transform(X, 2)

#add y values as column to xnew
        y_new = y.reshape((N, 1))
        x = np.hstack((x_new, y_new))

#make scatter plot
        plot = plt.scatter(x[:, 0], x[:, 1], c = x[:, 2], marker = "x")




        ##################### END YOUR CODE ABOVE, DO NOT CHANGE BELOW #######################
        plt.legend()
        plt.show()
