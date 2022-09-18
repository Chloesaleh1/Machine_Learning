import numpy as np

class Regression(object):

    def __init__(self):
        pass

    def rmse(self, pred, label): # [5pts]
        """
        Calculate the root mean square error.

        Args:
            pred: (N, 1) numpy array, the predicted labels
            label: (N, 1) numpy array, the ground truth labels
        Return:
            A float value
        """
        N = pred.shape[0]
        rmse = ((pred - label)**2).sum() / N
        return np.sqrt(rmse)

    def construct_polynomial_feats(self, x, degree): # [5pts]
        """
        Args:
            x: N x D numpy array, where N is number of instances and D is the
               dimensionality of each instance.
            degree: the max polynomial degree
        Return:
            feat:
                For 1-D array, numpy array of shape Nx(degree+1), remember to include
                the bias term. feat is in the format of:
                [[1.0, x1, x1^2, x1^3, ....,],
                 [1.0, x2, x2^2, x2^3, ....,],
                 ......
                ]

                For D-dimensional array: numpy array of shape N x (degree+1) x D, remember to include
                  the bias term.

                Example: print(feat)
                For an input where N=3, D=2, and degree=3...

                [[[ 1.0        1.0]
                  [ x_{1,1}    x_{1,1}]
                  [ x_{1,1}^2  x_{1,2}^2]
                  [ x_{1,1}^3  x_{1,2}^3]]

                 [[ 1.0        1.0]
                  [ x_{2,1}    x_{2,2}]
                  [ x_{2,1}^2  x_{2,2}^2]
                  [ x_{2,1}^3  x_{2,2}^3]]

                 [[ 1.0        1.0]
                  [ x_{3,1}    x_{3,2}]
                  [ x_{3,1}^2  x_{3,2}^2]
                  [ x_{3,1}^3  x_{3,2}^3]]]

        """

        if len(x.shape)==1:
            feats = np.zeros((len(x), degree+1)) + 1
            for i in range(degree):
                feats[:, i+1] = x ** (i+1)
        else:
            feats = np.zeros((x.shape[0], degree+1, x.shape[1]))
            for i in range(x.shape[0]):
                feats[i,0,:] = 1
                for j in range(degree):
                    feats[i, j+1, :] = x[i, :] ** (j+1)
        return feats


    def predict(self, xtest, weight): # [5pts]
        """
        Args:
            xtest: (N,D) numpy array, where N is the number
                   of instances and D is the dimensionality
                   of each instance
            weight: (D,1) numpy array, the weights of linear regression model
        Return:
            prediction: (N,1) numpy array, the predicted labels
        """
        prediction = np.matmul(xtest, weight)
        return prediction

    # =================
    # LINEAR REGRESSION
    # Hints:
    # - In the fit function, use close form solution of the linear regression to get weights.
    # - For pseudo inverse, you can use numpy linear algebra function (np.linalg.pinv)
    # - For the predict, you need to use linear combination of data points and their weights (y = theta0*1+theta1*X1+...)

    def linear_fit_closed(self, xtrain, ytrain): # [5pts]
        """
        Args:
            xtrain: (N,D) numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: (N,1) numpy array, the true labels
        Return:
            weight: (D,1) numpy array, the weights of linear regression model
        """
        XTXinv = np.linalg.inv(np.matmul(xtrain.T, xtrain))
        XT = np.matmul(XTXinv, xtrain.T)
        weights = np.matmul(XT, ytrain)
        return weights

#This is a Bonus

    def linear_fit_GD(self, xtrain, ytrain, epochs=5, learning_rate=0.001): # [5pts]
        """
        Args:
            xtrain: (N,D) numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: (N,1) numpy array, the true labels
        Return:
            weight: (D,1) numpy array, the weights of linear regression model
            loss_per_epoch: (epochs,) list of floats, rmse of each epoch
        """
        weights = np.zeros((xtrain.shape[1], 1))
        for e in range(epochs):
            weights += learning_rate * np.dot(xtrain.T, ytrain - np.dot(xtrain, weights)) / len(ytrain)
        return weights

#This is a Bonus

    def linear_fit_SGD(self, xtrain, ytrain, epochs=100, learning_rate=0.001): # [5pts]
        """
        Args:
            xtrain: (N,D) numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: (N,1) numpy array, the true labels
        Return:
            weight: (D,1) numpy array, the weights of linear regression model
            loss_per_step: (N*epochs,) list of floats, rmse calculated after each update step

        NOTE: For autograder purposes, iterate through the dataset SEQUENTIALLY, NOT stochastically.


        Note: Keep in mind that the number of epochs is the number of
        complete passes through the training dataset. SGD updates the
        weight for one datapoint at a time, but for each epoch, you'll
        need to go through all of the points.
        """
        weight = np.zeros((xtrain.shape[1],1))
        for i in range(epochs):
            # select samples
            indice = np.random.choice(len(xtrain), len(xtrain))
            x = xtrain[indice,:]
            y = ytrain[indice,:]
            # calculate gradient
            grad = 2/xtrain.shape[0] * xtrain.T.dot(xtrain.dot(weight) - ytrain)
            grad *= learning_rate
            weight -= grad
        return weight


    # =================
    # RIDGE REGRESSION

    def ridge_fit_closed(self, xtrain, ytrain, c_lambda): # [5pts]
        """
        Args:
            xtrain: (N,D) numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: (N,1) numpy array, the true labels
            c_lambda: float value
        Return:
            weight: (D,1) numpy array, the weights of ridge regression model
        """
        XTXlambda = np.matmul(xtrain.T, xtrain) + c_lambda * np.identity(xtrain.shape[1])
        weights = np.matmul(np.matmul(np.linalg.inv(XTXlambda), xtrain.T), ytrain)
        return weights

#This is a Bonus question

    def ridge_fit_GD(self, xtrain, ytrain, c_lambda, epochs=500, learning_rate=1e-7): # [5pts]
        """
        Args:
            xtrain: (N,D) numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: (N,1) numpy array, the true labels
            c_lambda: float value
        Return:
            weight: (D,1) numpy array, the weights of linear regression model
            loss_per_epoch: (epochs,) list of floats, rmse of each epoch
        """
        weight = np.zeros((xtrain.shape[1],1))
        for i in range(epochs):
            # calculate gradient
            grad = 2/xtrain.shape[0] * xtrain.T.dot(xtrain.dot(weight) - ytrain) + 2*c_lambda*weight
            grad *= learning_rate
            weight -= grad
        return weight

#This is a Bonus question

    def ridge_fit_SGD(self, xtrain, ytrain, c_lambda, epochs=100, learning_rate=0.001): # [5pts]
        """
        Args:
            xtrain: (N,D) numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: (N,1) numpy array, the true labels
        Return:
            weight: (D,1) numpy array, the weights of linear regression model
            loss_per_step: (N*epochs,) list of floats, rmse calculated after each update step

        NOTE: For autograder purposes, iterate through the dataset SEQUENTIALLY, NOT stochastically.

        Note: Keep in mind that the number of epochs is the number of
        complete passes through the training dataset. SGD updates the
        weight for one datapoint at a time, but for each epoch, you'll
        need to go through all of the points.
        """
        weight = np.zeros((xtrain.shape[1],1))
        for i in range(epochs):
            # select samples
            indice = np.random.choice(len(xtrain), len(xtrain))
            x = xtrain[indice,:]
            y = ytrain[indice,:]
            # calculate gradient
            grad = 2/xtrain.shape[0] * xtrain.T.dot(xtrain.dot(weight) - ytrain) + 2*c_lambda*weight
            grad *= learning_rate
            weight -= grad
        return weight


    def ridge_cross_validation(self, X, y, kfold=10, c_lambda=100): # [5 pts]
        """
        Args:
            X : (N,D) numpy array, where N is the number of instances and D is the dimensionality of each instance
            y : (N,1) numpy array, true labels
            kfold: Number of folds you should take while implementing cross validation.
            c_lambda: Value of regularization constant
        Returns:
            meanErrors: float, average rmse error
        Hint: np.concatenate might be helpful.
        Look at 3.5 to see how this function is being used.
        # For cross validation, use 10-fold method and only use it for your training data (you already have the train_indices to get training data).
        # For the training data, split them in 10 folds which means that use 10 percent of training data for test and 90 percent for training.
        """
        N = X.shape[0]
        k = (N // kfold) + 1
        final_error = 0

        for i in range(k):
            if i == k - 1:
                start = i*kfold
                end = N
            else:
                start = i*kfold
                end = i*kfold + 10

            xtrain = np.concatenate((X[:start, :], X[end:, :]))
            ytrain = np.concatenate((y[:start, :], y[end:, :]))
            weights = self.ridge_fit_closed(xtrain, ytrain, c_lambda)

            ypred = self.predict(X[start:end,:], weights)
            error = np.sum((y[start:end, :] - ypred) ** 2)
            final_error += error

        return final_error / N


