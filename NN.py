import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix

'''
We are going to use the diabetes dataset provided by sklearn
https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset
to train a 2 fully connected layer neural net. We are going to build the neural network from scratch.
'''


class dlnet:

    def __init__(self, x, y, lr = 0.01, batch_size=64):
        '''
        This method initializes the class, it is implemented for you.
        Args:
            x: data
            y: labels
            Yh: predicted labels
            dims: dimensions of different layers
            alpha: slope coefficient for leaky relu
            param: dictionary of different layers parameters
            ch: Cache dictionary to store forward parameters that are used in backpropagation
            loss: list to store loss values
            lr: learning rate
            sam: number of training samples we have

        '''
        self.X=x # features
        self.Y=y # ground truth labels

        self.Yh=np.zeros((1,self.Y.shape[1])) # estimated labels
        self.dims = [10, 15, 1] # dimensions of different layers
        self.alpha = 0.05

        self.param = { } # dictionary for different layer variables
        self.ch = {} # cache for holding variables during forward propagation to use them in back prop
        self.loss = [] # list to store loss values
        self.batch_y = [] # list of y batched numpy arrays

        self.iter = 0 # iterator to index into data for making a batch
        self.batch_size = batch_size # batch size

        self.lr=lr # learning rate
        self.sam = self.Y.shape[1] # number of training samples we have
        self._estimator_type = 'classifier'
        self.neural_net_type = "Leaky Relu -> Tanh"




    def nInit(self):
        '''
        This method initializes the neural network variables, it is already implemented for you.
        Check it and relate to the mathematical description above.
        You are going to use these variables in forward and backward propagation.
        '''
        np.random.seed(1)
        self.param['theta1'] = np.random.randn(self.dims[1], self.dims[0]) / np.sqrt(self.dims[0])
        self.param['b1'] = np.zeros((self.dims[1], 1))
        self.param['theta2'] = np.random.randn(self.dims[2], self.dims[1]) / np.sqrt(self.dims[1])
        self.param['b2'] = np.zeros((self.dims[2], 1))


    def Leaky_Relu(self,alpha, u):
        '''
        In this method you are going to implement element wise Leaky_Relu.
        Make sure that all operations here are element wise and can be applied to an input of any dimension.
        Input:
            u of any dimension
            alpha: the slope coefficent of the negative part.
        return: Leaky_Relu(u)
        '''
        #TODO: implement this

        return np.maximum(alpha*u, u)


    def Tanh(self, u):
        '''
        In this method you are going to implement element wise Tanh.
        Make sure that all operations here are element wise and can be applied to an input of any dimension.
        Input: u of any dimension
        return: Tanh(u)
        '''
        #TODO: implement this

        Tanhu = (np.exp(u) - np.exp(-u))/(np.exp(u) + np.exp(-u))
        return Tanhu




    def dL_Relu(self,alpha, u):
        '''
        This method implements element wise differentiation of Leaky Relu, it is already implemented for you.
        Input:
             u of any dimension
             alpha: the slope coefficent of the negative part.
        return: dL_Relu(u)
        '''

        u[u<=0] = alpha
        u[u>0] = 1
        return u


    def dTanh(self, u):
        '''
        This method implements element wise differentiation of Tanh, it is already implemented for you.
        Input: u of any dimension
        return: dTanh(u)
        '''

        o = np.tanh(u)
        return 1-o**2



    def nloss(self,y, yh):
        '''
        In this method you are going to implement mean squared loss.
        Refer to the description above and implement the appropriate mathematical equation.
        Input: y 1xN: ground truth labels
               yh 1xN: neural network output

        return: MSE 1x1: loss value
        '''

        #TODO: implement this

        return (np.sum(np.square(y-yh)))/(2*y.shape[1])





    def forward(self, x):
        '''
        Fill in the missing code lines, please refer to the description for more details.
        Check nInit method and use variables from there as well as other implemented methods.
        Refer to the description above and implement the appropriate mathematical equations.
        do not change the lines followed by #keep.

        Input: x DxN: input
        return: o2 1xN
        '''
        #TODO: implement this


        u0 = x
        u1 = np.dot(self.param["theta1"],u0) + self.param["b1"]
        o1 = self.Leaky_Relu(self.alpha, u1)
        u2 = np.dot(self.param["theta2"],o1) + self.param["b2"]
        o2 = self.Tanh(u2)
        yh = o2


        self.ch['X'] = x #keep



        self.ch['u1'],self.ch['o1']=u1,o1 #keep
        self.ch['u2'],self.ch['o2']=u2,o2 #keep

        return o2 #keep


    def backward(self, y, yh):
        '''
        Fill in the missing code lines, please refer to the description for more details
        You will need to use cache variables, some of the implemented methods, and other variables as well
        Refer to the description above and implement the appropriate mathematical equations.
        do not change the lines followed by #keep.

        Input: y 1xN: ground truth labels
               yh 1xN: neural network output

        Return: dLoss_theta2 (1x15), dLoss_b2 (1x1), dLoss_theta1 (15xD), dLoss_b1 (15x1)

        '''
        #TODO: implement this

        N = y.shape[1]

        dLoss_o2 = (self.ch["o2"]-y)/N
        dLoss_u2 = np.multiply(dLoss_o2,self.dTanh(self.ch["u2"]))
        dLoss_theta2 = np.matmul(dLoss_u2,self.ch["o1"].T)
        dLoss_b2 = np.matmul(dLoss_u2, np.ones(dLoss_u2.shape).T)
        dLoss_o1 = np.dot(self.param["theta2"].T, dLoss_u2)
        dLoss_u1 = np.multiply(dLoss_o1, self.dL_Relu(self.alpha, self.ch["o1"]))
        dLoss_theta1 = np.matmul(dLoss_u1,self.ch["X"].T)
        dLoss_b1 = np.matmul(dLoss_u1, np.ones(dLoss_u2.shape).T)




        # parameters update, no need to change these lines
        self.param["theta2"] = self.param["theta2"] - self.lr * dLoss_theta2 #keep
        self.param["b2"] = self.param["b2"] - self.lr * dLoss_b2 #keep
        self.param["theta1"] = self.param["theta1"] - self.lr * dLoss_theta1 #keep
        self.param["b1"] = self.param["b1"] - self.lr * dLoss_b1 #keep
        return dLoss_theta2, dLoss_b2, dLoss_theta1, dLoss_b1


    def gradient_descent(self, x, y, iter = 60000):
        '''
        This function is an implementation of the gradient descent algorithm.
        Notes:
        1. GD considers all examples in the dataset in one go and learns a gradient from them.
        2. One iteration here is one round of forward and backward propagation on the complete dataset.
        3. Append loss at multiples of 1000 i.e. at 0th, 1000th, 2000th .... iterations to self.loss

        Input: x DxN: input
               y 1xN: labels
               iter: scalar, number of epochs to iterate through
        '''

        #Todo: implement this

        self.nInit()
        for i in range(0, iter):
            yh = self.forward(x)
            dLoss_theta2, dLoss_b2, dLoss_theta1, dLoss_b1 = self.backward(y, yh)
            if i % 1000 == 0:
                self.loss.append(self.nloss(y,yh))


    def predict(self, x):
        '''
        This function predicts new data points
        It is implemented for you

        Input: x DxN: inputs
        Return: y 1xN: predictions

        '''
        Yh = self.forward(x)
        return Yh
