# -*- coding: utf-8 -*-
'''
Homework code - selfmade Artificial Neutral Network
Author: LUO Chensheng
Time: 12 April 2025
'''
import numpy as np
import matplotlib.pyplot as plt

class ANN:
    def __init__(self, input_size, hidden_layers, output_size,learning_rate=0.01):
        '''
        Parameters
        ----------
        input_size : int, size of the input layer
        hidden_layers : list of int, sizes of the hidden layers
        output_size : int, size of the output layer
        '''
        # Initialize the parameters
        self.input_size = input_size
        self.output_size = output_size
        self.layers = np.array([input_size] + hidden_layers + [output_size])
        self.training_layer_numbers = len(hidden_layers)+1
        self.learning_rate = learning_rate
        
        # Initialize the weights and biases
        '''
        W: list of weight matrices, for layer i, W[i]=np.matrix(former layer,next layer)
        dLdW : corresponding derivatives 
        b: list of bias vectors, for layer i, b[i]=np.matrix(next layer)
        dLdb : corresponding derivatives
        z: list of predicted hidden layer vectors
        dLdz : corresponding derivatives
        '''
        
        self.W = []
        self.dLdW = []
        for i in range(self.training_layer_numbers):
            self.W.append(np.random.rand(self.layers[i], self.layers[i+1]))
            self.dLdW.append(np.zeros((self.layers[i], self.layers[i+1])))
        
        self.b = []
        self.dLdb = []
        for i in range(self.training_layer_numbers):
            self.b.append(np.random.rand(self.layers[i+1]))
            self.dLdb.append(np.zeros(self.layers[i+1]))

        self.z = []
        self.dLdz = []
        for i in range(self.training_layer_numbers+1):
            self.z.append(np.zeros(self.layers[i]))
            self.dLdz.append(np.zeros(self.layers[i]))

    def sigmoid(self, x):
        if x > 0:
            y = 1 / (1 + np.exp(-x))
        else:
            y = np.exp(x) / (1 + np.exp(x))
        return y

    def sigmoid_derivative(self, y):
        '''
        The input of this function should be the value after sigmoid
        '''
        return y * (1 - y)

    def predict(self, x):
        '''
        x : input variable, should be a numpy array
        y : output variable, should be a numpy array
        '''
        # Test input size
        assert len(x) == self.input_size, "Input size mismatch"
        self.z[0]=x
        for i in range(self.training_layer_numbers):
            for k in range(np.size(self.W[i], 1)):
                self.z[i+1][k] = self.sigmoid(np.dot(self.z[i], self.W[i][:,k]) + self.b[i][k])
            
        return self.z[-1]
    
    def training(self, xs , ys, round = 1000):
        '''
        xs : input variable of training data, should be a numpy 2D array.
            - 1st dimension: number of samples
            - 2nd dimension: input size
        ys : output variable of training data, should be a numpy 2D array.
            - 1st dimension: number of samples
            - 2nd dimension: output size
        round: integer, number of training rounds, default is 1000
        '''
        assert np.size(xs,0) == np.size(ys,0), "Input and output sampling size mismatch"
        assert np.size(xs,1) == self.input_size, "Input size mismatch"
        assert np.size(ys,1) == self.output_size, "Output size mismatch"

        sampling_nb = np.size(xs,0)

        losses = []

        for _ in range(round):
            for i in range(self.training_layer_numbers):
                self.dLdW[i] = np.zeros_like(self.dLdW[i])
                self.dLdb[i] = np.zeros_like(self.dLdb[i])
            loss = 0

            for m in range(np.size(xs, 0)):
                # Forward pass
                x = xs[m]
                y = ys[m]
                y_pred = self.predict(x)
                loss += - y*np.log(y_pred) - (1-y)*np.log(1-y_pred)
                for i in range(self.training_layer_numbers+1):
                    self.dLdz[i] = np.zeros_like(self.dLdz[i])

                # Backward pass
                self.dLdz[-1] = - y/y_pred + (1-y)/(1-y_pred)
                for i in range(self.training_layer_numbers, 0, -1):
                    for j in range(self.layers[i-1]):
                        self.dLdz[i-1][j] = 0
                        for k in range(self.layers[i]):
                            self.dLdW[i-1][j,k] += self.dLdz[i][k] * self.sigmoid_derivative(self.z[i][k]) * self.z[i-1][j]
                            self.dLdb[i-1][k] += self.dLdz[i][k] * self.sigmoid_derivative(self.z[i][k])
                            self.dLdz[i-1][j] += self.dLdz[i][k] * self.sigmoid_derivative(self.z[i][k]) * self.W[i-1][j,k]

            # Update weights and biases
            for i in range(self.training_layer_numbers):
                # # Remove the 
                # self.dLdW[j] -= 1
                # self.dLdb[j] -= 1
                self.W[i] -= self.learning_rate * self.dLdW[i] / sampling_nb
                self.b[i] -= self.learning_rate * self.dLdb[i] / sampling_nb
            
            # print('W',self.W)
            # print('b',self.b)
            losses.append(loss/sampling_nb)
        
        return losses
    

if __name__ == "__main__":
    # Test the ANN class
    ann = ANN(input_size=2, hidden_layers=[2], output_size=1,learning_rate=1)
    
    
    x = np.array([0.01,0.1])
    y = ann.predict(x)
    print("Output:", y)

    print('Training')
    xs = np.array([[0,0],[0.1,0.1],[1,0],[0,1],[1,1]])
    ys = np.array([[1],[1],[0],[0],[1]])
    ann.training(xs , ys, round = 1000)

    x = np.array([0.01,0.1])
    y = ann.predict(x)
    print(x,"Output:", y)

    x = np.array([0.9,0.1])
    y = ann.predict(x)
    print(x,"Output:", y)

    x = np.array([0.1,0.9])
    y = ann.predict(x)
    print(x,"Output:", y)

    x = np.array([0.9,0.9])
    y = ann.predict(x)
    print(x,"Output:", y)