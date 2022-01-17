import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace

from perceptron import Perceptron, SignActivation
from activation import ActivationFunction

class Sigmoid(ActivationFunction):
   """ 
   Sigmoid activation: `f(x) = 1/(1+e^(-x))`
   """
   def forward(self, x):
      """
      Activation function output.
      """
      return 1/(1+np.exp(-x))[0]

   def gradient(self, x):
      """
      Activation function derivative.
      TODO: Change the function to return the correct value, given input `x`.
      """
      return self.forward(x) * (1 - self.forward(x))
      

class LinearActivation(ActivationFunction):
   """ 
   Linear activation: `f(x) = x`
   """
   def forward(self, x):
      """
      Activation function output.
      """
      return x

   def gradient(self, x):
      """
      TODO: Change the function to return the correct value, given input `x`.
      """
      return 1

class Layer:
   def __init__(self, n_inputs, n_units, act_f):
      """ Initialize the layer, creating `n_units` perceptrons with `num_inputs` inputs each. """
      self.n_inputs = n_inputs
      self.n_units = n_units

      # TODO Create the perceptrons required for the layer
      self.ps = [Perceptron(self.n_inputs,act_f) for i in range(self.n_units)]

   def activation(self, x):
      """ Returns the activation `a` of all perceptrons in the layer, given the input vector`x`. """
      return np.array([p.activation(x) for p in self.ps])

   def output(self, a):
      """ Returns the output `o` of all perceptrons in the layer, given the activation vector `a`. """
      return np.array([p.output(ai) for p, ai in zip(self.ps, a)])

   def predict(self, x):
      """ Returns the output `o` of all perceptrons in the layer, given the input vector `x`. """
      return np.array([p.predict(x) for p in self.ps])

   def gradient(self, a):
      """ Returns the gradient of the activation function for all perceptrons in the layer, given the activation vector `a`. """
      return np.array([p.gradient(ai) for p, ai in zip(self.ps, a)])

   def update_weights(self, dw):
      """ 
      Update the weights of all of the perceptrons in the layer, given the weight change of each.
      Input size: (n_inputs+1, n_units)
      """
      for j in range(self.n_units):      
         # print(self.ps[j].w)
         # print(self.ps[j].w.shape)
         # print(dw[:,j].reshape(-1,1))
         # print(dw[:,j].reshape(-1,1).shape)

         self.ps[j].w += dw[:,j].reshape(-1,1)

   @property
   def w(self):
      """
      Returns the weights of the neurons in the layer.
      Size: (n_inputs+1, n_units)
      """
      return np.array([p.w for p in self.ps]).T

   def import_weights(self, w):
      """ 
      Import the weights of all of the perceptrons in the layer.
      Input size: (n_inputs+1, n_units)
      """
      for j in range(self.n_units):
         self.ps[j].w = w[:,j] 

class MLP:
   """ 
   Multi-layer perceptron class

   Parameters
   ----------
   n_inputs : int
      Number of inputs
   n_hidden_units : int
      Number of units in the hidden layer
   n_outputs : int
      Number of outputs
   alpha : float
      Learning rate used for gradient descent
   """
   def __init__(self, n_inputs, n_hidden_units, n_outputs, alpha=1e-3):
      self.n_inputs = n_inputs
      self.n_hidden_units = n_hidden_units
      self.n_outputs = n_outputs

      self.alpha = alpha

      self.l1 = Layer(n_inputs+1, n_hidden_units, Sigmoid())
      self.l2 = Layer(n_hidden_units+1, n_outputs, LinearActivation())

   def predict(self, x):
      """ 
      Forward pass prediction given the input x
      """
      pred1 = self.l1.predict(x)
      pred1 = np.hstack((np.ones((1,1)),np.asarray(pred1).reshape(1,-1))).T
      pred2 = self.l2.predict(pred1)
      return pred2

   def train(self, x, t):
      """
      Train the network

      Parameters
      ----------
      `x` : numpy array
         Inputs (size: n_examples, n_inputs)
      `t` : numpy array
         Targets (size: n_examples, n_outputs)

      TODO: Write the function to iterate through training examples and apply gradient descent to update the neuron weights
      """


      hidden_layer_act = []
      hidden_layer_out = []
      output_layer_act = []
      output_layer_out = []

      # Add column of 1 for bias
      x = np.hstack((np.ones((x.shape[0],1)),x))

      # Loop over training examples
      for j in range(len(x)):
         # Forward pass
         x_i = x[j]
         t_i = t[j]
         # Activation for each perceptron
         act_h = self.l1.activation(x_i)
         hidden_layer_act.append(act_h)
         # Output from each perceptron
         out_h = self.l1.output(act_h)
         hidden_layer_out.append(out_h)
         # Activation output-layer for each datapoint
         act_o = self.l2.activation(np.hstack((np.ones((1)),out_h))) 
         output_layer_act.append(act_o)
         # Output from ouput-layer
         out_o = self.l2.output(act_o)
         output_layer_out.append(out_o)
      # end for

      # # Convert to np array
      hidden_layer_act = np.hstack((np.ones((len(x), 1)),np.asarray(hidden_layer_act).reshape(len(x),self.n_hidden_units)))
      hidden_layer_out = np.hstack((np.ones((len(x), 1)),np.asarray(hidden_layer_out).reshape(len(x),self.n_hidden_units)))
      output_layer_act = np.asarray(output_layer_act)
      output_layer_out = np.asarray(output_layer_out).reshape(len(x),1)      
      
      # # Backpropagation ##

      # Output error on predictions
      out_error = np.asarray(output_layer_out.T - t).T
      # Hidden layer error
      hidden_delta = (hidden_layer_out[:,1:] * (1 - hidden_layer_out[:,1:])) * np.dot(out_error, self.l2.w[0].T[:,1:])
      # Partial derivatives
      hidden_pd = x[:, :, np.newaxis] * hidden_delta[: , np.newaxis, :]
      output_pd = hidden_layer_out[:, :, np.newaxis] * out_error[:, np.newaxis, :]
      # Average for total gradients
      total_hidden_gradient = np.average(hidden_pd, axis=0)
      total_output_gradient = np.average(output_pd, axis=0)
      # print(total_hidden_gradient)
      # print(total_output_gradient)
      # Update weights
      self.l1.update_weights(-self.alpha*total_hidden_gradient)
      self.l2.update_weights(-self.alpha*total_output_gradient)

   def export_weights(self):
      return [self.l1.w, self.l2.w]
   
   def import_weights(self, ws):
      if ws[0].shape == (self.l1.n_units, self.n_inputs+1) and ws[1].shape == (self.l2.n_units, self.l1.n_units+1):
         print("Importing weights..")
         self.l1.import_weights(ws[0])
         self.l2.import_weights(ws[1])
      else:
         print("Sizes do not match")

def calc_prediction_error(model, x, t):
   """ Calculate the MSE prediction error """
   x = np.hstack((np.ones((x.shape[0],1)),x))
   y_hat = np.asarray([model.predict(x[i]) for i in range(len(x))]).reshape(-1,1)[:,0]
   err = y_hat - t
   return np.sum(err**2) / (2*len(t))

np.random.seed(1)

if __name__ == "__main__":


   data = np.array( [ [0.5, 0.5, 0], [1.0, 0, 0], [2.0, 3.0, 0], [0, 1.0, 1], [0, 2.0, 1], [1.0, 2.2, 1] ] )
   x = data[:,:2]
   t = data[:,2]
   # x = np.hstack((np.ones((x.shape[0],1)),x))

   '''2.1: Test new activation functions'''
   # data = np.array( [ [0.5, 0.5, 0], [1.0, 0, 0], [2.0, 3.0, 0], [0, 1.0, 1], [0, 2.0, 1], [1.0, 2.2, 1] ] )
   # x = data[:,:2]
   # act_f = Sigmoid()
   # perc = Perceptron(2,act_f)
   # print("\n")
   # for el in x:
   #    y = perc.predict(el)
   #    print("Data point: {} \t--> \tprediction: {:.4}".format(el,y))
   # print("\n")


   '''2.2 Test layer''' 
   # l = Layer(2,5,Sigmoid())
   # out = l.predict(np.array((np.pi,1)).T)
   # print("\n--- Output of a layer of 5 neurons to the input [pi 1].T ---\n\n{}\n".format(out))
   # print("\n--- Weights of the whole layer ---\n\n{}\n".format(l.w[0]))


   ''' 2.4 : Test MLP class init'''
   # mlp = MLP(2,3,1)
   # print("\nWeights from input to hidden layer:\n\n{}".format(mlp.export_weights()[0]))
   # print("\nWeights from hidden layer to output:\n\n{}\n".format(mlp.export_weights()[1]))


   ''' 2.5: Mean Square Error '''
   # data = np.array( [ [0.5, 0.5, 0], [1.0, 0, 0], [2.0, 3.0, 0], [0, 1.0, 1], [0, 2.0, 1], [1.0, 2.2, 1] ] )
   # x = data[:,:2]
   # t = data[:,2]
   # mlp = MLP(2,3,1)
   # mse = calc_prediction_error(mlp,x,t)
   # print("\n--- MSE for the first iteration: {} ---\n".format(mse))


   '''' 2.6: Write a for loop to train the network for a number of iterations. Make plots.'''
   # data = np.array( [ [0.5, 0.5, 0], [1.0, 0, 0], [2.0, 3.0, 0], [0, 1.0, 1], [0, 2.0, 1], [1.0, 2.2, 1] ] )
   # x = data[:,:2]
   # t = data[:,2]
   # mlp = MLP(2,3,1)
   # num_iterations = 10000
   # for i in range(num_iterations):
   #    mlp.train(x,t)

   ''' 2.7: Train as XOR gate '''
   data = np.array( [ [0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0] ] )
   x = data[:,:2]
   t = data[:,2]
   alpha = 0.1
   mlp = MLP(2,2,1,alpha)

   
   ## 1.7.a
   # err = calc_prediction_error(mlp,x,t)
   # it = 0
   # while err > 0.001:
   #    mlp.train(x,t)
   #    err = calc_prediction_error(mlp,x,t)
   #    it +=1
   # print("\n--- Converged after {} iterations with a learning rate of {} ---".format(it,alpha))
   # print("\n--- Output after convergence: ---")
   # x_bias = np.hstack((np.ones((x.shape[0],1)),x))
   # for i in range(len(x)):
   #    print("\t",mlp.predict(x_bias[i])[0][0])
   # print("\n")

   ## 1.7.b
   # err = calc_prediction_error(mlp,x,t)
   # err_l = []
   # it = 0
   # while err > 0.001:
   #    mlp.train(x,t)
   #    err = calc_prediction_error(mlp,x,t)
   #    err_l.append(err)
   #    it +=1
   # fig = plt.figure()
   # ax = plt.axes()
   # x1 = linspace(0,it,it)
   # plt.title("MSE as function of training epochs")
   # plt.xlabel("Nummber of training epoch")
   # plt.ylabel("MSE")
   # # plt.yscale('log')
   # plt.plot(x1,err_l)
   # plt.grid()
   # plt.show()