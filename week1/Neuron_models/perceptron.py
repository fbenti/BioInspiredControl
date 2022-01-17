import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
from activation import ActivationFunction

class SignActivation(ActivationFunction):
   """ 
   Sign activation: `f(x) = 1 if x > 0, 0 if x <= 0`
   """
   def forward(self, x):
      """
      Function output.
      """
      if x > 0: return 1
      return 0
      
   def gradient(self, x):
      """
      Function derivative.
      """
      return 0

class Perceptron:
   """ 
   Perceptron neuron model
   
   Parameters
   ----------
   n_inputs : int
      Number of inputs
   act_f : Subclass of `ActivationFunction`
      Activation function
   """
   def __init__(self, n_inputs, act_f):
      """
      Perceptron class initialization
      """
      if not isinstance(act_f,ActivationFunction) or not issubclass(type(act_f), ActivationFunction):
         raise TypeError('act_f has to be a subclass of ActivationFunction (not a class instance).')

      self.w = 2*np.random.random((n_inputs, 1)) - 1
      self.f = act_f # Define activation function

      if self.f is not None and not isinstance(self.f, ActivationFunction):
         raise TypeError("self.f should be a class instance.")

   def activation(self, x):
      """
      Computes the activation `a` given input `x`
      """
      return self.w.T @ x


   def output(self, a):
      """
      Computes the neuron output `y`, given the activation `a`
      """
      return self.f.forward(a)
      

   def predict(self, x):
      """
      Computes the neuron output `y`, given the input `x`
      """
      out = self.output(self.activation(x))
      return out


   def updateW(self,x,t,out,mu):
      self.w += (mu * (t - out) * x.reshape(self.w.shape[0],1))


   def gradient(self, a):
      """
      Computes the gradient of the activation function, given the activation `a`
      """
      return self.f.gradient(a)

if __name__ == '__main__':
   data = np.array( [ [0.5, 0.5, 0], [1.0, 0, 0], [2.0, 3.0, 0], [0, 1.0, 1], [0, 2.0, 1], [1.0, 2.2, 1] ] )
   x = data[:,:2]
   t = data[:,2]
   x = np.hstack((np.ones((x.shape[0],1)),x))
   # print(x)
   # print(t)
   
   '''1.1 Test your activation functions'''
   # act_f = SignActivation()
   # a = act_f = act_f.forward(1)
   # print(a)
     
   ''' 1.2 Test initialization'''
   # act_f = SignActivation()
   # perc = Perceptron(3,act_f)
   # print(perc.predict(data[5,:]))
   
   '''1.4'''
   mu = 0.1
   perc = Perceptron(3,SignActivation())
   notConverged = True

   it = 0
   while notConverged:
      y_hat = []
      for i in range(len(data)):

         x_i = x[i]
         t_i = t[i]
         out = perc.predict(x_i)

         while out != t_i:
            perc.updateW(x_i,t_i,out,mu)
            out = perc.predict(x_i)

         notConverged = False
         y_hat.append(out)
      
      for i in range(len(data)):
         if perc.predict(x[i]) != t[i]:
            notConverged = True
            break
      it += 1
   print("\nConverged after {}\n".format(it))
   print("\nFinal weights :\n{} \n".format(perc.w))   

   ## 1.6 Learn the weights
   # mu = 0.1
   # perc = Perceptron(len(x[0]),SignActivation())
   # notConverged = True

   # while notConverged:
   #    y_hat = []
   #    for i in range(len(data)):

   #       x_i = x[i]
   #       t_i = t[i]
   #       out = perc.predict(x_i)

   #       while out != t_i:
   #          perc.updateW(x_i,t_i,out,mu)
   #          out = perc.predict(x_i)

   #       notConverged = False
   #       y_hat.append(out)
      
   #    for i in range(len(data)):
   #       if perc.predict(x[i]) != t[i]:
   #          notConverged = True
   #          break


   # print("\nFinal weights :\n{} \n".format(perc.w))
   
   # PLOTS 1.6s
   # fig = plt.figure()
   # ax = plt.axes()
   # x1 = x[:,1]
   # x2 = x[:,2]

   # scatter = ax.scatter(x1,x2, c = y_hat)
   # legend = ax.legend(*scatter.legend_elements())
   # ax.add_artist(legend)
   
   # bias = perc.w[0]
   # w1 = perc.w[1]
   # w2 = perc.w[2]
   # x1 = linspace(0,2)
   # x2 = -bias/w2 - x1*w1/w2
   # a0 = -bias/w2
   # a1 = - w1/w2
   # print("\nBoundary equation :\nx2 = {:.3}*x1 + {:.3} \n".format( a1[0], a0[0]))
   # plt.plot(x1,x2)
   # plt.xlabel("x1")
   # plt.ylabel("x2")
   # plt.grid()
   # plt.show()