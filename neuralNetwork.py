import numpy as np
from math import *
import matplotlib.pyplot as plt

def sigmoid(x):
   return 1/(1+exp(-x))

def dsigmoid(y):
   return y*(1-y)

class NeuralNetwork:
    def __init__(self, n_inputs, n_hiddens, n_outputs):

       self.n_inputs = n_inputs
       self.n_hiddens = n_hiddens
       self.n_outputs = n_outputs

       self.learning_rate = 10

       self.w_i_h = np.random.rand(n_hiddens, n_inputs)*2 -1
       self.b_i_h = np.random.rand(n_hiddens,1)*2-1

       self.w_h_o = np.random.rand(n_outputs, n_hiddens)*2-1
       self.b_h_o = np.random.rand(n_outputs,1)*2-1

    def predict(self, input):
       sig = np.vectorize(sigmoid)
       dsig = np.vectorize(dsigmoid)
       
       inputs = np.array(input)
       inputs.reshape(self.n_inputs,1)

       hiddens = sig(np.matmul(self.w_i_h, inputs)+ self.b_i_h)
       outputs = sig(np.matmul(self.w_h_o, hiddens)+self.b_h_o)

       return outputs

    def train(self, input, target):
        sig = np.vectorize(sigmoid)
        dsig = np.vectorize(dsigmoid)
       
        inputs = np.array(input)
        inputs.reshape(self.n_inputs,1)
        
        targets = np.array(target)
        targets.reshape(self.n_outputs,1)
        
        hiddens = sig(np.add(np.dot(self.w_i_h, inputs), self.b_i_h))
        outputs = sig(np.add(np.dot(self.w_h_o, hiddens),self.b_h_o))
        
        error_o = np.subtract(targets, outputs)
        error_h = np.dot(np.transpose(self.w_h_o), error_o)
        
        grad_o = dsig(outputs)
        dw_h_o =  np.dot( self.learning_rate * grad_o * error_o, np.transpose(hiddens) )
        db_h_o = self.learning_rate * grad_o * error_o
        self.w_h_o = self.w_h_o + dw_h_o
        self.b_h_o = self.b_h_o + db_h_o
        
        grad_h = dsig(hiddens)
        dw_i_h =  np.dot( self.learning_rate * grad_h * error_h, np.transpose(inputs) )
        db_i_h = self.learning_rate * grad_h * error_h
        self.w_i_h = self.w_i_h + dw_i_h
        self.b_i_h = self.b_i_h + db_i_h
        
        
        

nn = NeuralNetwork(2,10,1)
for x in range(100):
   for y in range(100):
      nn.train([[x/100],[y/100]], [[sqrt((x/100-0.5)*(x/100-0.5) + (y/100-0.5)*(y/100-0.5))]])

im = np.zeros((100,100))
for x in range(100):
   for y in range(100):
      im[x][y] = nn.predict([[x/100], [y/100]])[0][0]
      print(nn.predict([[x/100], [y/100]]))
plt.imshow(im)
plt.show()