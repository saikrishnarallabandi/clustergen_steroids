from MLP import TwoLayerMLP
import dynet as dy
import numpy as np
from math import sqrt
from sklearn import preprocessing

# Load the training input and output
train_input = np.loadtxt('../data/input.test')
train_output = np.loadtxt('../data/output.test')

# Preprocess the data
input_scaler = preprocessing.MinMaxScaler().fit(train_input)
a = input_scaler.transform(train_input)
output_scaler = preprocessing.StandardScaler().fit(train_output)
b = output_scaler.transform(train_output)

# Hyperparaeters for the MLP
units_input = 711
units_hidden_1 = 1024
units_hidden_2 = 1024
units_output = 57

# Instantiate mlp and define the loss
m = dy.Model()
mlp = TwoLayerMLP(m, units_input, units_hidden_1, units_hidden_2, units_output, dy.rectify)
trainer = dy.AdamTrainer(m)


# Loop over the training instances and call the mlp
for epoch in range(30):
  train_loss = 0
  count = 1
  for (i,o) in zip(a,b):
     # Renew the computation graph
     dy.renew_cg()
     count += 1
     # 'Input' the vector as a dynet expression
     I = dy.inputTensor(i)
     O = dy.inputTensor(o)
     loss = mlp(I,O, 0)
     #mse = 0
     #for k in loss.value():
     #   mse += k*k
     #train_loss += sqrt(mse) 
     train_loss += loss.value()
     if count % 7000 == 1:
         print "  Loss at epoch ", epoch, " after " , count, " examples is ", float(train_loss/count)
     loss.backward()
     trainer.update() 
  print "Train Loss after epoch ", epoch , " : ", float(train_loss/count)
  print '\n'
