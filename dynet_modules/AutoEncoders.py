import dynet as dy
import os
import pickle

# This is a very basic autoencoder with no fancy regularizers
class AutoEncoder(object):
  # Accepts a list as input which indicates the number of neurons in each hidden layer
  def __init__(self, model, num_input, hidden_layer_list, num_out, act=dy.tanh):
    self.number_of_layers = len(hidden_layer_list)
    self.num_input = num_input
    self.hidden_layer_list = hidden_layer_list
    self.num_out = num_out
    num_hidden_1 = hidden_layer_list[0]
    self.W1 = model.add_parameters((num_hidden_1, num_input))
    self.b1 = model.add_parameters((num_hidden_1))
    self.weight_matrix_array = []
    self.biases_array = []
    self.weight_matrix_array.append(self.W1)
    self.biases_array.append(self.b1)
    for k in range(1, self.number_of_layers):
              self.weight_matrix_array.append(model.add_parameters((hidden_layer_list[k], hidden_layer_list[k-1])))
              self.biases_array.append(model.add_parameters((hidden_layer_list[k])))
    self.weight_matrix_array.append(model.add_parameters((num_out, hidden_layer_list[-1])))
    self.biases_array.append(model.add_parameters((num_out)))
    self.act = act
    self.spec = (num_input, hidden_layer_list, num_out, act)

  def __call__(self, input, output, classification_flag):
    weight_matrix_array = []
    biases_array = []
    for (W,b) in zip(self.weight_matrix_array, self.biases_array):
         weight_matrix_array.append(dy.parameter(W))
         biases_array.append(dy.parameter(b)) 
    #print " The number of total layers is ", len(weight_matrix_array)
    g = self.act
    w = weight_matrix_array[0]
    b = biases_array[0]
    intermediate = w*input + b
    activations = [intermediate]
    for (W,b) in zip(weight_matrix_array[1:], biases_array[1:]):
        pred =  (W * g(activations[-1]))  + b
        activations.append(pred)  
    if classification_flag == 1:
       return dy.softmax(pred)
    else:
       losses = output - pred
       return dy.l2_norm(losses)

  # support saving:
  def param_collection(self): return self.pc

  @staticmethod
  def from_spec(spec, model):
    num_input, hidden_layer_list, num_out, act = spec
    return AutoEncoder(model, num_input, hidden_layer_list, num_out, act)

  def save(self, path):
     if not os.path.exists(path): os.makedirs(path)
     arr = [self.num_input, self.hidden_layer_list, self.num_out]
     with open(path + '_model_hyps', 'w') as f: pickle.dump(arr, f) 
     self.model.save(path, '_model')

  @staticmethod
  def load(model, path, load_model_params=True):
      if not os.path.exists(path): raise Exception("Model "+path+" does not exist")
      with open(path+"_model_hyps", "r") as f: arr = pickle.load(f)
      model.populate(path + '_model')
      number_of_layers = len(arr) -2
      num_hidden_1 = arr[1]
      num_input = arr[0]
      num_output = arr[-1]
      return AutoEncoder(model, num_input, hidden_layer_list, num_out, act=dy.tanh)

