import dynet as dy

# This is a very basic autoencoder with no fancy regularizers
class AutoEncoder(object):
  # Accepts a list as input which indicates the number of neurons in each hidden layer
  def __init__(self, model, num_input, hidden_layer_list, num_out, act=dy.tanh):
    self.number_of_layers = len(hidden_layer_list)
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
    for (W,b) in zip(weight_matrix_array[1:], biases_array[1:]):
        pred =  (W * g(intermediate))  + b  
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


