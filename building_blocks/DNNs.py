import dynet as dy
import os
import pickle
import numpy as np
import numpy as np
import os,sys
from sklearn import preprocessing
import pickle, logging

debug = 0

class FeedForwardNeuralNet(object):

  def __init__(self, model, args):
    self.pc = model.add_subcollection()
    self.args = args
    self.num_input = int(args[0])
    self.num_output = int(args[2])
    self.hidden_list = args[1]
    self.act = args[3]
    self.model = model
    self.number_of_layers = len(self.hidden_list)
    num_hidden_1 = self.hidden_list[0]
    
    # Add first layer
    if debug == 1:
        print "Adding input to first hidden layer weights ", num_hidden_1, self.num_input
    self.W1 = self.pc.add_parameters((num_hidden_1, self.num_input))
    self.b1 = self.pc.add_parameters((num_hidden_1))
    
    # Add remaining layers
    self.weight_matrix_array = []
    self.biases_array = []
    self.weight_matrix_array.append(self.W1)
    self.biases_array.append(self.b1)
    for k in range(1, self.number_of_layers):
              if debug == 1: 
                   print "At ", k , " adding weights ", self.hidden_list[k], self.hidden_list[k-1]
              self.weight_matrix_array.append(self.model.add_parameters((self.hidden_list[k], self.hidden_list[k-1])))
              self.biases_array.append(self.model.add_parameters((self.hidden_list[k])))
    if debug == 1:
      print "Adding last hidden layer to output weights ", self.num_output, self.hidden_list[-1]
    self.weight_matrix_array.append(self.model.add_parameters((self.num_output, self.hidden_list[-1])))
    self.biases_array.append(self.model.add_parameters((self.num_output)))
    self.spec = (self.num_input, self.hidden_list, self.num_output, self.act)
   
  def basic_affine(self, exp):
    W1 = dy.parameter(self.W1)
    b1 = dy.parameter(self.b1)
    return dy.tanh(dy.affine_transform([b1,W1,exp]))

  def calculate_loss(self, input, output):
    #dy.renew_cg()
    weight_matrix_array = []
    biases_array = []
    for (W,b) in zip(self.weight_matrix_array, self.biases_array):
         weight_matrix_array.append(dy.parameter(W))
         biases_array.append(dy.parameter(b)) 
    acts = self.act
    w = weight_matrix_array[0]
    b = biases_array[0]
    act = acts[0]
    intermediate = act(dy.affine_transform([b, w, input]))
    activations = [intermediate]
    for (W,b,g) in zip(weight_matrix_array[1:], biases_array[1:], acts[1:]):
        pred = g(dy.affine_transform([b, W, activations[-1]]))
        activations.append(pred)  
    #print output.value(), pred.value()
    losses = output - pred
    return dy.l2_norm(losses)

  def calculate_loss_classification(self, input, output):
    #dy.renew_cg()
    weight_matrix_array = []
    biases_array = []
    for (W,b) in zip(self.weight_matrix_array, self.biases_array):
         weight_matrix_array.append(dy.parameter(W))
         biases_array.append(dy.parameter(b))
    acts = self.act
    w = weight_matrix_array[0]
    b = biases_array[0]
    act = acts[0]
    intermediate = act(dy.affine_transform([b, w, input]))
    activations = [intermediate]
    for (W,b,g) in zip(weight_matrix_array[1:], biases_array[1:], acts[1:]):
        pred = g(dy.affine_transform([b, W, activations[-1]]))
        activations.append(pred)
    #print output.value(), pred.value()
    losses = dy.binary_log_loss(pred, output)
    return losses



  def predict(self, input):
    weight_matrix_array = []
    biases_array = []
    acts = []
    for (W,b, act) in zip(self.weight_matrix_array, self.biases_array, self.act):
         weight_matrix_array.append(dy.parameter(W))
         biases_array.append(dy.parameter(b))
         acts.append(act)
    g = acts[0]
    w = weight_matrix_array[0]
    b = biases_array[0]
    intermediate = g(w*input + b)
    activations = [intermediate]
    for (W,b, act) in zip(weight_matrix_array[1:], biases_array[1:], acts):
        pred =  act(W * activations[-1]  + b)
        activations.append(pred)
    return pred

  # support saving:
  def param_collection(self): return self.pc

  @staticmethod
  def from_spec(spec, model):
    num_input, hidden_list, num_output, act = spec
    return FeedForwardNeuralNet(model, [num_input, hidden_list, num_output, act])

  '''
  def save(self, path):
     if debug:
         print "Trying to save"
     if not os.path.exists(path): os.makedirs(path)
     dy.save(path + '_model', [self.model])
     with open(path+"_model_params", "w") as f: pickle.dump(self.args, f)
     if debug:
        print "Saved to ", path

  @staticmethod
  def load(path, model,load_model_params=True):
      if debug:
         print "Trying to load the model"
      if not os.path.exists(path): raise Exception("Model "+path+" does not exist")
      with open(path+"_model_params", "r") as f: args = pickle.load(f)
      [model] = dy.load(path + '_model', model)
      return FeedForwardNeuralNet(m, args)
  '''
 
if debug:
  # Load sample file and get first frame
  a = np.loadtxt('VCC2SF1.frames')
  A = a[0]
  m = dy.Model()
  dnn_1 = FeedForwardNeuralNet(m, [56, [2,12,15], 3, [dy.rectify, dy.rectify, dy.rectify]])
  s = dnn_1.basic_affine(dy.inputTensor(A))
  dy.save('models/basic_dnn', [dnn_1])
  # Print a forward pass output
  print "S:" , s.value()

  # Reload the model
  m2 = dy.Model()
  [dnn_2] = dy.load('models/basic_dnn', m2)
  b = dnn_2.basic_affine(dy.inputTensor(A))
  print "B:" , b.value()

  assert s.value() == b.value() 


class LoggingCallback(Callback):
    """Callback that logs message at end of epoch.
    """
    def __init__(self, location, print_fcn="print"):
        Callback.__init__(self)
        self.print_fcn = print_fcn
        self.location = '/tmp'

    def on_epoch_end(self, epoch,location, logs={}):
 
        # If  first epoch, remove the log file
        if epoch == 0:
            g = open(location + '/logs_' + arch + '.txt','w')
            g.close()

        # Log the progress
        msg = "{Epoch: %i} %s" % (epoch, ", ".join("%s: %f" % (k, v) for k, v in logs.items()))
        self.print_fcn(msg)
        with open(location + '/logs_' +  arch + '.txt','a') as g:
            g.write(msg + '\n')
        
        # Save the model every 5 epochs
        if epoch % 5 == 1:
             print self.model
             self.model.save(location + '/models/feature_mapper_' + arch + '.h5')            



