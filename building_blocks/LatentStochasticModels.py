import dynet as dy
import os
import pickle
import numpy as np
from DNNs import *

# This is a very basic autoencoder with no fancy regularizers
class AutoEncoder(object):
  # Accepts a list as input which indicates the number of neurons in each hidden layer
  def __init__(self, model, num_input, hidden_layer_list, num_out, act=dy.tanh):
    self.number_of_layers = len(hidden_layer_list)
    self.num_input = num_input
    self.hidden_layer_list = hidden_layer_list
    self.num_out = num_out
    num_hidden_1 = hidden_layer_list[0]
    self.model = model
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
     with open(path + '/model_hyps', 'w') as f: pickle.dump(arr, f) 
     self.model.save(path + '/model')

  @staticmethod
  def load(model, path, load_model_params=True):
      if not os.path.exists(path): raise Exception("Model "+path+" does not exist")
      with open(path+"/model_hyps", "r") as f: arr = pickle.load(f)
      model.populate(path + '/model')
      hidden_layer_list = arr[1]
      num_input = arr[0]
      num_out = arr[2]
      return AutoEncoder(model, num_input, hidden_layer_list, num_out, act=dy.tanh)


class VariationalAutoEncoder(object):

  def __init__(self, model, num_input, num_hidden, num_output, act=dy.tanh):
    self.num_input = int(num_input)
    self.num_hidden = int(num_hidden)
    self.num_out = int(num_output)
    self.model = model
    print "Loaded params"
  
   # MLP parameters
    num_hidden_q = 3
    self.W_mean_p = model.add_parameters((num_hidden_q, num_input))
    self.V_mean_p = model.add_parameters((num_hidden, num_hidden_q))
    self.b_mean_p = model.add_parameters((num_hidden_q))
    print "Loaded params for means"
   
    self.W_var_p = model.add_parameters((num_hidden_q, num_input))
    self.V_var_p = model.add_parameters((num_hidden, num_hidden_q))
    self.b_var_p = model.add_parameters((num_hidden_q))
    print "Loaded params for variances"

    self.W_out_p = model.add_parameters((num_output, num_hidden))
    self.b_out_p = model.add_parameters((num_output)) 
    print "Loaded params for output"

  def reparameterize(self, mu, logvar):
    d = mu.dim()[0][0]
    eps = dy.random_normal(d)
    std = dy.exp(logvar * 0.5)
    return mu + dy.cmult(std, eps)

  def mlp(self, x, W, V, b):
    return V * dy.tanh(W * x + b)
  
  def calc_loss_scaling(self, input_frame , output_frame):

    # Renew the computation graph
    dy.renew_cg()

    # Instantiate the params
    W_mean = dy.parameter(self.W_mean_p)
    V_mean = dy.parameter(self.V_mean_p)
    b_mean = dy.parameter(self.b_mean_p)
    W_var = dy.parameter(self.W_var_p)
    V_var = dy.parameter(self.V_var_p)
    b_var = dy.parameter(self.b_var_p)
    input_frame = dy.inputTensor(input_frame)
    output_frame = dy.inputTensor(output_frame)

    # Get the mean and diagonal log covariance from the encoder
    mu = self.mlp(input_frame , W_mean, V_mean, b_mean)
    log_var = self.mlp(input_frame , W_mean, V_mean, b_mean)

    # Compute the KL Divergence loss
    kl_loss = -0.5 * dy.sum_elems(1 + log_var - dy.pow(mu, dy.inputVector([2])) - dy.exp(log_var))

    # Reparameterize
    z = self.reparameterize(mu, log_var)

    W_out = dy.parameter(self.W_out_p)
    b_out = dy.parameter(self.b_out_p)

    # Calculate the reconstruction loss
    pred = dy.affine_transform([b_out, W_out, z])
    recons_loss = dy.l2_norm(output_frame - pred)

    return kl_loss, recons_loss



  def calc_loss_basic(self, input_frame , output_frame):

    # Renew the computation graph
    dy.renew_cg()

    # Instantiate the params
    W_mean = dy.parameter(self.W_mean_p)
    V_mean = dy.parameter(self.V_mean_p)
    b_mean = dy.parameter(self.b_mean_p)
    W_var = dy.parameter(self.W_var_p)
    V_var = dy.parameter(self.V_var_p)
    b_var = dy.parameter(self.b_var_p)    
    input_frame = dy.inputTensor(input_frame)
    output_frame = dy.inputTensor(output_frame)

    # Get the mean and diagonal log covariance from the encoder
    mu = self.mlp(input_frame , W_mean, V_mean, b_mean)
    log_var = self.mlp(input_frame , W_mean, V_mean, b_mean)

    # Compute the KL Divergence loss
    kl_loss = -0.5 * dy.sum_elems(1 + log_var - dy.pow(mu, dy.inputVector([2])) - dy.exp(log_var))

    # Reparameterize
    z = self.reparameterize(mu, log_var)

    W_out = dy.parameter(self.W_out_p)
    b_out = dy.parameter(self.b_out_p)

    # Calculate the reconstruction loss
    pred = dy.affine_transform([b_out, W_out, z])
    recons_loss = dy.l2_norm(output_frame - pred)

    return kl_loss, recons_loss

  def predict(self, input_frame):

    # Renew the computation graph
    #dy.renew_cg()

    # Instantiate the params
    W_mean = dy.parameter(self.W_mean_p)
    V_mean = dy.parameter(self.V_mean_p)
    b_mean = dy.parameter(self.b_mean_p)
    W_var = dy.parameter(self.W_var_p)
    V_var = dy.parameter(self.V_var_p)
    b_var = dy.parameter(self.b_var_p)
    input_frame = dy.inputTensor(input_frame)

    # Get the mean and diagonal log covariance from the encoder
    mu = self.mlp(input_frame , W_mean, V_mean, b_mean)
    log_var = self.mlp(input_frame , W_mean, V_mean, b_mean)

    # Compute the KL Divergence loss
    kl_loss = -0.5 * dy.sum_elems(1 + log_var - dy.pow(mu, dy.inputVector([2])) - dy.exp(log_var))

    # Reparameterize
    z = self.reparameterize(mu, log_var)

    W_out = dy.parameter(self.W_out_p)
    b_out = dy.parameter(self.b_out_p)

    # Calculate the reconstruction loss
    pred = dy.affine_transform([b_out, W_out, z])

    return pred



  # support saving:
  def param_collection(self): return self.pc

  @staticmethod
  def from_spec(spec, model):
    num_input, hidden_layer_list, num_out, act = spec
    return AutoEncoder(model, num_input, hidden_layer_list, num_out, act)

  def save(self, path):
     if not os.path.exists(path): os.makedirs(path)
     arr = [self.num_input, self.num_hidden, self.num_out]
     with open(path + '/model_hyps', 'w') as f: pickle.dump(arr, f)
     self.model.save(path + '/model')

  @staticmethod
  def load(model, path, load_model_params=True):
      if not os.path.exists(path): raise Exception("Model "+path+" does not exist")
      with open(path+"/model_hyps", "r") as f: arr = pickle.load(f)
      model.populate(path + '/model')
      hidden_layer_list = arr[1]
      num_input = arr[0]
      num_out = arr[2]
      return AutoEncoder(model, num_input, hidden_layer_list, num_out, act=dy.tanh)






class VariationalAutoEncoder_file(object):

  def __init__(self, model, num_input, num_hidden, num_output, num_latent, act=dy.tanh):
    self.num_input = int(num_input)
    self.num_hidden = int(num_hidden)
    self.num_out = int(num_output)
    self.model = model
    self.num_latent = num_latent
    print "Loaded params"

   # Label embeddings
    self.num_embed = 5
    self.lookup = model.add_lookup_parameters((self.num_out, self.num_embed))
   
   # LSTM Parameters
    self.lstm_builder = dy.LSTMBuilder(1, self.num_input, num_hidden, model)  

   # MLP parameters
    num_hidden_q = self.num_latent
    self.W_mean_p = model.add_parameters((num_hidden_q, num_hidden))
    self.V_mean_p = model.add_parameters((num_hidden, num_hidden_q))
    self.b_mean_p = model.add_parameters((num_hidden_q))
    print "Loaded params for means"
   
    self.W_var_p = model.add_parameters((num_hidden_q, num_hidden))
    self.V_var_p = model.add_parameters((num_hidden, num_hidden_q))
    self.b_var_p = model.add_parameters((num_hidden_q))
    print "Loaded params for variances"

    self.W_sm_p = model.add_parameters((num_output, num_hidden))
    self.b_sm_p = model.add_parameters((num_output)) 
    print "Loaded params for output"

  def reparameterize(self, mu, logvar):
    d = mu.dim()[0][0]
    eps = dy.random_normal(d)
    std = dy.exp(logvar * 0.5)
    return mu + dy.cmult(std, eps)

  def mlp(self, x, W, V, b):
    return V * dy.tanh(W * x + b)
  
  def calc_loss_basic(self, file , label):

    # Renew the computation graph
    dy.renew_cg()

    # Initialize LSTM
    init_state_src = self.lstm_builder.initial_state()

    # Instantiate the params
    W_mean = dy.parameter(self.W_mean_p)
    V_mean = dy.parameter(self.V_mean_p)
    b_mean = dy.parameter(self.b_mean_p)
    W_var = dy.parameter(self.W_var_p)
    V_var = dy.parameter(self.V_var_p)
    b_var = dy.parameter(self.b_var_p)
       
    input_frames = dy.inputTensor(np.loadtxt(file))
    output_label = label

    # Get the LSTM embeddings
    src_output = init_state_src.add_inputs([frame for frame in input_frames])[-1].output()

    # Get the mean and diagonal log covariance from the encoder
    mu = self.mlp(src_output , W_mean, V_mean, b_mean)
    log_var = self.mlp(src_output , W_mean, V_mean, b_mean)

    # Compute the KL Divergence loss
    kl_loss = -0.5 * dy.sum_elems(1 + log_var - dy.pow(mu, dy.inputVector([2])) - dy.exp(log_var))

    # Reparameterize
    z = self.reparameterize(mu, log_var)

    W_sm = dy.parameter(self.W_sm_p)
    b_sm = dy.parameter(self.b_sm_p)

    # Calculate the reconstruction loss
    pred = dy.affine_transform([b_sm, W_sm, z])
    label_embedding = self.lookup[label]
    #print label, label_embedding
    recons_loss = dy.pickneglogsoftmax(pred, label)

    return kl_loss, recons_loss

  def predict_label(self, file):

    # Renew the computation graph
    #dy.renew_cg()

    # Initialize LSTM
    init_state_src = self.lstm_builder.initial_state()

    # Instantiate the params
    W_mean = dy.parameter(self.W_mean_p)
    V_mean = dy.parameter(self.V_mean_p)
    b_mean = dy.parameter(self.b_mean_p)
    W_var = dy.parameter(self.W_var_p)
    V_var = dy.parameter(self.V_var_p)
    b_var = dy.parameter(self.b_var_p)

    input_frames = dy.inputTensor(np.loadtxt(file))

    src_output = init_state_src.add_inputs([frame for frame in input_frames])[-1].output()

    # Get the mean and diagonal log covariance from the encoder
    mu = self.mlp(src_output , W_mean, V_mean, b_mean)
    log_var = self.mlp(src_output , W_mean, V_mean, b_mean)

    # Compute the KL Divergence loss
    #kl_loss = -0.5 * dy.sum_elems(1 + log_var - dy.pow(mu, dy.inputVector([2])) - dy.exp(log_var))

    # Reparameterize
    z = self.reparameterize(mu, log_var)

    W_sm = dy.parameter(self.W_sm_p)
    b_sm = dy.parameter(self.b_sm_p)

    # Calculate the reconstruction loss
    pred = dy.affine_transform([b_sm, W_sm, z])
    #label_embedding = self.lookup[pred]
    #print pred.value()
    #print label_embedding.value()
    #recons_loss = dy.pickneglogsoftmax(pred, label)
    return dy.softmax(pred)
    #return kl_loss, recons_loss

  # support saving:
  def param_collection(self): return self.pc

  @staticmethod
  def from_spec(spec, model):
    num_input, hidden_layer_list, num_out, act = spec
    return AutoEncoder(model, num_input, hidden_layer_list, num_out, act)

  def save(self, path):
     if not os.path.exists(path): os.makedirs(path)
     arr = [self.num_input, self.num_hidden, self.num_out]
     with open(path + '/model_hyps', 'w') as f: pickle.dump(arr, f)
     self.model.save(path + '/model')

  @staticmethod
  def load(model, path, load_model_params=True):
      if not os.path.exists(path): raise Exception("Model "+path+" does not exist")
      with open(path+"/model_hyps", "r") as f: arr = pickle.load(f)
      model.populate(path + '/model')
      hidden_layer_list = arr[1]
      num_input = arr[0]
      num_out = arr[2]
      return AutoEncoder(model, num_input, hidden_layer_list, num_out, act=dy.tanh)



class ProbabilisticAutoEncoder_file(object):

  def __init__(self, model, num_input, num_hidden, num_output, num_latent, act=dy.tanh):
    self.num_input = int(num_input)
    self.num_hidden = int(num_hidden)
    self.num_out = int(num_output)
    self.model = model
    self.num_latent = num_latent
    print "Loaded params"

   # Label embeddings
    self.num_embed = 5
    self.lookup = model.add_lookup_parameters((self.num_out, self.num_embed))
   
   # LSTM Parameters
    self.lstm_builder = dy.LSTMBuilder(1, self.num_input, num_hidden, model)  

   # MLP parameters
    num_hidden_q = self.num_latent
    self.W_mean_p = model.add_parameters((num_hidden_q, num_hidden))
    self.V_mean_p = model.add_parameters((num_hidden, num_hidden_q))
    self.b_mean_p = model.add_parameters((num_hidden_q))
    print "Loaded params for means"
   
    self.W_var_p = model.add_parameters((num_hidden_q, num_hidden))
    self.V_var_p = model.add_parameters((num_hidden, num_hidden_q))
    self.b_var_p = model.add_parameters((num_hidden_q))
    print "Loaded params for variances"

    self.W_sm_p = model.add_parameters((num_output, num_hidden))
    self.b_sm_p = model.add_parameters((num_output)) 
    print "Loaded params for output"

  def reparameterize(self, mu, logvar):
    d = mu.dim()[0][0]
    eps = dy.random_normal(d)
    std = dy.exp(logvar * 0.5)
    return mu + dy.cmult(std, eps)

  def mlp(self, x, W, V, b):
    return V * dy.tanh(W * x + b)
  
  def calc_loss_basic(self, file , label):

    # Renew the computation graph
    dy.renew_cg()

    # Initialize LSTM
    init_state_src = self.lstm_builder.initial_state()

    # Instantiate the params
    W_mean = dy.parameter(self.W_mean_p)
    V_mean = dy.parameter(self.V_mean_p)
    b_mean = dy.parameter(self.b_mean_p)
    W_var = dy.parameter(self.W_var_p)
    V_var = dy.parameter(self.V_var_p)
    b_var = dy.parameter(self.b_var_p)
       
    input_frames = dy.inputTensor(np.loadtxt(file))
    output_label = label

    # Get the LSTM embeddings
    src_output = init_state_src.add_inputs([frame for frame in input_frames])[-1].output()

    # Get the mean and diagonal log covariance from the encoder
    mu = self.mlp(src_output , W_mean, V_mean, b_mean)
    log_var = self.mlp(src_output , W_mean, V_mean, b_mean)

    # Compute the KL Divergence loss
    kl_loss = -0.5 * dy.sum_elems(1 + log_var - dy.pow(mu, dy.inputVector([2])) - dy.exp(log_var))

    # Reparameterize
    z = self.reparameterize(mu, log_var)

    W_sm = dy.parameter(self.W_sm_p)
    b_sm = dy.parameter(self.b_sm_p)

    # Calculate the reconstruction loss
    pred = dy.affine_transform([b_sm, W_sm, z])
    label_embedding = self.lookup[label]
    #print label, label_embedding
    recons_loss = dy.pickneglogsoftmax(pred, label)

    return kl_loss, recons_loss

  def predict_label(self, file):

    # Renew the computation graph
    #dy.renew_cg()

    # Initialize LSTM
    init_state_src = self.lstm_builder.initial_state()

    # Instantiate the params
    W_mean = dy.parameter(self.W_mean_p)
    V_mean = dy.parameter(self.V_mean_p)
    b_mean = dy.parameter(self.b_mean_p)
    W_var = dy.parameter(self.W_var_p)
    V_var = dy.parameter(self.V_var_p)
    b_var = dy.parameter(self.b_var_p)

    input_frames = dy.inputTensor(np.loadtxt(file))

    src_output = init_state_src.add_inputs([frame for frame in input_frames])[-1].output()

    # Get the mean and diagonal log covariance from the encoder
    mu = self.mlp(src_output , W_mean, V_mean, b_mean)
    log_var = self.mlp(src_output , W_mean, V_mean, b_mean)

    # Compute the KL Divergence loss
    #kl_loss = -0.5 * dy.sum_elems(1 + log_var - dy.pow(mu, dy.inputVector([2])) - dy.exp(log_var))

    # Reparameterize
    z = self.reparameterize(mu, log_var)

    W_sm = dy.parameter(self.W_sm_p)
    b_sm = dy.parameter(self.b_sm_p)

    # Calculate the reconstruction loss
    pred = dy.affine_transform([b_sm, W_sm, z])
    #label_embedding = self.lookup[pred]
    #print pred.value()
    #print label_embedding.value()
    #recons_loss = dy.pickneglogsoftmax(pred, label)
    return dy.softmax(pred)
    #return kl_loss, recons_loss

  # support saving:
  def param_collection(self): return self.pc

  @staticmethod
  def from_spec(spec, model):
    num_input, hidden_layer_list, num_out, act = spec
    return AutoEncoder(model, num_input, hidden_layer_list, num_out, act)

  def save(self, path):
     if not os.path.exists(path): os.makedirs(path)
     arr = [self.num_input, self.num_hidden, self.num_out]
     with open(path + '/model_hyps', 'w') as f: pickle.dump(arr, f)
     self.model.save(path + '/model')

  @staticmethod
  def load(model, path, load_model_params=True):
      if not os.path.exists(path): raise Exception("Model "+path+" does not exist")
      with open(path+"/model_hyps", "r") as f: arr = pickle.load(f)
      model.populate(path + '/model')
      hidden_layer_list = arr[1]
      num_input = arr[0]
      num_out = arr[2]
      return AutoEncoder(model, num_input, hidden_layer_list, num_out, act=dy.tanh)




class ProbabilisticAutoEncoderBiLSTM_file(object):

  def __init__(self, model, num_input, num_hidden, num_output, num_latent, act=dy.tanh):
    self.num_input = int(num_input)
    self.num_hidden = int(num_hidden)
    self.num_out = int(num_output)
    self.model = model
    self.num_latent = num_latent
    print "Loaded params"

   # Label embeddings
    self.num_embed = 5
    self.lookup = model.add_lookup_parameters((self.num_out, self.num_embed))
   
   # LSTM Parameters
    self.lstm_fwd_builder = dy.LSTMBuilder(2, self.num_input, num_hidden, model)  
    self.lstm_bwd_builder = dy.LSTMBuilder(1, self.num_input, num_hidden, model)

   # MLP parameters
    num_hidden_q = self.num_latent 
    self.W_mean_p = model.add_parameters((num_hidden_q, num_hidden*2))
    self.V_mean_p = model.add_parameters((2*num_hidden, num_hidden_q))
    self.b_mean_p = model.add_parameters((num_hidden_q))
    print "Loaded params for means"
   
    self.W_var_p = model.add_parameters((num_hidden_q, num_hidden*2))
    self.V_var_p = model.add_parameters((num_hidden*2, num_hidden_q))
    self.b_var_p = model.add_parameters((num_hidden_q))
    print "Loaded params for variances"

    self.W_sm_p = model.add_parameters((num_output, num_hidden*2))
    self.b_sm_p = model.add_parameters((num_output)) 
    print "Loaded params for output"

  def reparameterize(self, mu, logvar):
    d = mu.dim()[0][0]
    eps = dy.random_normal(d)
    std = dy.exp(logvar * 0.5)
    return mu + dy.cmult(std, eps)

  def mlp(self, x, W, V, b):
    return V * dy.tanh(W * x + b)
  
  def calc_loss_basic(self, file , label):

    # Renew the computation graph
    dy.renew_cg()

    # Initialize LSTM
    init_state_fwd = self.lstm_fwd_builder.initial_state()
    init_state_bwd = self.lstm_bwd_builder.initial_state()

    # Instantiate the params
    W_mean = dy.parameter(self.W_mean_p)
    V_mean = dy.parameter(self.V_mean_p)
    b_mean = dy.parameter(self.b_mean_p)
    W_var = dy.parameter(self.W_var_p)
    V_var = dy.parameter(self.V_var_p)
    b_var = dy.parameter(self.b_var_p)
       
    input_frames = dy.inputTensor(np.loadtxt(file))
    input_frames_reverse = dy.inputTensor(np.flipud(np.loadtxt(file)))
    output_label = label

    # Get the LSTM embeddings
    fwd_output = init_state_fwd.add_inputs([frame for frame in input_frames])[-1].output()
    bwd_output = init_state_bwd.add_inputs([frame for frame in input_frames_reverse])[-1].output()    

    # Concatenate
    bilstm_embeddings = dy.concatenate([fwd_output, bwd_output])

    # Get the mean and diagonal log covariance from the encoder
    mu = self.mlp(bilstm_embeddings , W_mean, V_mean, b_mean)
    log_var = self.mlp(bilstm_embeddings , W_mean, V_mean, b_mean)

    # Compute the KL Divergence loss
    kl_loss = -0.5 * dy.sum_elems(1 + log_var - dy.pow(mu, dy.inputVector([2])) - dy.exp(log_var))

    # Reparameterize
    z = self.reparameterize(mu, log_var)

    W_sm = dy.parameter(self.W_sm_p)
    b_sm = dy.parameter(self.b_sm_p)

    # Calculate the reconstruction loss
    pred = dy.affine_transform([b_sm, W_sm, z])
    label_embedding = self.lookup[label]
    #print label, label_embedding
    recons_loss = dy.pickneglogsoftmax(pred, label)

    return kl_loss, recons_loss

  def predict_label(self, file):

    # Initialize LSTM
    init_state_fwd = self.lstm_fwd_builder.initial_state()
    init_state_bwd = self.lstm_bwd_builder.initial_state()

    # Instantiate the params
    W_mean = dy.parameter(self.W_mean_p)
    V_mean = dy.parameter(self.V_mean_p)
    b_mean = dy.parameter(self.b_mean_p)
    W_var = dy.parameter(self.W_var_p)
    V_var = dy.parameter(self.V_var_p)
    b_var = dy.parameter(self.b_var_p)

    input_frames = dy.inputTensor(np.loadtxt(file))
    input_frames_reverse = dy.inputTensor(np.flipud(np.loadtxt(file)))

    # Get the LSTM embeddings
    fwd_output = init_state_fwd.add_inputs([frame for frame in input_frames])[-1].output()
    #input_frames.reverse()
    bwd_output = init_state_bwd.add_inputs([frame for frame in input_frames_reverse])[-1].output()

    # Concatenate
    bilstm_embeddings = dy.concatenate([fwd_output, bwd_output])


    # Get the mean and diagonal log covariance from the encoder
    mu = self.mlp(bilstm_embeddings , W_mean, V_mean, b_mean)
    log_var = self.mlp(bilstm_embeddings , W_mean, V_mean, b_mean)

    # Compute the KL Divergence loss
    kl_loss = -0.5 * dy.sum_elems(1 + log_var - dy.pow(mu, dy.inputVector([2])) - dy.exp(log_var))

    # Reparameterize
    z = self.reparameterize(mu, log_var)

    W_sm = dy.parameter(self.W_sm_p)
    b_sm = dy.parameter(self.b_sm_p)
    pred = dy.affine_transform([b_sm, W_sm, z])
    return dy.softmax(pred)

  # support saving:
  def param_collection(self): return self.pc

  @staticmethod
  def from_spec(spec, model):
    num_input, hidden_layer_list, num_out, act = spec
    return AutoEncoder(model, num_input, hidden_layer_list, num_out, act)

  def save(self, path):
     if not os.path.exists(path): os.makedirs(path)
     arr = [self.num_input, self.num_hidden, self.num_out]
     with open(path + '/model_hyps', 'w') as f: pickle.dump(arr, f)
     self.model.save(path + '/model')

  @staticmethod
  def load(model, path, load_model_params=True):
      if not os.path.exists(path): raise Exception("Model "+path+" does not exist")
      with open(path+"/model_hyps", "r") as f: arr = pickle.load(f)
      model.populate(path + '/model')
      hidden_layer_list = arr[1]
      num_input = arr[0]
      num_out = arr[2]
      return AutoEncoder(model, num_input, hidden_layer_list, num_out, act=dy.tanh)




class EncoderBiLSTM_file(object):

  def __init__(self, model, num_input, num_hidden, num_output, num_latent, act=dy.tanh):
    self.num_input = int(num_input)
    self.num_hidden = int(num_hidden)
    self.num_out = int(num_output)
    self.model = model
    self.num_latent = num_latent
    print "Loaded params"

   # Label embeddings
    self.num_embed = 5
    self.lookup = model.add_lookup_parameters((self.num_out, self.num_embed))
   
   # LSTM Parameters
    self.lstm_fwd_builder = dy.LSTMBuilder(1, self.num_input, num_hidden, model)  
    self.lstm_bwd_builder = dy.LSTMBuilder(1, self.num_input, num_hidden, model)

   # MLP parameters
    num_hidden_q = num_latent
    self.W_mean_p = model.add_parameters((num_hidden_q, num_hidden*2))
    self.V_mean_p = model.add_parameters((2*num_hidden, num_hidden_q))
    self.b_mean_p = model.add_parameters((num_hidden_q))
    print "Loaded params for means"
   
    self.W_var_p = model.add_parameters((num_hidden_q, num_hidden*2))
    self.V_var_p = model.add_parameters((num_hidden*2, num_hidden_q))
    self.b_var_p = model.add_parameters((num_hidden_q))
    print "Loaded params for variances"

    self.W_sm_p = model.add_parameters((num_output, num_hidden*2))
    self.b_sm_p = model.add_parameters((num_output)) 
    print "Loaded params for output"

  def reparameterize(self, mu, logvar):
    d = mu.dim()[0][0]
    eps = dy.random_normal(d)
    std = dy.exp(logvar * 0.5)
    return mu + dy.cmult(std, eps)

  def mlp(self, x, W, V, b):
    return V * dy.tanh(W * x + b)
  
  def calc_loss_basic(self, file , label):

    # Renew the computation graph
    dy.renew_cg()

    # Initialize LSTM
    init_state_fwd = self.lstm_fwd_builder.initial_state()
    init_state_bwd = self.lstm_bwd_builder.initial_state()

    # Instantiate the params
    W_mean = dy.parameter(self.W_mean_p)
    V_mean = dy.parameter(self.V_mean_p)
    b_mean = dy.parameter(self.b_mean_p)
    W_var = dy.parameter(self.W_var_p)
    V_var = dy.parameter(self.V_var_p)
    b_var = dy.parameter(self.b_var_p)
       
    input_frames = dy.inputTensor(np.loadtxt(file))
    input_frames_reverse = dy.inputTensor(np.flipud(np.loadtxt(file)))
    output_label = label

    # Get the LSTM embeddings
    fwd_output = init_state_fwd.add_inputs([frame for frame in input_frames])[-1].output()
    #input_frames.reverse()
    bwd_output = init_state_bwd.add_inputs([frame for frame in input_frames_reverse])[-1].output()    

    # Concatenate
    bilstm_embeddings = dy.concatenate([fwd_output, bwd_output])
    

    W_sm = dy.parameter(self.W_sm_p)
    b_sm = dy.parameter(self.b_sm_p)

    # Calculate the reconstruction loss
    pred = dy.affine_transform([b_sm, W_sm, bilstm_embeddings])
    label_embedding = self.lookup[label]
    #print label, label_embedding
    recons_loss = dy.pickneglogsoftmax(pred, label)

    return recons_loss

  def predict_label(self, file):

    # Initialize LSTM
    init_state_fwd = self.lstm_fwd_builder.initial_state()
    init_state_bwd = self.lstm_bwd_builder.initial_state()

    # Instantiate the params
    W_mean = dy.parameter(self.W_mean_p)
    V_mean = dy.parameter(self.V_mean_p)
    b_mean = dy.parameter(self.b_mean_p)
    W_var = dy.parameter(self.W_var_p)
    V_var = dy.parameter(self.V_var_p)
    b_var = dy.parameter(self.b_var_p)

    input_frames = dy.inputTensor(np.loadtxt(file))
    input_frames_reverse = dy.inputTensor(np.flipud(np.loadtxt(file)))

    # Get the LSTM embeddings
    fwd_output = init_state_fwd.add_inputs([frame for frame in input_frames])[-1].output()
    #input_frames.reverse()
    bwd_output = init_state_bwd.add_inputs([frame for frame in input_frames_reverse])[-1].output()

    # Concatenate
    bilstm_embeddings = dy.concatenate([fwd_output, bwd_output])
    

    W_sm = dy.parameter(self.W_sm_p)
    b_sm = dy.parameter(self.b_sm_p)
 
    pred = dy.affine_transform([b_sm, W_sm, bilstm_embeddings])
    return dy.softmax(pred)

  def predict_label_truncate(self, file):

    # Initialize LSTM
    init_state_fwd = self.lstm_fwd_builder.initial_state()
    init_state_bwd = self.lstm_bwd_builder.initial_state()

    # Instantiate the params
    W_mean = dy.parameter(self.W_mean_p)
    V_mean = dy.parameter(self.V_mean_p)
    b_mean = dy.parameter(self.b_mean_p)
    W_var = dy.parameter(self.W_var_p)
    V_var = dy.parameter(self.V_var_p)
    b_var = dy.parameter(self.b_var_p)

    input_frames = dy.inputTensor(np.loadtxt(file))
    input_frames_reverse = dy.inputTensor(np.flipud(np.loadtxt(file)))

    # Get the LSTM embeddings
    fwd_output = init_state_fwd.add_inputs([frame for frame in input_frames])[-1].output()
    #input_frames.reverse()
    bwd_output = init_state_bwd.add_inputs([frame for frame in input_frames_reverse])[-1].output()

    # Concatenate
    bilstm_embeddings = dy.concatenate([fwd_output, bwd_output])


    W_sm = dy.parameter(self.W_sm_p)
    b_sm = dy.parameter(self.b_sm_p)

    pred = dy.affine_transform([b_sm, W_sm, bilstm_embeddings])
    return dy.softmax(pred)


  # support saving:
  def param_collection(self): return self.pc

  @staticmethod
  def from_spec(spec, model):
    num_input, hidden_layer_list, num_out, act = spec
    return AutoEncoder(model, num_input, hidden_layer_list, num_out, act)

  def save(self, path):
     if not os.path.exists(path): os.makedirs(path)
     arr = [self.num_input, self.num_hidden, self.num_out]
     with open(path + '/model_hyps', 'w') as f: pickle.dump(arr, f)
     self.model.save(path + '/model')

  @staticmethod
  def load(model, path, load_model_params=True):
      if not os.path.exists(path): raise Exception("Model "+path+" does not exist")
      with open(path+"/model_hyps", "r") as f: arr = pickle.load(f)
      model.populate(path + '/model')
      hidden_layer_list = arr[1]
      num_input = arr[0]
      num_out = arr[2]
      return AutoEncoder(model, num_input, hidden_layer_list, num_out, act=dy.tanh)



class EncoderUniLSTM_file(object):

  def __init__(self, model, num_input, num_hidden, num_output, num_latent, act=dy.tanh):
    self.num_input = int(num_input)
    self.num_hidden = int(num_hidden)
    self.num_out = int(num_output)
    self.model = model
    self.num_latent = num_latent
    print "Loaded params"

   # Label embeddings
    self.num_embed = 5
    self.lookup = model.add_lookup_parameters((self.num_out, self.num_embed))
   
   # LSTM Parameters
    self.lstm_fwd_builder = dy.LSTMBuilder(1, self.num_input, num_hidden, model)  

   # MLP parameters
    num_hidden_q = self.num_latent
    self.W_mean_p = model.add_parameters((num_hidden_q, num_hidden))
    self.V_mean_p = model.add_parameters((num_hidden, num_hidden_q))
    self.b_mean_p = model.add_parameters((num_hidden_q))
    print "Loaded params for means"
   
    self.W_var_p = model.add_parameters((num_hidden_q, num_hidden))
    self.V_var_p = model.add_parameters((num_hidden, num_hidden_q))
    self.b_var_p = model.add_parameters((num_hidden_q))
    print "Loaded params for variances"

    self.W_sm_p = model.add_parameters((num_output, num_hidden))
    self.b_sm_p = model.add_parameters((num_output)) 
    print "Loaded params for output"

  def reparameterize(self, mu, logvar):
    d = mu.dim()[0][0]
    eps = dy.random_normal(d)
    std = dy.exp(logvar * 0.5)
    return mu + dy.cmult(std, eps)

  def mlp(self, x, W, V, b):
    return V * dy.tanh(W * x + b)
  
  def calc_loss_basic(self, file , label):

    # Renew the computation graph
    dy.renew_cg()

    # Initialize LSTM
    init_state_fwd = self.lstm_fwd_builder.initial_state()

    # Instantiate the params
    W_mean = dy.parameter(self.W_mean_p)
    V_mean = dy.parameter(self.V_mean_p)
    b_mean = dy.parameter(self.b_mean_p)
    W_var = dy.parameter(self.W_var_p)
    V_var = dy.parameter(self.V_var_p)
    b_var = dy.parameter(self.b_var_p)
       
    input_frames = dy.inputTensor(np.loadtxt(file))
    input_frames_reverse = dy.inputTensor(np.flipud(np.loadtxt(file)))
    output_label = label

    # Get the LSTM embeddings
    fwd_output = init_state_fwd.add_inputs([frame for frame in input_frames])[-1].output()

    W_sm = dy.parameter(self.W_sm_p)
    b_sm = dy.parameter(self.b_sm_p)

    # Calculate the reconstruction loss
    pred = dy.affine_transform([b_sm, W_sm, fwd_output])
    recons_loss = dy.pickneglogsoftmax(pred, label)

    return recons_loss

  def predict_label(self, file):

    # Initialize LSTM
    init_state_fwd = self.lstm_fwd_builder.initial_state()

    # Instantiate the params
    W_mean = dy.parameter(self.W_mean_p)
    V_mean = dy.parameter(self.V_mean_p)
    b_mean = dy.parameter(self.b_mean_p)
    W_var = dy.parameter(self.W_var_p)
    V_var = dy.parameter(self.V_var_p)
    b_var = dy.parameter(self.b_var_p)

    input_frames = dy.inputTensor(np.loadtxt(file))

    # Get the LSTM embeddings
    fwd_output = init_state_fwd.add_inputs([frame for frame in input_frames])[-1].output()

    W_sm = dy.parameter(self.W_sm_p)
    b_sm = dy.parameter(self.b_sm_p)
 
    pred = dy.affine_transform([b_sm, W_sm, fwd_output])
    return dy.softmax(pred)

  # support saving:
  def param_collection(self): return self.pc

  @staticmethod
  def from_spec(spec, model):
    num_input, hidden_layer_list, num_out, act = spec
    return AutoEncoder(model, num_input, hidden_layer_list, num_out, act)

  def save(self, path):
     if not os.path.exists(path): os.makedirs(path)
     arr = [self.num_input, self.num_hidden, self.num_out]
     with open(path + '/model_hyps', 'w') as f: pickle.dump(arr, f)
     self.model.save(path + '/model')

  @staticmethod
  def load(model, path, load_model_params=True):
      if not os.path.exists(path): raise Exception("Model "+path+" does not exist")
      with open(path+"/model_hyps", "r") as f: arr = pickle.load(f)
      model.populate(path + '/model')
      hidden_layer_list = arr[1]
      num_input = arr[0]
      num_out = arr[2]
      return AutoEncoder(model, num_input, hidden_layer_list, num_out, act=dy.tanh)



class ProbabilisticEncoderDecoderBiLSTM_file(object):

  def __init__(self, model, num_input, num_hidden, num_output, num_latent, act=dy.tanh):
    self.num_input = int(num_input)
    self.num_hidden = int(num_hidden)
    self.num_out = int(num_output)
    self.model = model
    self.num_latent = num_latent
    print "Loaded params"

   # Label embeddings
    self.num_embed = 5
    self.lookup = model.add_lookup_parameters((self.num_out, self.num_embed))
   
   # LSTM Parameters
    self.lstm_fwd_builder = dy.LSTMBuilder(2, self.num_input, num_hidden, model)  
    self.lstm_bwd_builder = dy.LSTMBuilder(1, self.num_input, num_hidden, model)

   # MLP parameters
    num_hidden_q = self.num_latent 
    self.W_mean_p = model.add_parameters((num_hidden_q, num_hidden*2))
    self.V_mean_p = model.add_parameters((2*num_hidden, num_hidden_q))
    self.b_mean_p = model.add_parameters((num_hidden_q))
    print "Loaded params for means"
   
    self.W_var_p = model.add_parameters((num_hidden_q, num_hidden*2))
    self.V_var_p = model.add_parameters((num_hidden*2, num_hidden_q))
    self.b_var_p = model.add_parameters((num_hidden_q))
    print "Loaded params for variances"

    self.W_sm_p = model.add_parameters((num_output, num_hidden*4))
    self.b_sm_p = model.add_parameters((num_output)) 
    print "Loaded params for output"

  def reparameterize(self, mu, logvar):
    d = mu.dim()[0][0]
    eps = dy.random_normal(d)
    std = dy.exp(logvar * 0.5)
    return mu + dy.cmult(std, eps)

  def mlp(self, x, W, V, b):
    return V * dy.tanh(W * x + b)
  
  def calc_loss_basic(self, file , label):

    # Renew the computation graph
    dy.renew_cg()

    # Initialize LSTM
    init_state_fwd = self.lstm_fwd_builder.initial_state()
    init_state_bwd = self.lstm_bwd_builder.initial_state()

    # Instantiate the params
    W_mean = dy.parameter(self.W_mean_p)
    V_mean = dy.parameter(self.V_mean_p)
    b_mean = dy.parameter(self.b_mean_p)
    W_var = dy.parameter(self.W_var_p)
    V_var = dy.parameter(self.V_var_p)
    b_var = dy.parameter(self.b_var_p)
       
    input_frames = dy.inputTensor(np.loadtxt(file))
    input_frames_reverse = dy.inputTensor(np.flipud(np.loadtxt(file)))
    output_label = label

    # Get the LSTM embeddings
    fwd_output = init_state_fwd.add_inputs([frame for frame in input_frames])[-1].output()
    bwd_output = init_state_bwd.add_inputs([frame for frame in input_frames_reverse])[-1].output()    

    # Concatenate
    bilstm_embeddings = dy.concatenate([fwd_output, bwd_output])

    # Get the mean and diagonal log covariance from the encoder
    mu = self.mlp(bilstm_embeddings , W_mean, V_mean, b_mean)
    log_var = self.mlp(bilstm_embeddings , W_mean, V_mean, b_mean)

    # Compute the KL Divergence loss
    kl_loss = -0.5 * dy.sum_elems(1 + log_var - dy.pow(mu, dy.inputVector([2])) - dy.exp(log_var))

    # Reparameterize
    z = self.reparameterize(mu, log_var)

    W_sm = dy.parameter(self.W_sm_p)
    b_sm = dy.parameter(self.b_sm_p)

    Z = dy.concatenate([bilstm_embeddings, z])
    # Calculate the reconstruction loss
    pred = dy.affine_transform([b_sm, W_sm, Z])
    label_embedding = self.lookup[label]
    #print label, label_embedding
    recons_loss = dy.pickneglogsoftmax(pred, label)

    return kl_loss, recons_loss

  def predict_label(self, file):

    # Initialize LSTM
    init_state_fwd = self.lstm_fwd_builder.initial_state()
    init_state_bwd = self.lstm_bwd_builder.initial_state()

    # Instantiate the params
    W_mean = dy.parameter(self.W_mean_p)
    V_mean = dy.parameter(self.V_mean_p)
    b_mean = dy.parameter(self.b_mean_p)
    W_var = dy.parameter(self.W_var_p)
    V_var = dy.parameter(self.V_var_p)
    b_var = dy.parameter(self.b_var_p)

    input_frames = dy.inputTensor(np.loadtxt(file))
    input_frames_reverse = dy.inputTensor(np.flipud(np.loadtxt(file)))

    # Get the LSTM embeddings
    fwd_output = init_state_fwd.add_inputs([frame for frame in input_frames])[-1].output()
    #input_frames.reverse()
    bwd_output = init_state_bwd.add_inputs([frame for frame in input_frames_reverse])[-1].output()

    # Concatenate
    bilstm_embeddings = dy.concatenate([fwd_output, bwd_output])


    # Get the mean and diagonal log covariance from the encoder
    mu = self.mlp(bilstm_embeddings , W_mean, V_mean, b_mean)
    log_var = self.mlp(bilstm_embeddings , W_mean, V_mean, b_mean)

    # Compute the KL Divergence loss
    kl_loss = -0.5 * dy.sum_elems(1 + log_var - dy.pow(mu, dy.inputVector([2])) - dy.exp(log_var))

    # Reparameterize
    z = self.reparameterize(mu, log_var)
    Z = dy.concatenate([bilstm_embeddings, z])
    W_sm = dy.parameter(self.W_sm_p)
    b_sm = dy.parameter(self.b_sm_p)
   
    pred = dy.affine_transform([b_sm, W_sm, Z])
    return dy.softmax(pred)

  # support saving:
  def param_collection(self): return self.pc

  @staticmethod
  def from_spec(spec, model):
    num_input, hidden_layer_list, num_out, act = spec
    return AutoEncoder(model, num_input, hidden_layer_list, num_out, act)

  def save(self, path):
     if not os.path.exists(path): os.makedirs(path)
     arr = [self.num_input, self.num_hidden, self.num_out]
     with open(path + '/model_hyps', 'w') as f: pickle.dump(arr, f)
     self.model.save(path + '/model')

  @staticmethod
  def load(model, path, load_model_params=True):
      if not os.path.exists(path): raise Exception("Model "+path+" does not exist")
      with open(path+"/model_hyps", "r") as f: arr = pickle.load(f)
      model.populate(path + '/model')
      hidden_layer_list = arr[1]
      num_input = arr[0]
      num_out = arr[2]
      return AutoEncoder(model, num_input, hidden_layer_list, num_out, act=dy.tanh)



class VariationalAutoEncoder_coeffs_file(object):

  def __init__(self, model, num_input, num_hidden, num_output, num_latent, act=dy.tanh):
    self.num_input = int(num_input)
    self.num_hidden = int(num_hidden)
    self.num_out = int(num_output)
    self.model = model
    self.num_latent = num_latent
    print "Loaded params"

   # MLP parameters
    num_hidden_q = self.num_latent
    self.W_mean_p = model.add_parameters((num_hidden_q, num_hidden))
    self.V_mean_p = model.add_parameters((num_hidden, num_hidden_q))
    self.b_mean_p = model.add_parameters((num_hidden_q))
    print "Loaded params for means"
   
    self.W_var_p = model.add_parameters((num_hidden_q, num_hidden))
    self.V_var_p = model.add_parameters((num_hidden, num_hidden_q))
    self.b_var_p = model.add_parameters((num_hidden_q))
    print "Loaded params for variances"
 
    self.W_in = model.add_parameters((num_hidden, num_input))
    self.b_in = model.add_parameters((num_hidden))
    self.W_hid = model.add_parameters((num_hidden, num_hidden))
    self.b_hid = model.add_parameters((num_hidden))
    self.W_out = model.add_parameters((num_output, num_hidden))
    self.b_out = model.add_parameters((num_output)) 
    print "Loaded params for output"

  def reparameterize(self, mu, logvar):
    d = mu.dim()[0][0]
    eps = dy.random_normal(d)
    std = dy.exp(logvar * 0.5)
    return mu + dy.cmult(std, eps)

  def mlp(self, x, W, V, b):
    return V * dy.tanh(W * x + b)
  
  def calculate_loss(self, src_file, tgt_file):

    ## Renew the computation graph
    dy.renew_cg()

    ## Instantiate the params
    W_mean = dy.parameter(self.W_mean_p)
    V_mean = dy.parameter(self.V_mean_p)
    b_mean = dy.parameter(self.b_mean_p)
    W_var = dy.parameter(self.W_var_p)
    V_var = dy.parameter(self.V_var_p)
    b_var = dy.parameter(self.b_var_p)

    W_frame = dy.parameter(self.W_in)
    b_frame = dy.parameter(self.b_in) 
    W_frame1 = dy.parameter(self.W_out)
    b_frame1 = dy.parameter(self.b_out)
    W_out = dy.parameter(self.W_out)
    b_out = dy.parameter(self.b_out)
  
    input_frames = dy.inputTensor(np.loadtxt(src_file,usecols=range(1,51)))
    output_frames = dy.inputTensor(np.loadtxt(tgt_file,usecols=range(1,51))) 
    
    KL_loss = []
    RECON_loss = []

    ## Loop through the frames and output mean and variance
    for (input_frame, output_frame) in zip(input_frames, output_frames):

      # Get output           
      src_output = dy.tanh(W_frame * input_frame + b_frame) #+ b_frame1 )

      # Get the mean and diagonal log covariance
      mu = self.mlp(src_output , W_mean, V_mean, b_mean)
      log_var = self.mlp(src_output , W_mean, V_mean, b_mean)

      # Compute the KL Divergence loss
      KL_loss.append(-0.5 * dy.sum_elems(1 + log_var - dy.pow(mu, dy.inputVector([2])) - dy.exp(log_var)))

      # Reparameterize
      z = self.reparameterize(mu, log_var)

      # Calculate the reconstruction loss
      pred = dy.affine_transform([b_out, W_out, z])
      RECON_loss.append(dy.l2_norm(output_frame - pred))

    return dy.esum(RECON_loss)
    return dy.esum(KL_loss), dy.esum(RECON_loss)


  def calculate_loss_frame(self, frame):

    ## Renew the computation graph
    dy.renew_cg()

    ## Instantiate the params
    W_mean = dy.parameter(self.W_mean_p)
    V_mean = dy.parameter(self.V_mean_p)
    b_mean = dy.parameter(self.b_mean_p)
    W_var = dy.parameter(self.W_var_p)
    V_var = dy.parameter(self.V_var_p)
    b_var = dy.parameter(self.b_var_p)

    W_frame = dy.parameter(self.W_in)
    b_frame = dy.parameter(self.b_in)
    W_frame1 = dy.parameter(self.W_hid)
    b_frame1 = dy.parameter(self.b_hid)
    W_out = dy.parameter(self.W_out)
    b_out = dy.parameter(self.b_out)
    frame = dy.inputTensor(frame)

    # Get output           
    src_output =  dy.tanh(W_frame1 * dy.tanh(W_frame * frame + b_frame) + b_frame1 ) 

    # Get the mean and diagonal log covariance
    mu = self.mlp(src_output , W_mean, V_mean, b_mean)
    log_var = self.mlp(src_output , W_mean, V_mean, b_mean)

    # Compute the KL Divergence loss
    KL_loss = -0.5 * dy.sum_elems(1 + log_var - dy.pow(mu, dy.inputVector([2])) - dy.exp(log_var))

    # Reparameterize
    z = self.reparameterize(mu, log_var)

    # Calculate the reconstruction loss
    pred = dy.affine_transform([b_out, W_out, z])
    RECON_loss = dy.l2_norm(frame - pred)

    return KL_loss,  RECON_loss

  def predict_frame(self, frame):

    ## Instantiate the params
    W_mean = dy.parameter(self.W_mean_p)
    V_mean = dy.parameter(self.V_mean_p)
    b_mean = dy.parameter(self.b_mean_p)
    W_var = dy.parameter(self.W_var_p)
    V_var = dy.parameter(self.V_var_p)
    b_var = dy.parameter(self.b_var_p)

    W_frame = dy.parameter(self.W_in)
    b_frame = dy.parameter(self.b_in)
    W_frame1 = dy.parameter(self.W_hid)
    b_frame1 = dy.parameter(self.b_hid)
    W_out = dy.parameter(self.W_out)
    b_out = dy.parameter(self.b_out)
    frame = dy.inputTensor(frame)

    # Get output           
    src_output =  dy.tanh(W_frame1 * dy.tanh(W_frame * frame + b_frame) + b_frame1 )

    # Get the mean and diagonal log covariance
    mu = self.mlp(src_output , W_mean, V_mean, b_mean)
    log_var = self.mlp(src_output , W_mean, V_mean, b_mean)

    # Compute the KL Divergence loss
    KL_loss = -0.5 * dy.sum_elems(1 + log_var - dy.pow(mu, dy.inputVector([2])) - dy.exp(log_var))

    # Reparameterize
    z = self.reparameterize(mu, log_var)

    # Calculate the reconstruction loss
    pred = dy.affine_transform([b_out, W_out, z])
    return pred



  # support saving:
  def param_collection(self): return self.pc

  @staticmethod
  def from_spec(spec, model):
    num_input, hidden_layer_list, num_out, act = spec
    return AutoEncoder(model, num_input, hidden_layer_list, num_out, act)

  def save(self, path):
     if not os.path.exists(path): os.makedirs(path)
     arr = [self.num_input, self.num_hidden, self.num_out]
     with open(path + '/model_hyps', 'w') as f: pickle.dump(arr, f)
     self.model.save(path + '/model')

  @staticmethod
  def load(model, path, load_model_params=True):
      if not os.path.exists(path): raise Exception("Model "+path+" does not exist")
      with open(path+"/model_hyps", "r") as f: arr = pickle.load(f)
      model.populate(path + '/model')
      hidden_layer_list = arr[1]
      num_input = arr[0]
      num_out = arr[2]
      return AutoEncoder(model, num_input, hidden_layer_list, num_out, act=dy.tanh)

#VAE_predict_f0_from_frames

# This class uses frame information to predict the f0 value
class VAE_predict_f0_from_frames(object):
  def __init__(self, model, num_input, num_hidden, num_output, num_latent, act=dy.tanh):
    self.num_input = int(num_input)
    self.num_hidden = int(num_hidden)
    self.num_out = int(num_output)
    self.model = model
    self.num_latent = num_latent
    print "Loaded params"

   # MLP parameters
    num_hidden_q = self.num_latent
    self.W_mean_p = model.add_parameters((num_hidden_q, num_hidden))
    self.V_mean_p = model.add_parameters((num_hidden, num_hidden_q))
    self.b_mean_p = model.add_parameters((num_hidden_q))
    print "Loaded params for means"
   
    self.W_var_p = model.add_parameters((num_hidden_q, num_hidden))
    self.V_var_p = model.add_parameters((num_hidden, num_hidden_q))
    self.b_var_p = model.add_parameters((num_hidden_q))
    print "Loaded params for variances"
 
    self.W_in = model.add_parameters((num_hidden, num_input-1))
    self.b_in = model.add_parameters((num_hidden))
    self.W_hid = model.add_parameters((num_hidden, num_hidden))
    self.b_hid = model.add_parameters((num_hidden))
    self.W_out = model.add_parameters((num_output, num_hidden))
    self.b_out = model.add_parameters((num_output)) 
    self.W_f0 = model.add_parameters((num_hidden, num_hidden))
    self.b_f0 = model.add_parameters((num_hidden))
    print "Loaded params for output"

  def reparameterize(self, mu, logvar):
    d = mu.dim()[0][0]
    eps = dy.random_normal(d)
    std = dy.exp(logvar * 0.5)
    return mu + dy.cmult(std, eps)

  def mlp(self, x, W, V, b):
    return V * dy.tanh(W * x + b)
  
  def calculate_loss_frame(self, frame):

    ## Renew the computation graph
    dy.renew_cg()

    ## Instantiate the params
    W_mean = dy.parameter(self.W_mean_p)
    V_mean = dy.parameter(self.V_mean_p)
    b_mean = dy.parameter(self.b_mean_p)
    W_var = dy.parameter(self.W_var_p)
    V_var = dy.parameter(self.V_var_p)
    b_var = dy.parameter(self.b_var_p)

    W_frame = dy.parameter(self.W_in)
    b_frame = dy.parameter(self.b_in)
    W_f0 = dy.parameter(self.W_f0)
    b_f0 = dy.parameter(self.b_f0)
    W_out = dy.parameter(self.W_out)
    b_out = dy.parameter(self.b_out)
    f0 = dy.inputTensor(np.asarray(frame[0]))
    frame = dy.inputTensor(frame[1:])

    # Get output           
    src_output =  dy.tanh(W_f0 * dy.tanh(W_frame * frame + b_frame) + b_f0 ) 

    # Get the mean and diagonal log covariance
    mu = self.mlp(src_output , W_mean, V_mean, b_mean)
    log_var = self.mlp(src_output , W_mean, V_mean, b_mean)

    # Compute the KL Divergence loss
    KL_loss = -0.5 * dy.sum_elems(1 + log_var - dy.pow(mu, dy.inputVector([2])) - dy.exp(log_var))

    # Reparameterize
    z = self.reparameterize(mu, log_var)

    # Calculate the reconstruction loss
    pred = dy.tanh(dy.affine_transform([b_out, W_out, z]))
    RECON_loss = dy.l2_norm(f0 - pred)

    return KL_loss,  RECON_loss

  def predict_frame(self, frame):

    ## Instantiate the params
    W_mean = dy.parameter(self.W_mean_p)
    V_mean = dy.parameter(self.V_mean_p)
    b_mean = dy.parameter(self.b_mean_p)
    W_var = dy.parameter(self.W_var_p)
    V_var = dy.parameter(self.V_var_p)
    b_var = dy.parameter(self.b_var_p)

    W_frame = dy.parameter(self.W_in)
    b_frame = dy.parameter(self.b_in)
    W_f0 = dy.parameter(self.W_f0)
    b_f0 = dy.parameter(self.b_f0)
    W_out = dy.parameter(self.W_out)
    b_out = dy.parameter(self.b_out)
    f0 = dy.inputTensor(np.asarray(frame[0]))
    frame = dy.inputTensor(frame[1:])

    # Get output           
    src_output =  dy.tanh(W_f0 * dy.tanh(W_frame * frame + b_frame) + b_f0 )

    # Get the mean and diagonal log covariance
    mu = self.mlp(src_output , W_mean, V_mean, b_mean)
    log_var = self.mlp(src_output , W_mean, V_mean, b_mean)

    # Compute the KL Divergence loss
    KL_loss = -0.5 * dy.sum_elems(1 + log_var - dy.pow(mu, dy.inputVector([2])) - dy.exp(log_var))

    # Reparameterize
    z = self.reparameterize(mu, log_var)

    # Calculate the reconstruction loss
    pred = dy.tanh(dy.affine_transform([b_out, W_out, z]))
    return pred, frame

# This class uses frame information to predict the frame value
class VAE_predict_frame_from_contextframes(object):
  def __init__(self, model, num_input, num_hidden, num_output, num_latent, act=dy.tanh):
    self.num_input = int(num_input)
    self.num_hidden = int(num_hidden)
    self.num_out = int(num_output)
    self.model = model
    self.num_latent = num_latent
    print "Loaded params"

    # MLP parameters
    num_hidden_q = self.num_latent
    self.W_mean_p = model.add_parameters((num_hidden_q, num_hidden))
    self.V_mean_p = model.add_parameters((num_hidden, num_hidden_q))
    self.b_mean_p = model.add_parameters((num_hidden_q))
    print "Loaded params for means"
   
    self.W_var_p = model.add_parameters((num_hidden_q, num_hidden))
    self.V_var_p = model.add_parameters((num_hidden, num_hidden_q))
    self.b_var_p = model.add_parameters((num_hidden_q))
    print "Loaded params for variances"
 
    self.W_in = model.add_parameters((num_hidden, num_input))
    self.b_in = model.add_parameters((num_hidden))
    self.W_hid = model.add_parameters((num_hidden, num_hidden))
    self.b_hid = model.add_parameters((num_hidden))
    self.W_out = model.add_parameters((num_output, num_hidden))
    self.b_out = model.add_parameters((num_output)) 
    self.W_f0 = model.add_parameters((num_hidden, num_hidden))
    self.b_f0 = model.add_parameters((num_hidden))
    print "Loaded params for output"

  def reparameterize(self, mu, logvar):
    d = mu.dim()[0][0]
    eps = dy.random_normal(d)
    std = dy.exp(logvar * 0.5)
    return mu + dy.cmult(std, eps)

  def mlp(self, x, W, V, b):
    return V * dy.tanh(W * x + b)
  
  def calculate_loss_frame(self, input_frame, output_frame):

    ## Renew the computation graph
    dy.renew_cg()

    ## Instantiate the params
    W_mean = dy.parameter(self.W_mean_p)
    V_mean = dy.parameter(self.V_mean_p)
    b_mean = dy.parameter(self.b_mean_p)
    W_var = dy.parameter(self.W_var_p)
    V_var = dy.parameter(self.V_var_p)
    b_var = dy.parameter(self.b_var_p)

    W_frame = dy.parameter(self.W_in)
    b_frame = dy.parameter(self.b_in)
    W_f0 = dy.parameter(self.W_f0)
    b_f0 = dy.parameter(self.b_f0)
    W_out = dy.parameter(self.W_out)
    b_out = dy.parameter(self.b_out)
    input_frame = dy.inputTensor(np.asarray(input_frame))
    ouptut_frame = dy.inputTensor(output_frame)

    # Get output   
    dy.tanh(W_frame * input_frame + b_frame)
    W_f0 * dy.tanh(W_frame * input_frame + b_frame)         
    src_output =  dy.tanh(W_f0 * dy.tanh(W_frame * input_frame + b_frame) + b_f0 ) 

    # Get the mean and diagonal log covariance
    mu = self.mlp(src_output , W_mean, V_mean, b_mean)
    log_var = self.mlp(src_output , W_mean, V_mean, b_mean)

    # Compute the KL Divergence loss
    KL_loss = -0.5 * dy.sum_elems(1 + log_var - dy.pow(mu, dy.inputVector([2])) - dy.exp(log_var))

    # Reparameterize
    z = self.reparameterize(mu, log_var)

    # Calculate the reconstruction loss
    pred = dy.affine_transform([b_out, W_out, z])
    RECON_loss = dy.l2_norm(ouptut_frame - pred)

    return KL_loss,  RECON_loss

  def predict_frame(self, frame):

    ## Instantiate the params
    W_mean = dy.parameter(self.W_mean_p)
    V_mean = dy.parameter(self.V_mean_p)
    b_mean = dy.parameter(self.b_mean_p)
    W_var = dy.parameter(self.W_var_p)
    V_var = dy.parameter(self.V_var_p)
    b_var = dy.parameter(self.b_var_p)

    W_frame = dy.parameter(self.W_in)
    b_frame = dy.parameter(self.b_in)
    W_f0 = dy.parameter(self.W_f0)
    b_f0 = dy.parameter(self.b_f0)
    W_out = dy.parameter(self.W_out)
    b_out = dy.parameter(self.b_out)
    input_frame = dy.inputTensor(np.asarray(frame))

    # Get output           
    print dy.tanh(W_frame * input_frame + b_frame)
    print W_f0 * dy.tanh(W_frame * input_frame + b_frame)
    src_output =  dy.tanh(W_f0 * dy.tanh(W_frame * input_frame + b_frame) + b_f0 )

    # Get the mean and diagonal log covariance
    mu = self.mlp(src_output , W_mean, V_mean, b_mean)
    log_var = self.mlp(src_output , W_mean, V_mean, b_mean)

    # Compute the KL Divergence loss
    KL_loss = -0.5 * dy.sum_elems(1 + log_var - dy.pow(mu, dy.inputVector([2])) - dy.exp(log_var))

    # Reparameterize
    z = self.reparameterize(mu, log_var)

    # Calculate the reconstruction loss
    pred = dy.affine_transform([b_out, W_out, z])
    return pred



class VariationalEncoderDecoder(object):
   
  def __init__(self, model, args):
    self.pc = model.add_subcollection()
    self.args = args
    self.num_input = int(args[0])
    self.hidden_list = args[1]
    self.num_latent = int(args[2])
    self.num_output = int(args[3])
    self.act = args[4]
    self.model = model
    num_hidden = self.hidden_list[-1]    

    # Use DNN as encoder and decoder
    self.enc_dnn = FeedForwardNeuralNet(model, [self.num_input, self.hidden_list, num_hidden, self.act])
    self.dec_dnn = FeedForwardNeuralNet(model, [num_hidden, self.hidden_list,self.num_output, self.act])

    num_hidden = self.hidden_list[-1] # This is not really  hum_hidden, you see
    self.W_mean_p = model.add_parameters((self.num_latent, num_hidden))
    self.V_mean_p = model.add_parameters((num_hidden, self.num_latent))
    self.b_mean_p = model.add_parameters((self.num_latent))
    print "Loaded params for means"
   
    self.W_var_p = model.add_parameters((self.num_latent, num_hidden))
    self.V_var_p = model.add_parameters((num_hidden, self.num_latent))
    self.b_var_p = model.add_parameters((self.num_latent))
    print "Loaded params for variances"

  def mlp(self, x, W, V, b):
       return V * dy.tanh(W * x + b)

  def reparameterize(self, mu, logvar):
       d = mu.dim()[0][0]
       eps = dy.random_normal(d)
       std = dy.exp(logvar * 0.5)
       return mu + dy.cmult(std, eps)

  def calculate_loss_basic(self, input, output):

       # Instantiate the params
       W_mean = dy.parameter(self.W_mean_p)
       V_mean = dy.parameter(self.V_mean_p)
       b_mean = dy.parameter(self.b_mean_p)
       W_var = dy.parameter(self.W_var_p)
       V_var = dy.parameter(self.V_var_p)
       b_var = dy.parameter(self.b_var_p)
       num_hidden = self.hidden_list[-1]
       
       enc_dnn = self.enc_dnn
       dec_dnn = self.dec_dnn
    
       # Get the output from encoder
       src_output = enc_dnn.predict(input)

       # Get the mean and log variance
       mu = self.mlp(src_output , W_mean, V_mean, b_mean)
       log_var = self.mlp(src_output , W_mean, V_mean, b_mean)       
    
       # Compute the KL Divergence loss
       kl_loss = -0.5 * dy.sum_elems( 1 + log_var - dy.pow(mu, dy.inputVector([2])) - dy.exp(log_var))

       # Reparameterize
       z = self.reparameterize(mu, log_var)
       
       return kl_loss, dec_dnn.calculate_loss(z , output)


  def predict(self, input):
       # Renew the computation graph
       #dy.renew_cg()

       # Instantiate the params
       W_mean = dy.parameter(self.W_mean_p)
       V_mean = dy.parameter(self.V_mean_p)
       b_mean = dy.parameter(self.b_mean_p)
       W_var = dy.parameter(self.W_var_p)
       V_var = dy.parameter(self.V_var_p)
       b_var = dy.parameter(self.b_var_p)
       num_hidden = self.hidden_list[-1]

       enc_dnn = self.enc_dnn
       dec_dnn = self.dec_dnn

       # Get the output from encoder
       src_output = enc_dnn.predict(input)

       # Get the mean and log variance
       mu = self.mlp(src_output , W_mean, V_mean, b_mean)
       log_var = self.mlp(src_output , W_mean, V_mean, b_mean)
 
       # Compute the KL Divergence loss
       #kl_loss = -0.5 * dy.sum_elems(1 + log_var - dy.pow(mu, dy.inputVector([2])) - dy.exp(log_var))

       # Reparameterize
       z = self.reparameterize(mu, log_var)

       return dec_dnn.predict(z)
