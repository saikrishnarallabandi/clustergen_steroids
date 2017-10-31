import dynet as dy
import os
import pickle
import numpy as np

class EncoderDecoderBiLSTM_file(object):

  def __init__(self, model, num_input, num_hidden, num_output, act=dy.tanh):
    self.num_input = int(num_input)
    self.num_hidden = int(num_hidden)
    self.num_out = int(num_output)
    self.model = model
    print "Loaded params"

    # LSTM Parameters
    self.enc_lstm_fwd_builder = dy.LSTMBuilder(1, self.num_input, num_hidden, model)  
    self.enc_lstm_bwd_builder = dy.LSTMBuilder(1, self.num_input, num_hidden, model)
    self.dec_lstm_builder = dy.LSTMBuilder(1, self.num_input, num_hidden*2, model) 

    # MLP to predict the duration
    self.W_duration = self.model.add_parameters((self.num_hidden, self.num_hidden*2))
    self.b_duration = self.model.add_parameters((1))

    # MLP to predict f0
    self.W_f0 = self.model.add_parameters((self.num_hidden, self.num_hidden*2))
    self.b_f0 = self.model.add_parameters((1))


  def mlp(self,x, W, b):
       return dy.tanh( W * x + b)

  def calculate_loss(self, src_file , tgt_file):

    # Renew the computation graph
    dy.renew_cg()

    # Initialize LSTMs
    enc_init_state_fwd = self.enc_lstm_fwd_builder.initial_state()
    enc_init_state_bwd = self.enc_lstm_bwd_builder.initial_state()

    # MLP to predict the duration
    W_d = dy.parameter(self.W_duration)
    b_d = dy.parameter(self.b_duration)

    # MLP to predict f0
    W_f0 = dy.parameter(self.W_f0)
    b_f0 = dy.parameter(self.b_duration)
   
    input_frames = dy.inputTensor(np.loadtxt(src_file))
    output_frames = dy.inputTensor(np.loadtxt(tgt_file))
    len_tgt = len(np.loadtxt(tgt_file))
    input_frames_reverse = dy.inputTensor(np.flipud(np.loadtxt(src_file)))

    # Get the LSTM embeddings
    fwd_output = enc_init_state_fwd.add_inputs([frame for frame in input_frames])[-1].output()
    bwd_output = enc_init_state_bwd.add_inputs([frame for frame in input_frames_reverse])[-1].output()    

    # Concatenate
    bilstm_embeddings = dy.concatenate([fwd_output, bwd_output])

    # Predict durations
    target_duration = self.mlp(bilstm_embeddings, W_d, b_d)    
    duration_loss = dy.l2_norm(target_duration - len_tgt)

    # initialize decoder LSTM
    dec_init_state = self.dec_lstm_builder.initial_state().add_inputs(bilstm_embeddings)[-1].output()

    # Generate target frames
    prediction_loss = []
    for k in range(len_tgt):
         predicted_frame = self.mlp(dec_init_state,W_f0, b_f0)
         prediction_loss.append(dy.l2_norm(predicted_frame - output_frames[k]))
    return duration_loss, dy.esum(prediction_loss)

  def predict_label(self, file):

    # Initialize LSTMs
    enc_init_state_fwd = self.enc_lstm_fwd_builder.initial_state()
    enc_init_state_bwd = self.enc_lstm_bwd_builder.initial_state()

    # MLP to predict the duration
    W_duration = self.model.add_parameters((1, num_hidden*2))
    b_duration = self.model.add_parameters((1))

    # MLP to predict f0
    W_f0 = self.model.add_parameters((1, num_hidden*2))
    b_duration = self.model.add_parameters((1))

    def mlp(self, x, W, b):
       return dy.tanh( W * x + b)

    input_frames = dy.inputTensor(np.loadtxt(src_file))
    output_frames = dy.inputTensor(np.loadtxt(tgt_file))
    input_frames_reverse = dy.inputTensor(np.flipud(np.loadtxt(file)))

    # Get the LSTM embeddings
    fwd_output = enc_init_state_fwd.add_inputs([frame for frame in input_frames])[-1].output()
    bwd_output = enc_init_state_bwd.add_inputs([frame for frame in input_frames_reverse])[-1].output()

    # Concatenate
    bilstm_embeddings = dy.concatenate([fwd_output, bwd_output])

    # Predict durations
    target_duration = self.mlp(bilistm_embeddings, W_duration, b_duration)

    # initialize decoder LSTM
    dec_init_state = self.dec_lstm_builder.initial_state().add_inputs([bilstm_embeddings])

    # Generate target frames
    predicted_frames = []
    for k in range(len(output_frames)):
         predicted_frame = self.mlp(dec_init_state,W_f0, b_f0)
         prediction_frames.append(predicted_frame)
    return predicted_frames

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
      return EncoderDecoder(model, num_input, hidden_layer_list, num_out, act=dy.tanh)


