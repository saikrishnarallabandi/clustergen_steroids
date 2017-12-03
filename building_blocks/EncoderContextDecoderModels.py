import _dynet
import dynet_config
dynet_config.set(mem=9000, requested_gpus=1, autobatch=1)
import dynet as dy
import os
import pickle
import numpy as np
import time
from DNNs import *

debug = 0
debug_time = 0

class EncoderDecoderModel(object):

    def __init__(self, model, args):
        self.pc =  model.add_subcollection()
        self.args = args
        self.num_layers = args[0]
        self.num_input = args[1]
        self.num_hidden = args[2]
        self.num_attention = args[3]
        self.num_output = args[4]
        self.act = args[5]
        self.model = model
        self.num_embeddings = self.num_output
        self.loss_fn = dy.l2_norm
        # Lookup parameter to store the end of sentence symbol
        self.M = self.args[6]
        self.spec = (self.num_layers,self.num_input, self.num_hidden, self.num_attention, self.num_output, self.act)

        # Add the LSTMs
        self.fwd_lstm_builder = dy.CompactVanillaLSTMBuilder(self.num_layers, self.num_input, self.num_hidden, model) 
        self.bwd_lstm_builder = dy.CompactVanillaLSTMBuilder(self.num_layers, self.num_input, self.num_hidden, model)
        self.w_decoder = self.model.add_parameters((self.num_output, self.num_hidden))
        self.b_decoder = self.model.add_parameters((self.num_output))

        # Attention MLP parameters
        self.attention_w1 = self.model.add_parameters((self.num_attention, self.num_hidden*2))
        self.attention_w2 = self.model.add_parameters((self.num_attention, self.num_hidden*self.num_layers))
        self.attention_v = self.model.add_parameters((1, self.num_attention))

        # Add the Decoder DNN
        self.decoder_dnn = FeedForwardNeuralNet(self.model, [self.num_input*11, [self.num_hidden], self.num_output, [dy.tanh, dy.tanh, dy.tanh, dy.tanh, dy.tanh, dy.tanh]])
        self.decoder_postfilter_dnn = FeedForwardNeuralNet(self.model, [self.num_hidden*2, [self.num_hidden], self.num_output, [dy.tanh, dy.tanh, dy.tanh, dy.tanh, dy.tanh, dy.tanh]])

    def set_M(self, n):
        print n
        self.M = n

    def set_loss(self, l):
       self.loss_fn = l

    @staticmethod
    def user_save_model(path, model, M):
        dy.save(path + '/edm', [model,M])
        #with open(path + '/params.pkl', 'wb') as f: 
        #       pickle.dump(M , f)

    @staticmethod
    def user_load_model(path, model):
        edm = dy.load(path + '/edm', model)
        print "Back from spec"
        print edm, M
        #with open(path + '/params.pkl', 'rb') as f:
        #    M = pickle.load(f)
        edm.set_M(M)
        return edm

    # support saving:
    def param_collection(self): return self.pc

    @staticmethod
    def from_spec(spec, model):
           print spec
           num_layers, num_input, num_hidden, num_attention, num_output, act = spec
           print " IN spec"      
           return EncoderDecoderModel(model, [num_layers,num_input, num_hidden, num_attention, num_output, act, " "])


    def attend(self, w1dt, vectors,state):
        import time
        start = time.time()
        if debug:
            print "In attention"
        w2 = dy.parameter(self.attention_w2)
        v = dy.parameter(self.attention_v)
        attention_weights = []

        if debug:
            print "State output: ", list(state.h())
        end = time.time()
        start = end
        w2dt = w2 * state
        end = time.time()
        if debug:
            print "W2dt output: ", len(w2dt.value())
        for input_vector in vectors:
            unnormalized = dy.transpose(v * dy.tanh(dy.colwise_add(w1dt, w2dt)))
            end = time.time()
            att_weights = dy.softmax(unnormalized)
            end = time.time()
        return vectors * att_weights

    def calculate_loss_minibatch(self, input_batch, output_batch):
       # This is a naive batching
       #dynet.renew_cg()
       loss_batch = 0
       for (input, output) in zip(input_batch, ouptut_batch):
           loss_batch += self.calculate_loss(input, output)
       return loss_batch/len(input_batch)



    def calculate_loss(self, input, output, input_context, output_context):
       start = time.time()
       start_time = start
       first_encode = 1
       if first_encode:

         # Apply forward LSTM
         init_state_fwd = self.fwd_lstm_builder.initial_state()
         states = init_state_fwd.add_inputs(dy.inputTensor(input))
         fwd_vectors = [state.output() for state in states]

         # Apply reverse LSTM
         init_state_bwd = self.bwd_lstm_builder.initial_state()
         input_rev = input[::-1]
         states = init_state_bwd.add_inputs(dy.inputTensor(input_rev))
         bwd_vectors = [state.output() for state in states]
         bwd_vectors = bwd_vectors[::-1]
         end = time.time()

         #  Concatenate the vectors
         lstm_vectors =  [dy.concatenate([fwd_out, bwd_out]) for fwd_out, bwd_out in zip(fwd_vectors, bwd_vectors)]
         bidirectional_vectors = dy.concatenate_cols(lstm_vectors)
         end = time.time()
         bidirectional_vectors_value = bidirectional_vectors.value()


       for idx, (input_frame_context, output_frame) in enumerate(zip(input_context, output_context)):
         dy.renew_cg()

         bidirectional_vectors_value_tensor  = dy.inputTensor(bidirectional_vectors_value)

         # Decoder
         output_frame = dy.inputTensor(output_frame)
         input_frame_context = dy.inputTensor(input_frame_context)
         w1 = dy.parameter(self.attention_w1)
         w1dt = w1 * bidirectional_vectors_value_tensor
         loss = []

         w_out = dy.parameter(self.w_decoder)
         b_out = dy.parameter(self.b_decoder)

         state_decoder = self.decoder_dnn.get_last_activation(input_frame_context)
         start = time.time()
         if debug_time :       
           print "Going to attention"
         attended_encoding = self.attend(w1dt, bidirectional_vectors_value_tensor, state_decoder)
         attention = [k[0].value() for k in attended_encoding]
         end = time.time()
         if debug_time :       
           print "Returned from attention ", end - start, end - start_time, idx 
         start = end

         frame_loss = self.decoder_postfilter_dnn.calculate_loss(dy.inputTensor(attention), output_frame)
         loss.append(frame_loss)         
       return dy.esum(loss)        



       
    def predict(self, input, input_context):

       start = time.time()
       start_time = start
       first_encode = 1
       if first_encode:

         # Apply forward LSTM
         init_state_fwd = self.fwd_lstm_builder.initial_state()
         states = init_state_fwd.add_inputs(dy.inputTensor(input))
         fwd_vectors = [state.output() for state in states]

         # Apply reverse LSTM
         init_state_bwd = self.bwd_lstm_builder.initial_state()
         input_rev = input[::-1]
         states = init_state_bwd.add_inputs(dy.inputTensor(input_rev))
         bwd_vectors = [state.output() for state in states]
         bwd_vectors = bwd_vectors[::-1]
         end = time.time()

         #  Concatenate the vectors
         lstm_vectors =  [dy.concatenate([fwd_out, bwd_out]) for fwd_out, bwd_out in zip(fwd_vectors, bwd_vectors)]
         bidirectional_vectors = dy.concatenate_cols(lstm_vectors)
         end = time.time()
         bidirectional_vectors_value = bidirectional_vectors.value()

       pred_frames = []
       for idx, input_frame_context in enumerate(input_context):

         dy.renew_cg()
         bidirectional_vectors_value_tensor  = dy.inputTensor(bidirectional_vectors_value)

         # Decoder
         input_frame_context = dy.inputTensor(input_frame_context)
         w1 = dy.parameter(self.attention_w1)
         w1dt = w1 * bidirectional_vectors_value_tensor
         loss = []

         w_out = dy.parameter(self.w_decoder)
         b_out = dy.parameter(self.b_decoder)

         state_decoder = self.decoder_dnn.get_last_activation(input_frame_context)
         start = time.time()
         if debug_time :
           print "Going to attention"
         attended_encoding = self.attend(w1dt, bidirectional_vectors_value_tensor, state_decoder)
         attention = [k[0].value() for k in attended_encoding]
         end = time.time()
         if debug_time :
           print "Returned from attention ", end - start, end - start_time, idx
         start = end

         frame_pred = self.decoder_postfilter_dnn.predict(dy.inputTensor(attention))
         pred_frames.append(frame_pred.value())
       return pred_frames
  

        
