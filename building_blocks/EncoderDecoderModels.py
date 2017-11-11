import dynet as dy
import os
import pickle
import numpy as np

class EncoderDecoderModel(object):

    def __init__(self, models, args):
        self.pc = pc
        self.args = args
        self.num_layers = args[0]
        self.num_input = args[1]
        self.num_hidden = args[2]
        self.num_attention = args[3]
        self.act = args[4]
        self.model = model
        
        # Add the LSTMs
        self.fwd_lstm_builder = dy.LSTMBuilder(self.num_layers, self.num_input, num_hidden, model) 
        self.bwd_lstm_builder = dy.LSTMBuilder(self.num_layers, self.num_input, num_hidden, model)
        self.decoder_lstm_builder = dy.LSTMBuilder(self.num_layers,self.num_hidden + self.num_attention, self.num_hidden, model)
       
        # Attention MLP parameters
        self.attention_w1 = self.model.add_parameters((self.num_hidden, self.num_hidden))
        self.attention_w2 = self.model.add_parameters((self.num_hidden, self.num_output))
        self.attention_v = self.model.add_parameters((1, self.num_output))
       

    def attend(self, vectors, state):
        w1 = dy.parameter(self.attention_w1)
        w2 = dy.parameter(self.attention_w2)
        v = dy.parameter(self.attention_v)
        attention_weights = []

        w2dt = w2 * state.h()[-1]
        for input_vector in input_vectors:
            attention_weight = v * dy.tanh(w1 * input_vector + w2dt)
            attention_weights.append(attention_weight)
        attention_weights = dy.softmax(dy.concatenate(attention_weights))

        output_vectors = dy.esum([vector * attention_weight for vector, attention_weight in zip(input_vectors, attention_weights)])
        return output_vectors

    def calculate_loss(self, input, output):
       
       # Apply forward LSTM
       init_state_fwd = self.fwd_lstm_builder.initial_state()
       states = init_state_fwd.add_inputs(input)
       fwd_vectors = [state.output() for state in states]

       # Apply reverse LSTM
       init_state_bwd = self.bwd_lstm_builder.initial_state()
       states = init_state_bwd.add_inputs(input[::-1])
       bwd_vectors = [state.output() for state in states]
       bwd_vectors = bwd_vectors[::-1]

       # Concatenate the vectors
       lstm_vectors =  [dy.concatenate([fwd_out, bwd_out]) for fwd_out, bwd_out in zip(fwd_vectors, bwd_vectors)]
       bidirectional_vectors = dy.concatenate_cols(lstm_vectors)

       # Decoder
       state_decoder = self.decoder_lstm_builder.initial_state().add_input(dy.concatenate([dy.vecInput(self.hidden*2)]))
       output_frames = []
       while True:
         attended_encoding = self.attend(bidirectional_vectors,state_decoder)
         state_decoder = state_decoder.add_input(attended_encoding)
         output_frames.append(state_decoder.output())
         if state_decoder.output() == EOS or len(output_frames) > 2*len(input):
               break
       output_predicted = np.asarray(output_frames)

       # Return the loss
       return dy.l2_norm(output, output_predicted)        

       

  

        
