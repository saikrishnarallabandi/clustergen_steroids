import dynet as dy
import os
import pickle
import numpy as np
import time

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
        self.fwd_lstm_builder = dy.LSTMBuilder(self.num_layers, self.num_input, self.num_hidden, model) 
        self.bwd_lstm_builder = dy.LSTMBuilder(self.num_layers, self.num_input, self.num_hidden, model)
        self.decoder_lstm_builder = dy.LSTMBuilder(self.num_layers,self.num_hidden*2+self.num_embeddings, self.num_hidden, model)
        self.w_decoder = self.model.add_parameters((self.num_output, self.num_hidden))
        self.b_decoder = self.model.add_parameters((self.num_output))

        # Attention MLP parameters
        self.attention_w1 = self.model.add_parameters((self.num_attention, self.num_hidden*2))
        self.attention_w2 = self.model.add_parameters((self.num_attention, self.num_hidden*2*self.num_layers))
        self.attention_v = self.model.add_parameters((1, self.num_attention))

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
        w2dt = w2 * dy.concatenate(list(state.s()))
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
       dynet.renew_cg()
       loss_batch = 0
       for (input, output) in zip(input_batch, ouptut_batch):
           loss_batch += self.calculate_loss(input, output)
       return loss_batch/len(input_batch)



    def calculate_loss(self, input, output):
       start = time.time()
       if debug_time :
           print "Applying Encoder"
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
       if debug_time :       
           print "Applied encoder, ", end - start
       start = end

       # Concatenate the vectors
       lstm_vectors =  [dy.concatenate([fwd_out, bwd_out]) for fwd_out, bwd_out in zip(fwd_vectors, bwd_vectors)]
       if debug:
           print "The number of bidirectional vectors: ", len(lstm_vectors), " which means I think this is the length of the source sentence"
       bidirectional_vectors = dy.concatenate_cols(lstm_vectors)
       if debug:
           print "The number of bidirectional vectors: ", len(bidirectional_vectors.value()), " which means I think this is the length of the source sentence"
       end = time.time()
       if debug_time :       
           print "COncatenated the vectors, ", end - start  
  
       # Decoder
       w_out = dy.parameter(self.w_decoder)
       b_out = dy.parameter(self.b_decoder)
       if debug:
           print "First input to decoder: ", len(dy.concatenate([dy.vecInput(self.num_hidden*2),self.M[0]]).value())
       state_decoder = self.decoder_lstm_builder.initial_state().add_input(dy.concatenate([dy.vecInput(self.num_hidden*2), self.M[0]]))
       last_embeddings = self.M[0]
       if debug:
            print "Length of last embeddings: ", len(last_embeddings.value())
       output = dy.inputTensor(output)
       loss = []
       w1 = dy.parameter(self.attention_w1)
       w1dt = w1 * bidirectional_vectors
       for output_frame in output:
         start = time.time()
         if debug_time :       
           print "Going to attention"
         attended_encoding = self.attend(w1dt, bidirectional_vectors, state_decoder)
         end = time.time()
         if debug_time :       
           print "Returned from attention ", end - start
         start = end
         #if debug:
         #    print "Attention output is: ", len(attended_encoding.value())
         if debug_time :       
           print "Adding to decoder"
         state_decoder = state_decoder.add_input(dy.concatenate([attended_encoding,last_embeddings]))
         end = time.time()
         if debug_time :       
           print "Added to deocder, ", end - start
         start = end
         #state_decoder = state_decoder.add_input(dy.concatenate([dy.vecInput(self.num_hidden*2),last_embeddings]))
         ### Predict the frames now
         ### Adding because there was a griendt inf error  15 Nov 2017 
         frame_predicted = w_out * state_decoder.output() + b_out
         frame_predicted = dy.rectify(w_out * state_decoder.output() + b_out)
         last_embeddings = frame_predicted
         if debug_time :       
           print "Obtained last embeddings, ", end - start
         if debug:
             print "Length of updated last embeddings is : ", len(last_embeddings.value())
             print '\n'
         if self.loss_fn == dy.binary_log_loss:
             loss.append(self.loss_fn(output_frame, frame_predicted)/self.num_output)
         else:
             loss.append(self.loss_fn(output_frame- frame_predicted))         
       return dy.esum(loss)        


    def calculate_loss_batch(self, input_batch, output_batch):
       start = time.time()
       if debug_time :
           print "Applying Encoder"
       # Transpos the inputs and outputs
       input = np.transpose(input_batch)
       outpt = np.transpose(output_batch)

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
       if debug_time :
           print "Applied encoder, ", end - start
       start = end

       # Concatenate the vectors
       lstm_vectors =  [dy.concatenate([fwd_out, bwd_out]) for fwd_out, bwd_out in zip(fwd_vectors, bwd_vectors)]
       if debug:
           print "The number of bidirectional vectors: ", len(lstm_vectors), " which means I think this is the length of the source sentence"
       bidirectional_vectors = dy.concatenate_cols(lstm_vectors)
       if debug:
           print "The number of bidirectional vectors: ", len(bidirectional_vectors.value()), " which means I think this is the length of the source sentence"
       end = time.time()
       if debug_time :
           print "COncatenated the vectors, ", end - start

       # Decoder
       w_out = dy.parameter(self.w_decoder)
       b_out = dy.parameter(self.b_decoder)
       if debug:
           print "First input to decoder: ", len(dy.concatenate([dy.vecInput(self.num_hidden*2),self.M[0]]).value())
       state_decoder = self.decoder_lstm_builder.initial_state().add_input(dy.concatenate([dy.vecInput(self.num_hidden*2), self.M[0]]))
       last_embeddings = self.M[0]
       if debug:
            print "Length of last embeddings: ", len(last_embeddings.value())
       output = dy.inputTensor(output)
       loss = []
       w1 = dy.parameter(self.attention_w1)
       w1dt = w1 * bidirectional_vectors
       for output_frame in output:
         start = time.time()
         if debug_time :
           print "Going to attention"
         attended_encoding = self.attend(w1dt, bidirectional_vectors, state_decoder)
         end = time.time()
         if debug_time :
           print "Returned from attention ", end - start
         start = end
         #if debug:
         #    print "Attention output is: ", len(attended_encoding.value())
         if debug_time :
           print "Adding to decoder"
         state_decoder = state_decoder.add_input(dy.concatenate([attended_encoding,last_embeddings]))
         end = time.time()
         if debug_time :
           print "Added to deocder, ", end - start
         start = end
         #state_decoder = state_decoder.add_input(dy.concatenate([dy.vecInput(self.num_hidden*2),last_embeddings]))
         ### Predict the frames now
         frame_predicted = w_out * state_decoder.output() + b_out
         last_embeddings = frame_predicted
         if debug_time :
           print "Obtained last embeddings, ", end - start
         if debug:
             print "Length of updated last embeddings is : ", len(last_embeddings.value())
             print '\n'
         cross_entropy = self.loss_fn(output_frame- frame_predicted)

       
    def predict(self, input):
       # Renew the CG
       #dy.renew_cg()

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

       # Concatenate the vectors
       lstm_vectors =  [dy.concatenate([fwd_out, bwd_out]) for fwd_out, bwd_out in zip(fwd_vectors, bwd_vectors)]
       if debug:
           print "The number of bidirectional vectors: ", len(lstm_vectors), " which means I think this is the length of the source sentence"
       bidirectional_vectors = dy.concatenate_cols(lstm_vectors)
       if debug:
           print "The number of bidirectional vectors: ", len(bidirectional_vectors.value()), " which means I think this is the length of the source sentence"

       # Decoder
       w_out = dy.parameter(self.w_decoder)
       b_out = dy.parameter(self.b_decoder)
       if debug:
           print "First input to decoder: ", len(dy.concatenate([dy.vecInput(self.num_hidden*2),self.M[0]]).value())
       state_decoder = self.decoder_lstm_builder.initial_state().add_input(dy.concatenate([dy.vecInput(self.num_hidden*2), self.M[0]]))

       last_embeddings = self.M[0]
       if debug:
            print "Length of last embeddings: ", len(last_embeddings.value())
       output_frames = []
       while True:
         #attended_encoding = self.attend(bidirectional_vectors,state_decoder)
         #if debug:
         #    print "Attention output is: ", len(attended_encoding.value())
         #state_decoder = state_decoder.add_input(dy.concatenate([attended_encoding,last_embeddings]))
         state_decoder = state_decoder.add_input(dy.concatenate([dy.vecInput(self.num_hidden*2),last_embeddings]))
         ### Predict the frames now
         frame_predicted = w_out * state_decoder.output() + b_out
         last_embeddings = frame_predicted
         if debug:
             print "Length of updated last embeddings is : ", len(last_embeddings.value())
             print '\n'
         output_frames.append(frame_predicted.value())
         if last_embeddings == self.M[0] or len(output_frames) > 2* len(input):
              break
       return output_frames


  

        
