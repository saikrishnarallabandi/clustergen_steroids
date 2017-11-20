import dynet as dy
import random
import numpy as np
import sys

debug = 0
debug_dimensions = 1
debug_tensor = 0
debug_vector = 0
debug_emergency = 0

'''
hE -> Encoder Hidden state
hD -> Decoder Hidden state
n  -> Size of Minibatch
l  -> length of maximum sequence in the minibatch
x  -> Time Steps in a single sequence
e  -> Size of Embedding

Input          -> [n * e * x]
Encoder Input  -> [n * e * x]
Encoder Output -> [l * 2hE * n]
Decoder Input  -> [l * (2hE + e) * n]
W1             -> [a * 2hE]
W1dt           -> [l * a * n]
W2             -> [a * 2hE]
W2dt           -> [l * a * n]
V              -> [1 * a]

sentences = ["abcd", "de", "wxy"]
'''

EOS = "<EOS>"
characters = list("abcdefghijklmnopqrstuvwxyz. ")
characters.append(EOS)

int2char = list(characters)
char2int = {c:i for i,c in enumerate(characters)}

VOCAB_SIZE = len(characters)

LSTM_NUM_OF_LAYERS = 1 
EMBEDDINGS_SIZE = 40
STATE_SIZE = 64
ATTENTION_SIZE = 32

model = dy.Model()

enc_fwd_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, model)
enc_bwd_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, model)

dec_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, STATE_SIZE*2+EMBEDDINGS_SIZE, STATE_SIZE, model)

input_lookup = model.add_lookup_parameters((VOCAB_SIZE, EMBEDDINGS_SIZE))
attention_w1 = model.add_parameters( (ATTENTION_SIZE, STATE_SIZE*2))
attention_w2 = model.add_parameters( (ATTENTION_SIZE, STATE_SIZE*LSTM_NUM_OF_LAYERS*2))
attention_v = model.add_parameters( (1, ATTENTION_SIZE))
decoder_w = model.add_parameters( (VOCAB_SIZE, STATE_SIZE))
decoder_b = model.add_parameters( (VOCAB_SIZE))
output_lookup = model.add_lookup_parameters((VOCAB_SIZE, EMBEDDINGS_SIZE))


def get_matrix_size(v):
   # Assuming v is a list of lists, return its size
   if type(v) == list:
      if debug_vector:
          print "  01.I got a list"
      if type(v[0]) == float:
          if debug_vector:
              print "  02.I got a list of lists"
          return np.array(v).shape, "matrix"
      else:
          if debug_vector:
              print "  02. This is a list of dynet variables or numpy array"
          if isinstance(v[0] , np.ndarray):
              print "  03.It is a numpy array"
              return len(v), v[0].shape
          first_dimension = len(v)
          second_dimension = len(v[0].value())
          return first_dimension, second_dimension, "matrix"

   elif isinstance(v, np.ndarray):
      if debug_vector:
         print "  01.I got a numpy array"
         
      return v.shape, "matrix"
   else:
     if debug_vector:
         print "   01.I got a dynet list" 
         print v
     sys.exit()     
     

def get_tensor_size(t):
   # Assuming t is a tensor, print its size
   # Python puts it as a list of list of list
   
   first_dimension = len(t)
   if debug_tensor:
       print "   I got " , t
       print "   1.First dimension:", first_dimension
   k = t[0]

   ## If second dimension is a list, input can be a list of lists or a list of dynet variables
   if type(k) == list:
     if debug_tensor:
        print "   2.I got a list of list" 
     try:
        third_dimension = len(k.value())
     except AttributeError:
        third_dimension = len(k)
     if type(k[0]) == list:
        if debug_tensor:
            print "  3.Its a list of list of lists"
        second_dimension = len(k[0])
        return first_dimension, second_dimension, third_dimension
     else:
        if debug_tensor:
            print "  3.Its a list of list of dynet variables"
        second_dimension = len(k[0].value())
        return first_dimension, second_dimension, third_dimension      
   # If second dimension is a dynet variable, it can be a dynet list or dynet scalar 
   else:
     if debug_tensor:
        print "  2.I got a dynet thing"
     if type(k[0]) == list:
         if debug_tensor:
             print "  3.I got a list of Dynetlist of list"
     else:
         if debug_tensor:
            print "  3. I dont think its a tensor. Its type is ", type(k[0])
         # I will check if its a numpy nd array
         if isinstance(k[0], np.ndarray):
            print k[0]
            second = k[0].shape
            return first_dimension, second
            #print k[0].value()
         return get_matrix_size(t)
         



def embed_sentences(sentences, flag=0):
    if debug:
      print "In embedding"
    sentences_new = []
    global input_lookup
    global output_lookup
    if flag == 1:
      selected_lookup = output_lookup
    else:
      selected_lookup = input_lookup

    for sentence in sentences:
       sentence = [EOS] + list(sentence) + [EOS]
       sentence = [char2int[c] for c in sentence]
       sentences_new.append([selected_lookup[char] for char in sentence])
    if debug_dimensions:
      print "  Length of encoded sentences is: ", len(sentences_new)
    if debug:
      print "  ", sentences
      return sentences_new
    else:
      return sentences_new

def encode_sentences(enc_fwd_lstm, enc_bwd_lstm, encodings):
# Takes in [n*e*x] and returns [l*2hE*n] as output
    if debug:
       print "In Encoder"
    if debug_dimensions:
       print "  The input dimensions are: ", get_tensor_size(encodings)

    encodings_transposed = []
    masks = []
    for i in range(len(encodings[0])):
       encodings_transposed.append([(sent[i] if len(sent)> i else input_lookup[0]) for sent in encodings])
       mask = [(1 if len(sent)>i else 0) for sent in encodings]
       masks.append(mask)
    encodings_transposed_rev = list(reversed(encodings_transposed))

    if debug_dimensions:
        print "  The dimensions of transposed inputs to Encoder are: ", get_tensor_size(encodings_transposed)
        print "  The dimensions of transposed reversed inputs to Encoder are: ", get_tensor_size(encodings_transposed_rev)

    state_fwd = enc_fwd_lstm.initial_state()
    state_bwd = enc_bwd_lstm.initial_state()
    count_debug = 0
    fwd_list = []
    bwd_list = []
    for (enc, encb) in zip(encodings_transposed, encodings_transposed_rev):
        fwd_list.append(state_fwd.add_inputs(enc))
        bwd_list.append(state_bwd.add_inputs(encb))
        count_debug += 1 
    if debug_emergency:
        print "####### Emergency check begin"
        for (i,s) in enumerate(fwd_list):
             print "Level 01: Time steps ", i  
             for k in s:
                 print " Level 02: Encoder output ", len(k.output().value())
        print "#######  Emergency check Done"

    fwd_vectors = []    
    bwd_vectors = []   
    vectors = []
    for (idx,s) in enumerate(fwd_list): 
       # Each element corresponds to output from a time step by the encoder 
       fwd = []
       for (id,k) in enumerate(s):   
               fwd.append(k.output())
       fwd_vectors.append(fwd)

    for (idx,s) in enumerate(bwd_list):
       # Each element corresponds to output from a time step by the encoder 
       bwd = []
       for (id,k) in enumerate(s):
               bwd.append(k.output())
       bwd_vectors.append(bwd)

    # Reverse the backward vectors
    bwd_vectors_rev = []
    for bwd in bwd_vectors:
        bwd_vectors_rev.append(list(reversed(bwd)))

    # Concatenate 
    for (f,b) in zip(fwd_vectors, bwd_vectors_rev):
         lstm_vectors =  [dy.concatenate([fwd_out, bwd_out]) for fwd_out, bwd_out in zip(f, b)]
         vectors.append(lstm_vectors)
 
    if debug_dimensions:
        print "  The dimensions of output from forward Encoder are: ", get_tensor_size(fwd_vectors)
        print "  The dimensions of output from reversed Encoder are: ", get_tensor_size(bwd_vectors)
        print "  The Encoder output dimensions are: ", get_tensor_size(vectors)
    return vectors

def attend(input_mat, state, w1dt_array):
# Takes in [l * 2hE * n], [l * 2hD * n], [l * a * n] as input and returns [l * a * n] as output 
  
    global attention_w2
    global attention_v
    w2 = dy.parameter(attention_w2)
    v = dy.parameter(attention_v)
    if debug_dimensions:
      print "In attention "
      print "   Dimensions of input mat are : ", get_tensor_size(input_mat)
      print "   Dimensions of w1dt array: ", get_tensor_size(w1dt_array)
      print "   Dimensions of state ", len(state)
    # Get w2dt = weight matrix * decoder state output
    w2dt_array = []      
    for s in state:
      w2dt =  w2*dy.concatenate(list(s.s()))
      w2dt_array.append(w2dt)
    if debug_dimensions:
      print "   Dimensions of w2dt array: ", get_tensor_size(w2dt_array) 
    unnormalized_array = []
    att_weights_array = []
    for (a,b) in zip(w1dt_array, w2dt_array): 
        unnormalized = dy.transpose(v * dy.tanh(dy.colwise_add(a,b)))
        att_weights = dy.softmax(unnormalized)
        att_weights_array.append(att_weights)
    if debug_dimensions:
       print "   Dimensions of attention weights array: ", get_tensor_size(att_weights_array)
    context_array = []
    for (im, at) in zip(input_mat, att_weights_array):    
        context = im * at
        context_array.append(context)
    if debug_dimensions: 
       print "   Dimensions of contexts array: ", get_tensor_size(context_array)
    return context_array

def decode_sentences(dec_lstm, vectors, outputs):
# Takes in [l*(2hE+e)*n] as input and returns [l*2hD*n] as output
    w = dy.parameter(decoder_w)
    b = dy.parameter(decoder_b)
    w1 = dy.parameter(attention_w1)
    w1_array = []
    # Concatenate the columns of the BiLSTM encodings
    bidirectional_vectors = []
    for (i,v) in enumerate(vectors):
         bidirectional_vectors.append(dy.concatenate_cols(v))
         w1_array.append(w1)  
    # Repeat w1 and make it a tensor    
    w1_repeated = [[w1]] * len(vectors)

    if debug_dimensions:
        print "In Decoder"
        print "  The dimensions of w1: ", get_tensor_size(w1.value())
        print "  The dimensions of w1 array: " , get_tensor_size(w1_array)
        print "  The dimensions of v are: ", get_tensor_size(v)
        print "  The dimensions of the bidirectional encodings is ", get_tensor_size(bidirectional_vectors)
        print "  The dimensions of w1_repeated: ", get_tensor_size(w1_repeated)
        print "  The dimensions of the first dimension of the w1 is : ", get_matrix_size(w1_repeated[2])

    dots = char2int[EOS]
    dots_batch = [[dots]]*len(bidirectional_vectors)
    last_output_embeddings = dy.lookup(output_lookup, dots)
    last_output_embeddings_batch = [[last_output_embeddings]]*len(outputs[0])       
    concatenated_stuff = [[dy.concatenate([dy.vecInput(STATE_SIZE*2), last_output_embeddings])]] * len(outputs[0])
    dec_state_array = dec_lstm.initial_state().add_inputs((k[0] for k in concatenated_stuff)) 

    if debug_dimensions:
       print "  The dimensions of last output embeddings batch: ", get_tensor_size(last_output_embeddings_batch) 
       print "  The dimensions of the concatenated stuff: ", get_tensor_size(concatenated_stuff)
    loss = 0

    decodings_transposed = []
    masks = []
    for i in range(len(outputs[0])):
       decodings_transposed.append([(sent[i] if len(sent)> i else output_lookup[0]) for sent in outputs])
       mask = [(1 if len(sent)>i else 0) for sent in outputs]
       masks.append(mask)
    if debug:
        print "Transposed Decodings: ", decodings_transposed

    # Get w1dt
    w1dt_array = []
    for (w,b) in zip(w1_array, bidirectional_vectors):
         w1dt_array.append(w*b)

    batch_loss = []
    for (y_batch, mask) in zip(decodings_transposed, masks):
      attention_output = attend(bidirectional_vectors, dec_state_array, w1dt_array)
      if debug_dimensions:
           print "Back in Decoder"
           print "  The dimensions of the concatenated stuff: ", get_tensor_size(concatenated_stuff)
           print "  The dimensions of the attention output: ", get_tensor_size(attention_output)
           #print "  The dimensions of Dec state : ", get_tensor_size(dec_state)
           print dec_state_array
      vector_array = []
      for (a,b) in zip(attention_output, concatenated_stuff):
            vector = dy.concatenate([a,b[0]])
            vector_array.append(vector)
      for (dec_state, vector) in zip(dec_state_array, vector_array):
           dec_state.add_input(vector)
      out_vectors_array = []
      for dec_state in dec_state_array:
           out_vectors = w * dec_state.output()
           out_vectors_array.append(out_vectors)
      print out_vectors_array
      loss = dy.pickneglogsoftmax_batch(out_vectors, y_batch)
      batch_loss.append(loss)
    return dy.esum(batch_loss)
            
def generate(in_seq, enc_fwd_lstm, enc_bwd_lstm, dec_lstm):
    embedded = embed_sentence(in_seq)
    encoded = encode_sentence(enc_fwd_lstm, enc_bwd_lstm, embedded)

    w = dy.parameter(decoder_w)
    b = dy.parameter(decoder_b)
    w1 = dy.parameter(attention_w1)
    input_mat = dy.concatenate_cols(encoded)
    w1dt = None
    last_output_embeddings = output_lookup[char2int[EOS]]
    s = dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(STATE_SIZE * 2), last_output_embeddings]))
    out = ''
    count_EOS = 0
    for i in range(len(in_seq)*2):
        if count_EOS == 2: break
        w1dt = w1dt or w1 * input_mat
        vector = dy.concatenate([attend(input_mat, s, w1dt), last_output_embeddings])
        s = s.add_input(vector)
        out_vector = w * s.output() + b
        probs = dy.softmax(out_vector).vec_value()
        next_char = probs.index(max(probs))
        last_output_embeddings = output_lookup[next_char]
        if int2char[next_char] == EOS:
            count_EOS += 1
            continue

        out += int2char[next_char]
    return out


def get_loss(input_sentences, output_sentences, enc_fwd_lstm, enc_bwd_lstm, dec_lstm):
    dy.renew_cg()
    embedded = embed_sentences(input_sentences)
    encoded = encode_sentences(enc_fwd_lstm, enc_bwd_lstm, embedded)
    embedded = embed_sentences(output_sentences,1)
    decode_sentences(dec_lstm, encoded, embedded)
    #return decode(dec_lstm, encoded, output_sentence)


def train(model, sentences):
    trainer = dy.SimpleSGDTrainer(model)
    for i in range(1):
        get_loss(sentences, sentences, enc_fwd_lstm, enc_bwd_lstm, dec_lstm)
        '''
        loss = get_loss(sentences, sentences, enc_fwd_lstm, enc_bwd_lstm, dec_lstm)
        loss_value = loss.value()
        loss.backward()
        trainer.update()
        if i % 20 == 0:
            print(loss_value)
            print(generate(sentence, enc_fwd_lstm, enc_bwd_lstm, dec_lstm))
        '''

sentences = ["abcd", "de", "wxy"]
sentences.sort(key=lambda x: -len(x))
if debug:
   print '\n\n\n'
   print "Sorted Sentences: ", sentences
train(model, sentences)
