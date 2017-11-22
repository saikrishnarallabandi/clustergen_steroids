import dynet as dy
import os
import pickle
import numpy as np
import numpy as np
import os,sys
from sklearn import preprocessing
import pickle, logging
import argparse

debug = 0

class falcon_heavy(object):
    
   def __init__(self, model, args):
      self.pc = model.add_subcollection()
      self.args = args
      self.num_input = args.num_input
      self.num_output = args.num_output
      self.generic_layer_list = args.generic_layer_list
      self.postspecificlayers = args.postspecificlayers
      self.speakers = args.speakers
      self.number_of_layers = 1 + self.num_hidden + 1
      num_hidden_1 = self.generic_layers[0]
      self.act = args.act

      # Add first layer
      if debug :
           print "Adding input to first hidden layer weights ", num_hidden_1, self.num_input
      self.W1 = self.pc.add_parameters((num_hidden_1, self.num_input))
      self.b1 = self.pc.add_parameters((num_hidden_1))

      # Add generic layers
      self.weight_matrix_array = []
      self.biases_array = []
      self.weight_matrix_array.append(self.W1)
      self.biases_array.append(self.b1)
      for k in range(1, len(self.generic_layer_list)):
          if debug: 
             print "At ", k , " adding weights ", self.generic_layer_list[k], self.generic_layer_list[k-1]
          self.weight_matrix_array.append(self.model.add_parameters((self.generic_layer_list[k], self.generic_layer_list[k-1])))
          self.biases_array.append(self.model.add_parameters((self.generic_layer_list[k])))

      # Add specific layers
      if debug :
          self.specific_weights_array = []
          self.specific_biases_array = []
          print "Adding specific layers "
          for (i, layer) in enumerate(self.postspecificlayers):
            self.specific_weights_array.append(  self.model.add_parameters(( int(layer)    , self.generic_layer_list[-1] )) )     
            self.specific_biases_array.append( self.model.add_parameters(( int(layer) )) )

      # Residual
      self.specific_weights_array.append(  self.model.add_parameters(( num_output   , int(layer)+self.num_input  )) )
      self.biases_array.append(self.model.add_parameters((self.num_output)))

      # Spec
      self.spec = (args)


      def calculate_loss(self,input,output,tgtspk):
         # Initial layer
         weight_matrix_array = [dy.parameter(W1)]
         biases_array = [dy.parameter(b1)]
         # Generic layers
         for (W,b) in zip(self.weight_matrix_array, self.biases_array):
             weight_matrix_array.append(dy.parameter(W))
             biases_array.append(dy.parameter(b)) 
         # Specific layers
         start_index = tgtspk
         length = len(self.postspecificlayers) + 1
         idx = 0
         for (W,b) in zip(self.specific_weights_array[start_index:start_index+length], self.specific_biases_array[start_index:start_index+length]):
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
            losses = output - pred
         return dy.l2_norm(losses)
 
      def predict(self,input,output,tgtspk):
         # Initial layer
         weight_matrix_array = [dy.parameter(W1)]
         biases_array = [dy.parameter(b1)]
         # Generic layers
         for (W,b) in zip(self.weight_matrix_array, self.biases_array):
             weight_matrix_array.append(dy.parameter(W))
             biases_array.append(dy.parameter(b))
         # Specific layers
         start_index = tgtspk
         length = len(self.postspecificlayers) + 1
         idx = 0
         for (W,b) in zip(self.specific_weights_array[start_index:start_index+length], self.specific_biases_array[start_index:start_index+length]):
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
         return pred
                  
