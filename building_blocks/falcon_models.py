import dynet as dy
import os
import pickle
import numpy as np
import numpy as np
import os,sys
from sklearn import preprocessing
import pickle, logging
import argparse

debug = 1

class falcon_heavy(object):
    
   def __init__(self, model, args):
      self.pc = model.add_subcollection()
      self.model = model
      self.args = args
      self.num_input = args.num_input
      self.num_output = args.num_output
      self.generic_layer_list = args.generic_layer_list
      self.postspecificlayers = args.postspecificlayers
      self.number_of_layers =  len(self.generic_layer_list) + len(self.postspecificlayers) + 1
      num_hidden_1 = self.generic_layer_list[0]
      self.act_generic = args.act_generic
      self.act_postspecific = args.act_postspecific

      # Add first layer
      if debug :
           print "Adding input to the network ", num_hidden_1, self.num_input
      self.W1 = self.pc.add_parameters((num_hidden_1, self.num_input))
      self.b1 = self.pc.add_parameters((num_hidden_1))

      # Add generic layers
      self.weight_matrix_array = []
      self.biases_array = []
      self.weight_matrix_array.append(self.W1)
      self.biases_array.append(self.b1)
      for k in range(1, len(self.generic_layer_list)):
          if debug: 
             print "At ", k , " adding generic weights ", self.generic_layer_list[k], self.generic_layer_list[k-1]
          self.weight_matrix_array.append(self.model.add_parameters((self.generic_layer_list[k], self.generic_layer_list[k-1])))
          self.biases_array.append(self.model.add_parameters((self.generic_layer_list[k])))

      # Add specific layers
      self.specific_weights_array = []
      self.specific_biases_array = []
      print "Adding specific layers "
      for (i, layer) in enumerate(self.postspecificlayers):
          if debug:
             print "At ", i , " adding specific weights ", self.postspecificlayers[i], self.postspecificlayers[i-1]
          self.specific_weights_array.append(  self.model.add_parameters(( int(layer)    , self.postspecificlayers[-1] )) )     
          self.specific_biases_array.append( self.model.add_parameters(( int(layer) )) )

      # Residual
      if debug:
         print "Adding final layer ", self.num_output   , int(layer)+self.num_input  
      self.W_final =  self.model.add_parameters(( self.num_output   , int(layer)+self.num_input  )) 
      self.b_final = self.model.add_parameters((self.num_output))

      # Spec
      self.spec = (args)


   def calculate_loss(self,input,output,tgtspk):
         # Initial layer
         weight_matrix_array = [dy.parameter(self.W1)]
         biases_array = [dy.parameter(self.b1)]
         acts = []
         # Generic layers
         for (W,b,a) in zip(self.weight_matrix_array, self.biases_array, self.act_generic):
             weight_matrix_array.append(dy.parameter(W))
             biases_array.append(dy.parameter(b)) 
             acts.append(a)
         # Specific layers
         length = len(self.postspecificlayers)
         start_index = (tgtspk -1)*length  
         idx = 0
         for (W,b,a) in zip(self.specific_weights_array[start_index:start_index+length], self.specific_biases_array[start_index:start_index+length], self.act_postspecific):
             weight_matrix_array.append(dy.parameter(W))
             biases_array.append(dy.parameter(b))              
             acts.append(a)
         # Final Layer
         weight_matrix_array.append(dy.parameter(self.W_final))
         biases_array.append(dy.parameter(self.b_final))

         w = weight_matrix_array[0]
         b = biases_array[0]
         act = acts[0]
         intermediate = act(dy.affine_transform([b, w, input]))
         if debug:
             print "Dimensions of the intermediate: "
             print len(intermediate.value())
         activations = [intermediate]
         count = 0
         for (W,b,g) in zip(weight_matrix_array[1:], biases_array[1:], acts[1:]):
            if debug:
               print "Adding to the layer number: ", count+1
            #if count == self.number_of_layers+1:
            #      pred = g(dy.affine_transform([b, W, activations[-1]+input  ]))
            #else:     
            pred = g(dy.affine_transform([b, W, activations[-1]]))
            activations.append(pred)  
            count += 1
         if debug:
            print "Activation dimensions are : ", [len(k.value()) for k in activations]
            print "Output dimensions are: ", len(output.value())
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
                  
