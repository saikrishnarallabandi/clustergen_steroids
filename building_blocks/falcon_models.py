print "In building block"
import _dynet
import dynet_config
dynet_config.set_gpu()
dynet_config.set(mem=11000, requested_gpus=1, autobatch=1)

from DNNs import *
import dynet as dy
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
      self.model = model
      self.args = args
      self.num_input = args.num_input
      self.num_output = args.num_output
      self.num_hidden = args.num_hidden
      self.act_generic = args.act_generic
      self.act_postspecific = args.act_postspecific
      self.act_final = args.act_final

      # Add first layer
      self.pre_dnn_array = []
      for i in range(4):
          self.pre_dnn_array.append(FeedForwardNeuralNet(self.model, [self.num_input, [self.num_hidden], self.num_hidden, [dy.selu, dy.selu, dy.selu, dy.selu] ]))
      # Add generic layers
      self.generic_dnn = FeedForwardNeuralNet(self.model, [self.num_hidden, [self.num_hidden], self.num_hidden, [dy.selu, dy.selu, dy.selu, dy.selu, dy.selu, dy.selu] ])
      # Add specific layers
      self.post_dnn_array = []
      self.post_dnn_array = []
      for i in range(4):
          self.post_dnn_array.append(FeedForwardNeuralNet(self.model, [self.num_hidden, [self.num_hidden], self.num_output, [dy.selu, dy.selu, dy.selu] ]))

      # Spec
      self.spec = (args)


   def calculate_loss(self,input,output,srcspk, tgtspk):

         source_dnn = self.pre_dnn_array[srcspk]
         target_dnn = self.post_dnn_array[tgtspk]
         
         source_encoding = source_dnn.predict(input)
         generic = self.generic_dnn.predict(source_encoding)
         loss = target_dnn.calculate_loss(generic, output)
         return loss
 
   def predict(self,input,srcspk, tgtspk):

         source_dnn = self.pre_dnn_array[srcspk]
         target_dnn = self.post_dnn_array[tgtspk]

         source_encoding = source_dnn.predict(input)
         generic = self.generic_dnn.predict(source_encoding)
         frame = target_dnn.predict(generic)
         return frame
 
