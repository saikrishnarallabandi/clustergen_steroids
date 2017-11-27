import sys, os
sys.path.append('/home/srallaba/hacks/repos/clustergen_steroids')
from building_blocks.falcon_models import *
import dynet as dy
import numpy as np
from math import sqrt
from sklearn import preprocessing
import random
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
import time

data_dir = '/home/srallaba/data/vc_stuff/database'

src = sys.argv[1]
tgt = sys.argv[2]
hidden = sys.argv[3]
scaling = int(sys.argv[4])
exp_name = sys.argv[5]
context = int(sys.argv[6])

debug  = 1
use_predicted_f0 = 1
use_predicted_mcep = 1
use_predicted_str = 1
test = 1
s = 0
e = 56

# Architecture
arch = hidden + 'T' + hidden + 'T' + hidden + 'T' + hidden + 'T_DNN'

# Log file
logfile_name = exp_name + '/logs/log_' + exp_name.split('/')[-1] + '_' + arch + '_' + src +  '_' + tgt + '_'  + 'scaling_' + str(scaling) + 'context_' + str(context) + '_phonealigned.txt'
g = open(logfile_name, 'w')
g.close()

# Saving directory
save_dir =  exp_name + '/outputs/' + arch + '_' + 'scaling_' + str(scaling) + 'context_' + str(context) + '_phonealigned'
if not os.path.exists(save_dir):
   os.makedirs(save_dir)

# Results file
results_file = save_dir + '/' + src + '_' + tgt + '_' + 'results.txt'
g = open(results_file , 'w')
g.close()

# Accumulate test data
test_files = sorted(os.listdir(data_dir + '/coeffs_test/'))
test_data_src = []
test_data_tgt = []
for file in test_files:
   if src in file:
       test_data_src.append(data_dir +  '/coeffs_test/' + file)
   elif tgt in file:
       test_data_tgt.append(data_dir +  '/coeffs_test/' + file)

# Load the train data
data_file =  data_dir + '/' + src + '_' + tgt + '_aligned.vectors'
train_data_src = np.loadtxt(data_file, usecols=range(56))
train_data_tgt = np.loadtxt(data_file,usecols=range(56,112))
test_data = zip(test_data_src, test_data_tgt)

# Contexts
def make_contexts(nparray, window):
   context_frames = np.asarray(zip(*[nparray[n:] for n in range(window)]))
   temp = []
   for f in context_frames:
       temp.append(np.concatenate(([f[n] for n in range(window)])))
   return np.asarray(temp)

# Hyperparameters 
units_input = 56 * int(context)
units_hidden = int(hidden)
units_output = 56

# Define the arguments
def parse_arguments(parser):
  parser.add_argument('src', type=str)
  parser.add_argument('tgt', type=str) 
  parser.add_argument('hidden',type=int)
  parser.add_argument('scaling',type=int)
  parser.add_argument('exp_name',type=str)
  parser.add_argument('context',type=int)
  args = parser.parse_args()
  return args

# Instantiate DNN and define the loss
m = dy.Model()
parser = argparse.ArgumentParser()
args = parse_arguments(parser)
args.num_input = units_input
args.num_output = units_output
args.generic_layer_list = [ units_input, units_hidden+1, units_hidden+2, units_hidden+3, units_hidden+4]
args.postspecificlayers = [units_hidden+4]
args.act_generic = [ dy.tanh, dy.tanh, dy.tanh, dy.tanh, dy.tanh]
args.act_postspecific = [dy.tanh]
args.act_final = dy.tanh
dnn = falcon_heavy(m, args)
trainer = dy.AdamTrainer(m)
update_params = 32
num_epochs = 30

# Scaling
if scaling:
   input_scaler = preprocessing.StandardScaler().fit(train_data_src)
   output_scaler = preprocessing.StandardScaler().fit(train_data_tgt)

   frames_handled_input = input_scaler.transform(train_data_src)
   frames_handled_output = output_scaler.transform(train_data_tgt)

else:

    frames_handled_input = train_data_src
    frames_handled_output = train_data_tgt

# Context:
train_frames_input = make_contexts(frames_handled_input, context)
train_frames_output = frames_handled_output[(context-1)/2:-(context-1)/2]
train_data = zip(train_frames_input, train_frames_output)
num_train = len(train_data)

startTime = time.time()
# Loop over the training instances and call the mlp
for epoch in range(num_epochs):
  start_time = time.time()
  print " Epoch ", epoch
  train_loss = 0
  recons_loss = 0
  random.shuffle(train_data)
  K = 0
  count = 0
  frame_count = 0
  for (i,o) in train_data:
       if debug and count == 0:
          print "Input dimensions: ", len(i)
          print "Number of frames: ", num_train
       count = 1
       K += 1
       dy.renew_cg()
       frame_count += 1
       count += 1
       loss = dnn.calculate_loss(dy.inputTensor(i), dy.inputTensor(o), int(tgt[-1]))
       train_loss += loss.value()
       loss.backward()
       if debug and frame_count % int(0.1*num_train) == 1:
           print "   Train Loss after processing " +  str(frame_count) + " number of frames : " +  str(float(train_loss/frame_count))
       if frame_count % update_params == 1:
            trainer.update() 
  end_time = time.time()
  duration = end_time - start_time
  start_time = end_time
  with open(logfile_name, 'a') as g:
      g.write("Train Loss after epoch " +  str(epoch) + " : " +  str(float(train_loss/frame_count)) + '\n')
  print "Train Loss after epoch " +  str(epoch) + " : " +  str(float(train_loss/frame_count)), " with ", frame_count, " frames, in ", float((end_time - startTime)/60)  , " minutes "  
  print "I think I will run for another ", float( duration * ( num_epochs - epoch) / 60 ), " minutes "
  print '\n'

  if test :
    # Test the files
    for (i_t, o_t) in test_data:
       dy.renew_cg()
       mcd = 0
       y_true = []
       y_pred = []
       if debug:
           print " Testing: ", i_t
       pred = []
       frames_test_input = np.loadtxt(i_t)
       frames_test_output = np.loadtxt(o_t)
 
       if debug:
          print "Length of source test sentence: ", len(frames_test_input)
          print "Length of target test sentence: ", len(frames_test_output)
 
       # Scaling
       if scaling:
           frames_test_input_handled = input_scaler.transform(frames_test_input)
           frames_test_output_handled = output_scaler.transform(frames_test_output)
       else:
           frames_test_input_handled = frames_test_input
           frames_test_output_handled = frames_test_output

       # Context
       test_frames_input = make_contexts(frames_test_input_handled, context)
       test_frames_output = frames_test_output_handled[(context-1)/2:-(context-1)/2]
  
       for (frame, frame_test_output_handled) in zip(test_frames_input, test_frames_output):
            t_pred = dnn.predict(dy.inputTensor(frame), int(tgt[-1]))
            if scaling:
               pred_handled = output_scaler.inverse_transform(t_pred.value())
               orig_handled = output_scaler.inverse_transform(frame_test_output_handled)
            else:
               pred_handled = t_pred.value()
               orig_handled = frame_test_output_handled
            mcd += mean_squared_error(orig_handled, pred_handled)
            pred.append(pred_handled)
     
       # Repeat last frame to account for mismatch due to context
       #pred.append(pred[-1])

       np.savetxt(save_dir + '/' +  o_t.split('/')[-1] ,pred, fmt='%.3f')

       if debug:
            print "Saved the prediction to ", save_dir + '/' +  o_t.split('/')[-1] 

       # Handle f0
       if use_predicted_f0:
           if debug:
               print "Using predicted f0 from test_outputs/", o_t.split('/')[-1]
           cmd = 'cat ' + save_dir + '/'   + o_t.split('/')[-1] + ' | cut -d " " -f 1 | x2x +af > ' + save_dir + '/' + 't.lf0 || exit 0'
           if debug:
               print cmd
           os.system(cmd)
           cmd = 'cat ' + save_dir + '/'  + o_t.split('/')[-1] + ' | cut -d " " -f 1 | x2x +af > test_outputs/t.lf0 || exit 0'
           os.system(cmd)
           if debug:
               print " Got log f0"
           cmd = 'sopr -magic -1.1E+2 -EXP -MAGIC 0.0 ' + save_dir + '/t.lf0 | x2x +fa > ' + save_dir + '/t.f0_ascii || exit 0'
           if debug:
               print cmd
           os.system(cmd)
           cmd = 'sopr -magic -1.1E+2 -EXP -MAGIC 0.0 ' + save_dir + '/t.lf0 | x2x +fa > test_outputs/t.f0_ascii || exit 0'
           os.system(cmd)
           if debug:
               print 'Handled f0'
 

       # Handle str
       if use_predicted_str:
           cmd = 'cat ' + save_dir + '/' + o_t.split('/')[-1] + ' | cut -d " " -f 52-56 | ch_track -itype ascii -s 0.005 -otype est_binary -o ' + save_dir + '/t.str'
           if debug: 
               print cmd
           os.system(cmd)
           cmd = 'cat ' + save_dir + '/' + o_t.split('/')[-1] + ' | cut -d " " -f 52-56 | ch_track -itype ascii -s 0.005 -otype est_binary -o test_outputs/t.str'
           os.system(cmd)

       # Handle mceps
       if debug:
           print "Handling mceps"
       cmd = 'cat ' + save_dir + '/'  + o_t.split('/')[-1] +  ' | cut -d " " -f 2-51 > ' + save_dir + '/t.mcep'
       if debug:
           print cmd
       os.system(cmd)
       if debug:
          print "Executed this"
       cmd = 'cat ' + save_dir + '/'  + o_t.split('/')[-1] +  ' | cut -d " " -f 2-51 > test_outputs/t.mcep'
       os.system(cmd)
       #print "Length of mceps ", len(np.loadtxt("test_outputs/t.mcep")) 

       # Combine coefficients
       cmd = 'paste -d " " ' + save_dir + '/t.f0_ascii ' + save_dir + '/t.mcep | ch_track -itype ascii -s 0.005 -otype est_binary -o ' + save_dir + '/t.coeffs'
       if debug:
          print cmd
       os.system(cmd)
       cmd = 'paste -d " " ' + save_dir + '/t.f0_ascii ' + save_dir + '/t.mcep | ch_track -itype ascii -s 0.005 -otype est_binary -o test_outputs/t.coeffs'
       os.system(cmd)

       # Copy the target lpf track
       cmd = ' cp ' + tgt + '_lpf.track lpf.track'
       if debug:
          print cmd
       os.system(cmd)        

       # Synthesize
       cmd = 'festival -b synth_simple.scm'
       if debug:
          print cmd
       os.system(cmd)
      
       # Replace
       cmd = 'cp test_outputs/new.wav ' + save_dir + '/' + i_t.split('/')[-1] + '_'  + o_t.split("/")[-1] +  '_'  + str(epoch).zfill(3) + '.wav'
       if debug:
          print cmd
       os.system(cmd)      
       if debug:
           print "Updated the wavefile"

       '''
       # Calculate the cestral distortion
       if debug:       
            print "Dumping predicted feats"       
       cmd = 'x2x +af test_outputs/t.mcep > t_pred_sptk.txt'
       if debug:
            print cmd
       os.system(cmd)
       if debug:
            print "Dumping original feats"
       cmd = 'cat '  + o_t  +  ' | cut -d " " -f 2-51 |  x2x +af  > t_orig_sptk.txt'
       if debug:
            print cmd
       os.system(cmd)
       if debug:
            print "CDIST:"
       cmd = 'cdist -m 56 -o 2 t_orig_sptk.txt < t_pred_sptk.txt | x2x +fa '
       os.system(cmd) 
       
       # Write the mcd to results file
       a = np.loadtxt('test_outputs/t.mcep')
       b = np.loadtxt(o_t, usecols=range(1,51))
       with open(results_file, 'a') as f:
        f.write('RMSE Error for ' + o_t.split('/')[-1].split('.')[0] + ' at epoch ' + str(epoch).zfill(3) + ' : ' +  str(mean_squared_error(a,b)) + '\n')
       print " Wrote report to ", results_file
       '''

       # Delete the intermediate files
       cmd = "rm -f save_dir + '/t.*'"
       os.system(cmd)
       cmd = "rm -f 'test_outputs/t.*'"
       os.system(cmd)
    print '\n\n\n'
