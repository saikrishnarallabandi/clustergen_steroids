import sys
sys.path.append('/home/srallaba/hacks/repos/clustergen_steroids')
from dynet_modules.AutoEncoders import *
import dynet as dy
import numpy as np
from math import sqrt
from sklearn import preprocessing
import random
from sklearn.metrics import mean_squared_error 

src = sys.argv[1]
src_name = sys.argv[1].split('/')[-1]
enc = sys.argv[2]
bot = sys.argv[3]

arch = str(enc) + 'T' + str(bot) + 'T' + str(enc) + 'T'
train_input = np.loadtxt(src)

# Preprocess the data
input_scaler = preprocessing.StandardScaler()
a = input_scaler.fit_transform(train_input)

num_examples = len(a)
num_toprint = 0.1 * num_examples

# Hyperparameters for the AE
units_input = 50
units_hidden = int(enc)
units_output = 50


# Instantiate AE and define the loss
m = dy.Model()
ae = VariationalAutoEncoder(m, units_input, units_hidden, units_output, dy.rectify)
trainer = dy.AdamTrainer(m)


# Loop over the training instances and call the AE
for epoch in range(30):
  train_loss = 0
  count = 1
  random.shuffle(a)
  for (i,o) in zip(a,a):
     dy.renew_cg()
     count += 1
     I = dy.inputTensor(i)
     O = dy.inputTensor(o)
     kl_loss, recons_loss = ae.calc_loss_basic(i,o)
     loss = dy.esum([kl_loss, recons_loss])
     train_loss += loss.value()
     if count % num_toprint == 1:
         print "  Loss at epoch ", epoch, " after " , count, " examples is ", float(train_loss/count)
         ae.save('models/' +  arch) 
         pickle.dump(input_scaler, open('models/' + arch + '/input_scaler', 'wb'))
         '''      
         # Test a random file and print the results
         rint = random.randint(0, 30)
         file_num = str(10*rint + 9)
         filename = '../../mcep_deltas_ascii/tel_' + file_num.zfill(4) + '.mcep'
         print "Testing on ", filename
         test_input = np.loadtxt(filename)
         test_input_scaled = input_scaler.transform(test_input)
         mse = 0
         for te in test_input_scaled:
               output_unscaled = ae.predict(te).value()
               #print output_unscaled
               output_scaled = input_scaler.inverse_transform(output_unscaled)
               mse +=  mean_squared_error(te, output_scaled)
         print "Test Error: ", float(mse)/len(test_input_scaled)
         '''
     loss.backward()
     if count % 100 == 1:
        trainer.update() 
  print "KL Loss after epoch ", epoch , " : ", float(kl_loss.value()/count)
  print "Reconstruction Loss after epoch ", epoch , " : ", float(recons_loss.value()/count)
  print "Total Loss: ", float(train_loss/count)
  
  # Test on all
  filename = 'tel_rsk.mcep_deltas.test'  
  test_input = np.loadtxt(filename)
  test_input_scaled = input_scaler.transform(test_input)
  mse = 0
  for te in test_input_scaled:
     output_unscaled = ae.predict(te).value()
     output_scaled = input_scaler.inverse_transform(output_unscaled)
     mse +=  mean_squared_error(te, output_scaled)
  print "Test Error: ", float(mse)/len(test_input_scaled)      

  ae.save('models/' + arch + '_latestepoch')
  print '\n'
