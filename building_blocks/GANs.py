from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import sys
import numpy as np

class GAN_keras():

     def __init__(self):
         self.optimizer = Adam(0.0002, 0.5)
         
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', 
            optimizer=self.optimizer,
            metrics=['accuracy'])

       # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='mse', optimizer=self.optimizer)        

        # The generator takes noise as input and generates audio
        z = Input(shape=(56,))
        frame = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated frames as input and determines validity
        valid = self.discriminator(frame)              

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates frames => determines validity 
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)



    def build_generator(self):

        noise_shape = (56,)
        
        model = Sequential()

        model.add(Dense(256, input_shape=noise_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(56, activation='tanh'))

        model.summary()

        noise = Input(shape=noise_shape)
        frame = model(noise)

        return Model(noise, frame)


    def build_discriminator(self):

        
        model = Sequential()

        model.add(Dropout(0.0, input_shape=(56,)))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        frame = Input(shape=56)
        validity = model(frame)

        return Model(frame, validity)

   def calculate_loss(self, epochs, batch_size=128, input_frames, output_frames):
  
        for epoch in range(epochs):
            
             half_batch = int(batch_size / 2)

             idx = np.random.randint(0, input_frames.shape[0], half_batch)
             frames = input_frames[idx]

             noise = np.random.normal(0, 1, (half_batch, 100))

             # Generate a half batch of new images
             gen_frames = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(frames, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_frames, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            ## Train the generator
            noise = np.random.normal(0, 1, (batch_size, 56))

           # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)
        




