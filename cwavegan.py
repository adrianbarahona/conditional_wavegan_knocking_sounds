from tensorflow.keras.layers import Input, Conv1D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, LeakyReLU, ReLU, Embedding, Concatenate, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow import pad, maximum, random, int32
import librosa
import numpy as np
import os
from functools import partial
import json
from datetime import datetime
#TODO
from tensorflow.compat.v1 import disable_eager_execution
disable_eager_execution()


'''
Conditional waveGAN Keras implementation.
waveGAN: https://github.com/chrisdonahue/wavegan
Part of the GAN code from https://github.com/davidADSP/GDL_code
'''

# conv1Dtranspose
def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, padding='same'
                    , kernel_initializer = RandomNormal(mean=0., stddev=0.02), name = '1DTConv', activation = 'relu'):
    x = Conv2DTranspose(filters=filters, kernel_size=(1, kernel_size), strides=(1, strides), padding=padding, 
                        kernel_initializer = kernel_initializer, name = name, 
                        activation = activation)(K.expand_dims(input_tensor, axis=1))
    x = Lambda(lambda x: K.squeeze(x, axis=1))(x)
    return x

#phase shuffle [directly from the waveGAN paper ]
def apply_phaseshuffle(x, rad, pad_type='reflect'):
    
  b, x_len, nch = x.get_shape().as_list()
  phase = random.uniform([], minval=-rad, maxval=rad + 1, dtype=int32)
  pad_l = maximum(phase, 0)
  pad_r = maximum(-phase, 0)
  phase_start = pad_r
  x = pad(x, [[0, 0], [pad_l, pad_r], [0, 0]], mode=pad_type)

  x = x[:, phase_start:phase_start+x_len]
  x.set_shape([b, x_len, nch])

  return x

class RandomWeightedAverage(Concatenate):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
    """Provides a (random) weighted average between real and generated audio"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((self.batch_size, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


class cwavegan():
    def __init__(self
        , critic_learning_rate
        , generator_learning_rate
        , batch_size
        , n_classes
        , sample_rate_audio
        , architecture_size
        , z_size
        , use_batch_norm
        , phaseshuffle_samples
        , kernel_size):

        self.name = 'conditional_wavegan'

        #critic and generator params
        self.critic_learning_rate = critic_learning_rate
        self.generator_learning_rate = generator_learning_rate
        self.critic_filters = [64, 128, 256, 512, 1024, 2048]
        self.generator_filters = [1024, 512, 256, 128, 64]

        #z dimension
        self.z_dim = z_size
        
        #architecture size
        self.architecture_size = architecture_size
        if self.architecture_size == 'large':
            self.audio_input_dim = 65536
        elif self.architecture_size == 'medium':
            self.audio_input_dim = 32768
        elif self.architecture_size == 'small':
            self.audio_input_dim = 16384
        
        #kernel size (int)
        self.kernel_size = kernel_size

        #number of samples to shuffle the phase (int)
        self.phaseshuffle_samples = phaseshuffle_samples
        
        #use or not batch normalization (bool)
        self.use_batch_norm = use_batch_norm
        
        #audio sample rate (int)
        self.sample_rate_audio = sample_rate_audio
        
        #classes (int)
        self.n_classes = n_classes
        
        #batch size (int)
        self.batch_size = batch_size

        #weights
        self.grad_weight = 10
        self.weight_init = RandomNormal(mean=0., stddev=0.02)

        #losses
        self.d_losses = []
        self.g_losses = []

        self._build_critic()
        self._build_generator()
        self._build_adversarial()
    
    #Gradient penalty (for the WGAN-GP implementation)
    def gradient_penalty_loss(self, y_true, y_pred, interpolated_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake audio
        """
        gradients = K.gradients(y_pred, interpolated_samples)[0]
    
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)
    
    
    # critic
    def _build_critic(self):
        if self.phaseshuffle_samples > 0:
            phaseshuffle_samples = self.phaseshuffle_samples
        #label input
        label_input = Input(shape=(1,))
        label_em = Embedding(self.n_classes, 50)(label_input)
        
        nodes = self.audio_input_dim
        label_em = Dense(nodes)(label_em)
    
        # reshape to additional channel
        label_em = Reshape((self.audio_input_dim, 1))(label_em)

        #audio input
        critic_input = Input(shape=(self.audio_input_dim,1), name='critic_input')
        
        #merge
        x = Concatenate()([critic_input, label_em]) 
        
        if self.architecture_size == 'small':

            # layers 0 to 3
            for i in range(4):
                x = Conv1D(
                    filters = self.critic_filters[i]
                    , kernel_size = self.kernel_size
                    , strides = 4
                    , padding = 'same'
                    , name = 'critic_conv_' + str(i)
                    , kernel_initializer = self.weight_init
                    )(x)
                
                    
                x = LeakyReLU(alpha = 0.2)(x)
                if self.phaseshuffle_samples > 0:
                    x = Lambda(lambda x: apply_phaseshuffle(x, phaseshuffle_samples))(x)
            
            #layer 4, no phase shuffle
            x = Conv1D(
                filters = self.critic_filters[4]
                , kernel_size = self.kernel_size
                , strides = 4
                , padding = 'same'
                , name = 'critic_conv_4'
                , kernel_initializer = self.weight_init
                )(x)
            
            x = Flatten()(x)
                
        if self.architecture_size == 'medium':
            
            # layers
            for i in range(4):
                x = Conv1D(
                    filters = self.critic_filters[i]
                    , kernel_size = self.kernel_size
                    , strides = 4
                    , padding = 'same'
                    , name = 'critic_conv_' + str(i)
                    , kernel_initializer = self.weight_init
                    )(x)
                
                    
                x = LeakyReLU(alpha = 0.2)(x)
                if self.phaseshuffle_samples > 0:
                    x = Lambda(lambda x: apply_phaseshuffle(x, phaseshuffle_samples))(x)
                
                
            x = Conv1D(
                filters = self.critic_filters[4]
                , kernel_size = self.kernel_size
                , strides = 4
                , padding = 'same'
                , name = 'critic_conv_4'
                , kernel_initializer = self.weight_init
                )(x)
            
            x = LeakyReLU(alpha = 0.2)(x)
            
            x = Conv1D(
                filters = self.critic_filters[5]
                , kernel_size = self.kernel_size
                , strides = 2
                , padding = 'same'
                , name = 'critic_conv_5' 
                , kernel_initializer = self.weight_init
                )(x)
            
        
            x = LeakyReLU(alpha = 0.2)(x)
            
            x = Flatten()(x)
            
        if self.architecture_size == 'large':
            
            # layers
            for i in range(4):
                x = Conv1D(
                    filters = self.critic_filters[i]
                    , kernel_size = self.kernel_size
                    , strides = 4
                    , padding = 'same'
                    , name = 'critic_conv_' + str(i)
                    , kernel_initializer = self.weight_init
                    )(x)
                
                x = LeakyReLU(alpha = 0.2)(x)
                if self.phaseshuffle_samples > 0:
                    x = Lambda(lambda x: apply_phaseshuffle(x, phaseshuffle_samples))(x)

            #last 2 layers without phase shuffle
            
            
            x = Conv1D(
                filters = self.critic_filters[4]
                , kernel_size = self.kernel_size
                , strides = 4
                , padding = 'same'
                , name = 'critic_conv_4'
                , kernel_initializer = self.weight_init
                )(x)
            
            x = LeakyReLU(alpha = 0.2)(x)
            
            x = Conv1D(
                filters = self.critic_filters[5]
                , kernel_size = self.kernel_size
                , strides = 4
                , padding = 'same'
                , name = 'critic_conv_5'
                , kernel_initializer = self.weight_init
                )(x)
                
            x = LeakyReLU(alpha = 0.2)(x)
            
            x = Flatten()(x)
        
        critic_output = Dense(1, kernel_initializer = self.weight_init)(x)

        self.critic = Model([critic_input, label_input], critic_output)
        
    # GENERATOR
    def _build_generator(self):

        #label input
        label_input = Input(shape=(1,), dtype='int32', name='generator_input_label')
        
        # embedding 
        label_em = Embedding(self.n_classes, self.z_dim, name='embedding')(label_input)
        nodes = 16
        label_em = Dense(nodes)(label_em)
        label_em = Reshape((16, 1), name='label_reshape')(label_em)
        
        #z input
        generator_input = Input(shape=(self.z_dim,), name='generator_input')
        
        x = generator_input
        
        if self.architecture_size == 'small':
            x = Dense(16384, kernel_initializer = self.weight_init, name='generator_input_dense')(x)
            x = Reshape((16, 1024), name='generator_input_reshape')(x)
            if self.use_batch_norm == True:
                    x = BatchNormalization()(x)

        if self.architecture_size == 'medium' or self.architecture_size == 'large':
            x = Dense(32768, kernel_initializer = self.weight_init, name='generator_input_dense')(x)
            x = Reshape((16, 2048), name='generator_input_reshape')(x)
            if self.use_batch_norm == True:
                    x = BatchNormalization()(x)
            
        x = ReLU()(x)

        
        # merge
        x = Concatenate(name='concatenate_input_and_label')([x, label_em])
        
        if self.architecture_size == 'small':
            #layer 0 to 4
            for i in range(4):
                x = Conv1DTranspose(
                    input_tensor = x
                    #filter idx + 1
                    , filters = self.generator_filters[i+1]
                    , kernel_size = self.kernel_size 
                    , strides = 4
                    , padding='same'
                    , kernel_initializer = self.weight_init
                    , name = 'generator_Tconv_' + str(i)
                    , activation = 'relu'
                    )
                if self.use_batch_norm == True:
                    x = BatchNormalization()(x)
            
            #layer 4
            x = Conv1DTranspose(
                input_tensor = x
                , filters = 1
                , kernel_size = self.kernel_size
                , strides = 4
                , padding='same'
                , kernel_initializer = self.weight_init
                , name = 'generator_Tconv_4'
                , activation = 'tanh'
                )
            
            generator_output = x 
            
        if self.architecture_size == 'medium':
            #layer 0 to 4
            for i in range(5):
                x = Conv1DTranspose(
                    input_tensor = x
                    , filters = self.generator_filters[i]
                    , kernel_size = self.kernel_size 
                    , strides = 4
                    , padding='same'
                    , kernel_initializer = self.weight_init
                    , name = 'generator_Tconv_' + str(i)
                    , activation = 'relu'
                    )
                if self.use_batch_norm == True:
                    x = BatchNormalization()(x)
            #layer 5
            x = Conv1DTranspose(
                input_tensor = x
                , filters = 1
                , kernel_size = self.kernel_size
                , strides = 2
                , padding='same'
                , kernel_initializer = self.weight_init
                , name = 'generator_Tconv_5'
                , activation = 'tanh'
                )     
            
            generator_output = x 
        
        if self.architecture_size == 'large':
            #layer 0 to 4
            for i in range(5):
                x = Conv1DTranspose(
                    input_tensor = x
                    , filters = self.generator_filters[i]
                    , kernel_size = self.kernel_size 
                    , strides = 4
                    , padding='same'
                    , kernel_initializer = self.weight_init
                    , name = 'generator_Tconv_' + str(i)
                    , activation = 'relu'
                    )
                if self.use_batch_norm == True:
                    x = BatchNormalization()(x)
            
            #layer 5
            x = Conv1DTranspose(
                input_tensor = x
                , filters = 1
                , kernel_size = self.kernel_size
                , strides = 4
                , padding='same'
                , kernel_initializer = self.weight_init
                , name = 'generator_Tconv_5'
                , activation = 'tanh'
                )     
            
            generator_output = x
            
        self.generator = Model([generator_input, label_input], generator_output)


    # set critic layers trainable true/false 
    def set_trainable(self, m, val):
        m.trainable = val
        for l in m.layers:
            l.trainable = val
     
    # W loss 
    def wasserstein(self, y_true, y_pred):
        return - K.mean(y_true * y_pred)
    
    # COMPILE GAN 
    def _build_adversarial(self):
        
        # Freeze generator's layers while training critic
        self.set_trainable(self.generator, False)

        # Audio input (real audio)
        real_audio_sample = Input(shape=(self.audio_input_dim,1,))
        label = Input(shape=(1,))

        # Fake audio
        z_disc = Input(shape=(self.z_dim,))

        fake_synthesised_audio = self.generator([z_disc, label])

        # critic determines validity of the real and fake audio
        fake = self.critic([fake_synthesised_audio, label])
        valid = self.critic([real_audio_sample, label]) 
       
        # Construct weighted average between real and fake audio
        interpolated_audio = RandomWeightedAverage(self.batch_size)([real_audio_sample, fake_synthesised_audio])
        # Determine validity of weighted audio
        validity_interpolated = self.critic([interpolated_audio, label])

        # Use Python partial to provide loss function with additional
        # 'interpolated_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          interpolated_samples=interpolated_audio)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.critic_model = Model(inputs=([real_audio_sample ,label, z_disc]),
                            outputs=[valid, fake, validity_interpolated])

        self.critic_model.compile(
            loss=[self.wasserstein
                  , self.wasserstein
                  , partial_gp_loss]
            ,optimizer =Adam(lr = self.critic_learning_rate, beta_1=0.5, beta_2=0.9)
            ,loss_weights=[1, 1, self.grad_weight]
            )
        
        # For the generator we freeze the critic's layers
        self.set_trainable(self.critic, False)
        self.set_trainable(self.generator, True)

        # Sampled noise for input to generator
        model_input = Input(shape=(self.z_dim,))
        # generator label
        label_input = Input(shape=(1,))
        # Generate audio
        audio_synth = self.generator([model_input, label_input])
        # Discriminator determines validity
        model_output = self.critic([audio_synth, label_input])
        # Defines generator model
        self.model = Model([model_input, label_input], model_output)

        self.model.compile(optimizer=Adam(lr = self.critic_learning_rate, beta_1=0.5, beta_2=0.9)
        , loss=self.wasserstein
        )

        self.set_trainable(self.critic, True)


    # train critic 
    def train_critic(self, x_train, y_train):

        valid = np.ones((self.batch_size,1))
        fake = -np.ones((self.batch_size,1))
        dummy = np.zeros((self.batch_size, 1), dtype=np.float32) # Dummy gt for gradient penalty

        idx = np.random.randint(0, x_train.shape[0], self.batch_size)
        true_sounds = x_train[idx]
        labels = y_train[idx]
        
        noise = np.random.normal(0, 1, (self.batch_size, self.z_dim))

        d_loss = self.critic_model.train_on_batch([true_sounds, labels, noise], [valid, fake, dummy])

        return d_loss
    
    
    #train generator 
    def train_generator(self):
        valid = np.ones((self.batch_size,1))
        noise = np.random.normal(0, 1, (self.batch_size, self.z_dim))
        labels = np.random.randint(0, self.n_classes, self.batch_size).reshape(-1,1)
        return self.model.train_on_batch([noise, labels], valid)
    
    
    #train the whole GAN 
    def train(self, x_train, y_train, class_names, batches, run_folder, save_every_n_batches = 200, sample_every_n_batches = 50, n_critic = 5):
        
        #create date folder (for the synthesised audio) every time a new train starts
        date_folder = self.create_date_folder(run_folder)
        
        for batch in range(batches):
            for _ in range(n_critic):
                
                d_loss = self.train_critic(x_train, y_train)
        
            g_loss = self.train_generator()
        
            
            print ("%d (Critic_loops: %d) [D loss: (%.1f)(R %.1f, F %.1f, G %.1f)] [G loss: %.1f]" % (batch, n_critic, d_loss[0], d_loss[1],d_loss[2],d_loss[3],g_loss))
            
        
            self.d_losses.append(d_loss)
            self.g_losses.append(g_loss)
        
            #synthesise audio every 'sample_every_n_batches'
            if batch % sample_every_n_batches == 0:
                self.synth_sample_audio(batch, class_names, date_folder)

            #save the model every n batches
            if batch % save_every_n_batches == 0:
                self.save_model(batch, run_folder)
  
    #create folder with current date (to avoid overriding the synthesised audio when resuming the training)
    def create_date_folder(self, run_folder):
        date = datetime.now()
        day = date.strftime('%d-%m-%Y_')
        path = f'{run_folder}/audio/{day}{str(date.hour)}h'
        if not os.path.exists(path):
            os.mkdir(path)
        return path
    
    #synthesise a wav audio file for each class
    def synth_sample_audio(self, batch, class_names, date_folder):
        print(f'Synthesizing audio samples at batch {str(batch)}. Sample rate: {self.sample_rate_audio}')
        noise = np.random.normal(0,1, (1,self.z_dim))
        for label in range(self.n_classes):
            label_synth = np.array(label).reshape(-1,1)
            synth_audio = self.generator.predict([noise, label_synth])
            name_of_the_class = class_names[label]
            #save wav file
            librosa.output.write_wav(f'{date_folder}/{str(name_of_the_class)}_synth_batch_{str(batch)}.wav', y = synth_audio[0], sr = self.sample_rate_audio, norm=False)
        print(f'Audio successfully synthesised in {date_folder}')
        
    #save model
    def save_model(self, batch, run_folder):
        #check whether or not we are resuming training...
        if (os.path.exists(f'{run_folder}/weights/weights_0.h5')):
            weights_path = f'{run_folder}/weights/'
            weights_batch = []
            for file in os.listdir(weights_path):
                number = file.split("_")[1]
                number = number.split(".")[0]
                weights_batch.append(int(number))
            last_batch = (max(weights_batch))
        else:
            last_batch = 0

        print(f'Saving model at batch {str(batch+last_batch)}...')
        self.model.save(os.path.join(run_folder, f'model_{str(batch+last_batch)}.h5'))
        self.critic.save(os.path.join(run_folder, f'critic_{str(batch+last_batch)}.h5'))
        self.generator.save(os.path.join(run_folder, f'generator_{str(batch+last_batch)}.h5'))
        self.model.save_weights(os.path.join(run_folder, f'weights/weights_{str(batch+last_batch)}.h5'))
        print('Model saved.')

    #Load the data (X samples of audio, resampled to sample_rate_audio Hz, with labels from the folder)
    def load_audio(self, path_to_wavfiles, run_folder):
        label_names = {}
        x_train = []
        y_train = []
        n_samples_file = self.audio_input_dim
        for folder in next(os.walk(path_to_wavfiles))[1]:
          for wavfile in os.listdir(path_to_wavfiles+folder):
              y_train.append(folder)
              loaded_audio, _ = librosa.load(path_to_wavfiles+folder+'/'+wavfile, sr = self.sample_rate_audio)
              if loaded_audio.size < n_samples_file:
                  padding = n_samples_file - loaded_audio.size
                  audio = np.pad(loaded_audio, (0, padding), mode = 'constant')
              else:
                  audio = loaded_audio[0:n_samples_file]
              x_train.append(audio)
        y_train_label = np.unique(y_train, return_inverse=True)[1]
        y_train_label = np.expand_dims(y_train_label, 1)
        x_train_np = np.asarray(x_train)
        x_train_np = np.expand_dims(x_train_np, 2)
        #get the class names as a dictionary
        i = 0
        for folder in next(os.walk(path_to_wavfiles))[1]:
          label_names[i] = folder
          i += 1
        #save the dictionary to use it later with the standalone generator
        with open(os.path.join(run_folder, 'label_names.json'), 'w') as outfile:
            json.dump(label_names, outfile)
            
        return x_train_np, y_train_label, label_names
        