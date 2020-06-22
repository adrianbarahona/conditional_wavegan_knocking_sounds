from tensorflow.keras.layers import Input, Conv1D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, LeakyReLU, ReLU, Embedding, Concatenate, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow import pad, maximum, random, int32

#Original WaveGAN: https://github.com/chrisdonahue/wavegan
#Label embeding using the method in https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/

#phase shuffle [directly from the original waveGAN implementation]
def apply_phaseshuffle(args):
  x, rad = args
  pad_type = 'reflect'
  b, x_len, nch = x.get_shape().as_list()
  phase = random.uniform([], minval=-rad, maxval=rad + 1, dtype=int32)
  pad_l = maximum(phase, 0)
  pad_r = maximum(-phase, 0)
  phase_start = pad_r
  x = pad(x, [[0, 0], [pad_l, pad_r], [0, 0]], mode=pad_type)

  x = x[:, phase_start:phase_start+x_len]
  x.set_shape([b, x_len, nch])

  return x


#TODO: clean/redo this
def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, padding='same'
                    , name = '1DTConv', activation = 'relu'):
    x = Conv2DTranspose(filters=filters, kernel_size=(1, kernel_size), strides=(1, strides), padding=padding, 
                        name = name, activation = activation)(K.expand_dims(input_tensor, axis=1))
    x = K.squeeze(x, axis=1)
    return x

def generator(z_dim = 100,
              architecture_size = 'large',
              use_batch_norm = False,
              n_classes = 5):
        
    generator_filters = [1024, 512, 256, 128, 64]
    
    label_input = Input(shape=(1,), dtype='int32', name='generator_label_input')
    label_em = Embedding(n_classes, n_classes * 20, name = 'label_embedding')(label_input)
    label_em = Dense(16, name = 'label_dense')(label_em)
    label_em = Reshape((16, 1), name = 'label_respahe')(label_em)
    
    generator_input = Input(shape=(z_dim,), name='generator_input')
    x = generator_input
    
    if architecture_size == 'small':
        x = Dense(16384, name='generator_input_dense')(x)
        x = Reshape((16, 1024), name='generator_input_reshape')(x)
        if use_batch_norm == True:
                x = BatchNormalization()(x)

    if architecture_size == 'medium' or architecture_size == 'large':
        x = Dense(32768, name='generator_input_dense')(x)
        x = Reshape((16, 2048), name='generator_input_reshape')(x)
        if use_batch_norm == True:
                x = BatchNormalization()(x)
        
    x = ReLU()(x)
    
    x = Concatenate()([x, label_em]) 
    
    if architecture_size == 'small':
        for i in range(4):
            x = Conv1DTranspose(
                input_tensor = x
                , filters = generator_filters[i+1]
                , kernel_size = 25 
                , strides = 4
                , padding='same'
                , name = f'generator_Tconv_{i}'
                , activation = 'relu'
                )
            if use_batch_norm == True:
                x = BatchNormalization()(x)
                
        x = Conv1DTranspose(
            input_tensor = x
            , filters = 1
            , kernel_size = 25 
            , strides = 4
            , padding='same'
            , name = 'generator_Tconv_4'
            , activation = 'tanh'
            )
    
    if architecture_size == 'medium':
        #layer 0 to 4
        for i in range(5):
            x = Conv1DTranspose(
                input_tensor = x
                , filters = generator_filters[i]
                , kernel_size = 25 
                , strides = 4
                , padding='same'
                , name = f'generator_Tconv_{i}'
                , activation = 'relu'
                )
            if use_batch_norm == True:
                x = BatchNormalization()(x)
        #layer 5
        x = Conv1DTranspose(
            input_tensor = x
            , filters = 1
            , kernel_size = 25
            , strides = 2
            , padding='same'
            , name = 'generator_Tconv_5'
            , activation = 'tanh'
            )     
    
    
    if architecture_size == 'large':
        #layer 0 to 4
        for i in range(5):
            x = Conv1DTranspose(
                input_tensor = x
                , filters = generator_filters[i]
                , kernel_size = 25
                , strides = 4
                , padding='same'
                , name = f'generator_Tconv_{i}'
                , activation = 'relu'
                )
            if use_batch_norm == True:
                x = BatchNormalization()(x)
        
        #layer 5
        x = Conv1DTranspose(
            input_tensor = x
            , filters = 1
            , kernel_size = 25
            , strides = 4
            , padding='same'
            , name = 'generator_Tconv_5'
            , activation = 'tanh'
            ) 
    
    generator_output = x 
    generator = Model([generator_input, label_input], generator_output, name = 'Generator')
    return generator

def discriminator(architecture_size='small',
                  phaseshuffle_samples = 0,
                  n_classes = 5):
    
    discriminator_filters = [64, 128, 256, 512, 1024, 2048]
    
    if architecture_size == 'large':
        audio_input_dim = 65536
    elif architecture_size == 'medium':
        audio_input_dim = 32768
    elif architecture_size == 'small':
        audio_input_dim = 16384
        
    label_input = Input(shape=(1,), dtype='int32', name='discriminator_label_input')
    label_em = Embedding(n_classes, n_classes * 20)(label_input)
    label_em = Dense(audio_input_dim)(label_em)
    label_em = Reshape((audio_input_dim, 1))(label_em)
    
    discriminator_input = Input(shape=(audio_input_dim,1), name='discriminator_input')
    x = Concatenate()([discriminator_input, label_em]) 

    if architecture_size == 'small':
        # layers 0 to 3
        for i in range(4):
            x = Conv1D(
                filters = discriminator_filters[i]
                , kernel_size = 25
                , strides = 4
                , padding = 'same'
                , name = f'discriminator_conv_{i}'
                )(x)
            
            x = LeakyReLU(alpha = 0.2)(x)
            if phaseshuffle_samples > 0:
                x = Lambda(apply_phaseshuffle)([x, phaseshuffle_samples])
                
        #layer 4, no phase shuffle
        x = Conv1D(
            filters = discriminator_filters[4]
            , kernel_size = 25
            , strides = 4
            , padding = 'same'
            , name = f'discriminator_conv_4'
            )(x)
            
        x = Flatten()(x)
            
    if architecture_size == 'medium':
        
        # layers
        for i in range(4):
            x = Conv1D(
                filters = discriminator_filters[i]
                , kernel_size = 25
                , strides = 4
                , padding = 'same'
                , name = f'discriminator_conv_{i}'
                )(x)
            
                
            x = LeakyReLU(alpha = 0.2)(x)
            if phaseshuffle_samples > 0:
                x = Lambda(apply_phaseshuffle)([x, phaseshuffle_samples])
            
            
        x = Conv1D(
            filters = discriminator_filters[4]
            , kernel_size = 25
            , strides = 4
            , padding = 'same'
            , name = 'discriminator_conv_4'
            )(x)
        
        x = LeakyReLU(alpha = 0.2)(x)
        
        x = Conv1D(
            filters = discriminator_filters[5]
            , kernel_size = 25
            , strides = 2
            , padding = 'same'
            , name = 'discriminator_conv_5' 
            )(x)
        
    
        x = LeakyReLU(alpha = 0.2)(x)
        x = Flatten()(x)
    
    if architecture_size == 'large':
        
        # layers
        for i in range(4):
            x = Conv1D(
                filters = discriminator_filters[i]
                , kernel_size = 25
                , strides = 4
                , padding = 'same'
                , name = f'discriminator_conv_{i}'
                )(x)
            x = LeakyReLU(alpha = 0.2)(x)
            if phaseshuffle_samples > 0:
                x = Lambda(apply_phaseshuffle)([x, phaseshuffle_samples])

        #last 2 layers without phase shuffle
        x = Conv1D(
            filters = discriminator_filters[4]
            , kernel_size = 25
            , strides = 4
            , padding = 'same'
            , name = 'discriminator_conv_4'
            )(x)
        x = LeakyReLU(alpha = 0.2)(x)
        
        x = Conv1D(
            filters = discriminator_filters[5]
            , kernel_size = 25
            , strides = 4
            , padding = 'same'
            , name = 'discriminator_conv_5'
            )(x)
        x = LeakyReLU(alpha = 0.2)(x)
        x = Flatten()(x)
        
    discriminator_output = Dense(1)(x)
    discriminator = Model([discriminator_input, label_input], discriminator_output, name = 'Discriminator')
    return discriminator