import cwawegan_architecture
import wgangp
import utils
from tensorflow.keras.optimizers import Adam

def train_model(sampling_rate = 22050,
                n_batches = 10000,
                batch_size = 128,
                audio_path = 'audio/',
                checkpoints_path = 'checkpoints/',
                architecture_size = 'large',
                resume_training = False,
                path_to_weights = 'checkpoints/model_weights.h5',
                override_saved_model = False,
                synth_frequency = 200,
                save_frequency = 200,
                latent_dim = 100,
                use_batch_norm = False,
                discriminator_learning_rate = 0.00004,
                generator_learning_rate = 0.00004,
                discriminator_extra_steps = 5,
                phaseshuffle_samples = 0):
    
    '''
    Train the conditional WaveGAN architecture.
    Args:
        sampling_rate (int): Sampling rate of the loaded/synthesised audio.
        n_batches (int): Number of batches to train for.
        batch_size (int): batch size (for the training process).
        audio_path (str): Path where your training data (wav files) are store. 
            Each class should be in a folder with the class name
        checkpoints_path (str): Path to save the model / synth the audio during training
        architecture_size (str) = size of the wavegan architecture. Eeach size processes the following number 
            of audio samples: 'small' = 16384, 'medium' = 32768, 'large' = 65536"
        resume_training (bool) = Restore the model weights from a previous session?
        path_to_weights (str) = Where the model weights are (when resuming training)
        override_saved_model (bool) = save the model overwriting 
            the previous saved model (in a past epoch)?. Be aware the saved files could be large!
        synth_frequency (int): How often do you want to synthesise a sample during training (in batches).
        save_frequency (int): How often do you want to save the model during training (in batches).
        latent_dim (int): Dimension of the latent space.
        use_batch_norm (bool): Use batch normalization?
        discriminator_learning_rate (float): Discriminator learning rate.
        generator_learning_rate (float): Generator learning rate.
        discriminator_extra_steps (int): How many steps the discriminator is trained per step of the generator.
        phaseshuffle_samples (int): Discriminator phase shuffle. 0 for no phases shuffle.
    '''
    
    #get the number of classes from the audio folder
    n_classes = utils.get_n_classes(audio_path)
    
    #build the discriminator
    discriminator = cwawegan_architecture.discriminator(architecture_size=architecture_size,
                                                    phaseshuffle_samples = phaseshuffle_samples,
                                                    n_classes = n_classes)
    #build the generator
    generator = cwawegan_architecture.generator(architecture_size=architecture_size,
                                                z_dim = latent_dim,
                                                use_batch_norm = use_batch_norm,
                                                n_classes = n_classes)
    #set the optimizers
    discriminator_optimizer = Adam(learning_rate = discriminator_learning_rate)
    generator_optimizer = Adam(learning_rate = generator_learning_rate)
    
    #build the gan
    gan = wgangp.WGANGP(latent_dim=latent_dim, discriminator=discriminator, generator=generator,
                    n_classes = n_classes, discriminator_extra_steps = discriminator_extra_steps,
                    d_optimizer = discriminator_optimizer, g_optimizer = generator_optimizer)

    # Compile the wgan model
    gan.compile(
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer)

    #make a folder with the current date to store the current session to
    #avoid overriding past synth audio files and checkpoints
    checkpoints_path = utils.create_date_folder(checkpoints_path)
    
    #save the training parameters used to the checkpoints folder,
    #it makes it easier to retrieve the parameters/hyperparameters afterwards
    utils.write_parameters(sampling_rate, n_batches, batch_size, audio_path, checkpoints_path, architecture_size,
                path_to_weights, resume_training, override_saved_model, synth_frequency, save_frequency,
                latent_dim, use_batch_norm, discriminator_learning_rate, generator_learning_rate,
                discriminator_extra_steps, phaseshuffle_samples)
    
    #create the dataset from the class folders in '/audio'
    audio, labels = utils.create_dataset(audio_path, sampling_rate, architecture_size, checkpoints_path)

    #load the desired weights in path (if resuming training)
    if resume_training == True:
        print(f'Resuming training. Loading weights in {path_to_weights}')
        gan.load_weights(path_to_weights)
    
    #train the gan for the desired number of batches
    gan.train(x = audio, y = labels, batch_size = batch_size, batches = n_batches, 
                 synth_frequency = synth_frequency, save_frequency = save_frequency,
                 checkpoints_path = checkpoints_path, override_saved_model = override_saved_model,
                 sampling_rate = sampling_rate, n_classes = n_classes)


if __name__ == '__main__':
    train_model(sampling_rate = 22050,
                n_batches = 30000,
                batch_size = 128,
                audio_path = 'audio/',
                checkpoints_path = 'checkpoints/',
                architecture_size = 'large',
                path_to_weights = 'model_weights.h5',
                resume_training = False,
                override_saved_model = True,
                synth_frequency = 200,
                save_frequency = 200,
                latent_dim = 100,
                use_batch_norm = True,
                discriminator_learning_rate = 0.0002,
                generator_learning_rate = 0.0002,
                discriminator_extra_steps = 5,
                phaseshuffle_samples = 0)