import cwavegan
import os
import argparse
from datetime import datetime
import json

def traincwavegan(run_folder = 'train'
                 , resume_training = 0
                 , path_to_wavfiles = 'audio/'
                 , batches = 50000
                 , batch_size = 128
                 , critic_loops = 5
                 , sample_rate_audio = 22050
                 , sample_every_n_batches = 200   
                 , save_every_n_batches = 400             
                 , architecture_size = 'large'         
                 , critic_learning_rate = 0.0002
                 , generator_learning_rate = 0.0002
                 , z_size = 100
                 , use_batch_norm = 1
                 , phaseshuffle_samples = 0
                 , kernel_size = 25):
    
    #create the train folder (if it does not exist)
    if not os.path.exists(run_folder):
        os.mkdir(run_folder)
        os.mkdir(os.path.join(run_folder, 'audio'))
        os.mkdir(os.path.join(run_folder, 'weights'))

    #get the number of classes from the number of folders inside the wavfile directory
    n_classes = 0
    for folder in next(os.walk(path_to_wavfiles))[1]:
      n_classes += 1
    
    #build GAN
    gan = cwavegan.cwavegan(critic_learning_rate = critic_learning_rate
                          , generator_learning_rate = generator_learning_rate
                          , batch_size = batch_size
                          , n_classes = n_classes
                          , sample_rate_audio = sample_rate_audio
                          , architecture_size = architecture_size
                          , z_size = z_size
                          , use_batch_norm = use_batch_norm
                          , phaseshuffle_samples = phaseshuffle_samples
                          , kernel_size = kernel_size)
                          
    
    #load the latest weights (if resuming the training)
    if resume_training:
        weights_path = f'{run_folder}/weights/'
        weights_batch = []
        for file in os.listdir(weights_path):
            number = file.split("_")[1]
            number = number.split(".")[0]
            weights_batch.append(int(number))
        last_batch = (max(weights_batch))
        most_recent_weights = f'{weights_path}weights_{last_batch}.h5'
        print(f'Loading the weights from {most_recent_weights}...')
        gan.model.load_weights(most_recent_weights)
        print('Weights loaded.')

    #get the training data
    x_train, y_train, class_names = gan.load_audio(path_to_wavfiles = path_to_wavfiles, run_folder = run_folder)
    
    #print the architecture
    gan.critic.summary()
    gan.generator.summary()

    #train the GAN
    gan.train(x_train = x_train, y_train = y_train, class_names = class_names, batches = batches, run_folder = run_folder
              , save_every_n_batches = save_every_n_batches, sample_every_n_batches = sample_every_n_batches, n_critic = critic_loops)


#get the arguments from the command line
parser = argparse.ArgumentParser(description="Implements conditional WaveGAN with class conditioning in Keras")

parser.add_argument("--run_folder", help="Path to create the folder where the model will store the trained weights and synthesised audio.", default='train', type=str)
parser.add_argument("--resume_training", help="Load weights and model to resume training. 0=False, 1=True", default=0, type=int)
parser.add_argument("--path_to_wavfiles", help="Path to the root folder where the training data (wav files in their class folders) is located.", default='audio/', type=str)
parser.add_argument("--batches", help="Number of batches.", default=50000, type=int)
parser.add_argument("--batch_size", help="Batch_size.", default=8, type=int)
parser.add_argument("--critic_loops", help="Critic traning loops per discriminator training.", default=5, type=int)
parser.add_argument("--sample_rate_audio", help="Sample rate of loaded/synthesised audio.", default=22050, type=int)
parser.add_argument("--sample_every_n_batches", help="Synthesise a sample every N batches", default=100, type=int)
parser.add_argument("--save_every_n_batches", help="Save the model every N batches", default=500, type=int)
parser.add_argument("--architecture_size", help="Number of samples the architecture handles. 'small' = 16384, 'medium' = 32768, 'large' = 65536", default='large', type=str)
parser.add_argument("--critic_learning_rate", help="Discriminator (critic) learning rate.", default=0.0002, type=float)
parser.add_argument("--generator_learning_rate", help="Generator learning rate.", default=0.0002, type=float)
parser.add_argument("--z_size", help="Dimension of the latent vector Z", default=100, type=int)
parser.add_argument("--use_batch_norm", help="Using batch normalisation. 0=False, 1=True", default=1, type=int)
parser.add_argument("--phaseshuffle_samples", help="How many samples of phase shuffle (0 = disabled)", default=0, type=int)
parser.add_argument("--kernel_size", help="Kernel size.", default=25, type=int)
#parse arguments
args = parser.parse_args()
print(args)

#save args to a txt file
date = datetime.now()
day = date.strftime('%d-%m-%Y_')
full_date_json = f'args_{day}{str(date.hour)}h.json'
with open(full_date_json, 'w') as outfile:
    json.dump(str(args), outfile)

# train the model
traincwavegan(run_folder = args.run_folder
                 , critic_learning_rate = args.critic_learning_rate
                 , generator_learning_rate = args.generator_learning_rate
                 , batch_size = args.batch_size
                 , path_to_wavfiles = args.path_to_wavfiles
                 , batches = args.batches
                 , save_every_n_batches = args.save_every_n_batches
                 , sample_every_n_batches = args.sample_every_n_batches
                 , sample_rate_audio = args.sample_rate_audio
                 , resume_training = args.resume_training
                 , architecture_size = args.architecture_size
                 , z_size = args.z_size
                 , use_batch_norm = args.use_batch_norm
                 , phaseshuffle_samples = args.phaseshuffle_samples
                 , kernel_size = args.kernel_size
                 , critic_loops = args.critic_loops)
