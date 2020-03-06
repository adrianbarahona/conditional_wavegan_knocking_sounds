#### Keras implementation of conditional waveGAN. Application to knocking sound effects.

Original waveGAN architecture: https://github.com/chrisdonahue/wavegan

#### Requirements
```
Tensorflow >= 2.0
Librosa
```

#### Training

##### Using the knocking sound effects with emotion dataset or your own sounds.

We focused on knocking sound effects and recorded a dataset to train the model. If you want to train your model on the knocking sound effects with emotion dataset you can download it from [here](https://zenodo.org/record/3668503) and put it on an '/audio' subdirectory.

If you want to use your own sounds, just place your .wav files (organised in folders for your labels) on an '/audio' subdirectory. If for instance you want to train the conditional waveGAN on footsteps on concrete and grass, put your sounds in '/audio/concrete' and '/audio/grass'.

##### Training

To train the model using the recommended parameters for the knocking sound effects dataset run:

```
python train_cwavegan.py --run_folder=train --resume_training=0 --path_to_wavfiles=audio/ --batches=50000 --batch_size=1 --critic_loops=5 --sample_rate_audio=22050 --sample_every_n_batches=200 --save_every_n_batches=500 --architecture_size=large --critic_learning_rate=0.0002 --generator_learning_rate=0.0002 --z_size=100 --use_batch_norm=1 --phaseshuffle_samples=0 --kernel_size=25
```

If you are using your own sounds please run ``` python train_cwavegan.py -h ``` to check what the different parameters do. You will probably want to change (at least) the architecture size and the sampling rate.

##### Synthesising audio

Once the model is trained, just use the trained generator. You can find an example on how to use it on the generation notebook.
