#### Keras implementation of conditional waveGAN. Application to knocking sound effects.

Original waveGAN architecture: https://github.com/chrisdonahue/wavegan

#### Requirements
```
Tensorflow >= 2.0
Librosa
```

##### Using the knocking sound effects with emotion dataset or your own sounds.

We focused on knocking sound effects and recorded a dataset to train the model. If you want to train your model on the knocking sound effects with emotion dataset you can download it from [here](https://zenodo.org/record/3668503) and put it on an '/audio' subdirectory.

If you want to use your own sounds, just place your .wav files (organised in folders for your labels) on an '/audio' subdirectory. If for instance you want to train the conditional waveGAN on footsteps on concrete and grass, put your sounds in '/audio/concrete' and '/audio/grass'.

##### Training

To train the model just call ``` train_cwavegan.py ```. You can see and edit the parameters/hyperparameters of the model directly in the python file. Depending on your dataset you will probably want to change (at least) the architecture size and the sampling rate.  Once you start training, you will find a date/time folder in the ```checkpoints``` directory. Inside you will find your saved model, a file with the list of the parameters used and a dictionary with the labels (for inference).


##### Synthesising audio

Once the model is trained, just use the trained generator. You can find an example on how to use it on the generation notebook.