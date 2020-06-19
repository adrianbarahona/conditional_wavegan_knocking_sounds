import tensorflow as tf
from tensorflow import keras
import numpy as np
import librosa
import time

#Baseline WGANGP model directly from the Keras documentation: https://keras.io/examples/generative/wgan_gp/
#Original WaveGAN: https://github.com/chrisdonahue/wavegan

class WGANGP(keras.Model):
    def __init__(
        self,
        latent_dim,
        discriminator,
        generator,
        n_classes,
        discriminator_extra_steps=5,
        gp_weight=10.0,
        d_optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0004),
        g_optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0004)
    ):
        super(WGANGP, self).__init__()
        self.latent_dim = latent_dim
        self.discriminator = discriminator
        self.generator = generator
        self.n_classes = n_classes
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight
        self.d_optimizer=d_optimizer
        self.g_optimizer=g_optimizer

    def compile(self, d_optimizer, g_optimizer):
        super(WGANGP, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = self.discriminator_loss
        self.g_loss_fn = self.generator_loss      
    
    # Define the loss functions to be used for discriminator
    # This should be (fake_loss - real_loss)
    # We will add the gradient penalty later to this loss function
    def discriminator_loss(self, real_img, fake_img):
        real_loss = tf.reduce_mean(real_img)
        fake_loss = tf.reduce_mean(fake_img)
        return fake_loss - real_loss
    
    # Define the loss functions to be used for generator
    def generator_loss(self, fake_img):
        return -tf.reduce_mean(fake_img)
    
    def gradient_penalty(self, batch_size, real_images, fake_images, labels):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # get the interplated image
        alpha = tf.random.normal([batch_size, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator([interpolated, labels], training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calcuate the norm of the gradients
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp
    
    def train_batch(self, x, y, batch_size):
        #get a random indexes for the batch
        idx = np.random.randint(0, x.shape[0], batch_size)
        real_images = x[idx]
        labels = y[idx]
        
        # For each batch, we are going to perform the
        # following steps as laid out in the original paper.
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add gradient penalty to the discriminator loss
        # 6. Return generator and discriminator losses as a loss dictionary.

        # Train discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator.
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )

            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator([random_latent_vectors, labels], training=True)
                # Get the logits for the fake images
                fake_logits = self.discriminator([fake_images, labels], training=True)
                # Get the logits for real images
                real_logits = self.discriminator([real_images, labels], training=True)
                # Calculate discriminator loss using fake and real logits
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images, labels) 
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )
            
        # Train the generator now.
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator([random_latent_vectors, labels], training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator([generated_images, labels], training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return d_loss, g_loss
    
    def train(self, x, y, batch_size, batches, synth_frequency, save_frequency,
              sampling_rate, n_classes, checkpoints_path, override_saved_model):
        
        for batch in range(batches):
            start_time = time.time()
            d_loss, g_loss = self.train_batch(x, y, batch_size)
            end_time = time.time()
            time_batch = (end_time - start_time)
            print(f'Batch: {batch} == Batch size: {batch_size} == Time elapsed: {time_batch:.2f} == d_loss: {d_loss:.4f}, g_loss: {g_loss:.4f}')
            
            #This works as a callback
            if batch % synth_frequency == 0 :
                print(f'Synthesising audio at batch {batch}. Path: {checkpoints_path}/synth_audio')
                random_latent_vectors = tf.random.normal(shape=(1, self.latent_dim))
                for i in range (n_classes):
                    generated_audio = self.generator([random_latent_vectors, np.array(i).reshape(-1,1)])
                    librosa.output.write_wav(f'{checkpoints_path}/synth_audio/{batch}_batch_synth_class_{i}.wav', 
                                             y = tf.squeeze(generated_audio).numpy(), sr = sampling_rate, norm=False)
                print(f'Done.')
                
            if batch % save_frequency == 0:
                print(f'Saving the model at batch {batch}. Path: {checkpoints_path}')
                if override_saved_model == False:
                    self.generator.save(f'{checkpoints_path}/{batch}_batch_generator.h5')
                    self.discriminator.save(f'{checkpoints_path}/{batch}_batch_discriminator.h5')
                    self.save_weights(f'{checkpoints_path}/{batch}_batch_weights.h5')
                else:
                    self.generator.save(f'{checkpoints_path}/generator.h5')
                    self.discriminator.save(f'{checkpoints_path}/discriminator.h5')
                    self.save_weights(f'{checkpoints_path}/model_weights.h5')
                print(f'Model saved.')