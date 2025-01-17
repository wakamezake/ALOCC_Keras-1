from __future__ import print_function, division

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import BatchNormalization
from keras.layers import Input, Dense, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
from keras.optimizers import RMSprop

from kh_tools import get_noisy_data


class AloccModel:
    def __init__(self, data, checkpoint_dir, sample_dir,
                 input_height=28, input_width=28, output_height=28, output_width=28, attention_label=1,
                 z_dim=100, gf_dim=16, df_dim=16, c_dim=1, r_alpha=0.2):
        """
        This is the main class of our Adversarially Learned One-Class Classifier for Novelty Detection.
        :param input_height: The height of image to use.
        :param input_width: The width of image to use.
        :param output_height: The height of the output images to produce.
        :param output_width: The width of the output images to produce.
        :param attention_label: Conditioned label that growth attention of training label [1]
        :param z_dim:  (optional) Dimension of dim for Z, the output of encoder. [100]
        :param gf_dim: (optional) Dimension of gen filters in first conv layer, i.e. g_decoder_h0. [16] 
        :param df_dim: (optional) Dimension of discrim filters in first conv layer, i.e. d_h0_conv. [16] 
        :param c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        :param checkpoint_dir: path to saved checkpoint(s) directory.
        :param sample_dir: Directory address which save some samples [.]
        :param r_alpha: Refinement parameter, trade-off hyperparameter
         for the G network loss to reconstruct input images. [0.2]
        """
        self.r_alpha = r_alpha

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.z_dim = z_dim
        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.checkpoint_path = checkpoint_dir
        self.sample_path = sample_dir

        self.attention_label = attention_label
        self.c_dim = c_dim
        self.data = data

        self.discriminator = None
        self.generator = None
        self.adversarial_model = None
        self.build_model()

    def build_generator(self, input_shape):
        """Build the generator/R network.
        
        Arguments:
            input_shape {list} -- Generator input shape.
        
        Returns:
            [Tensor] -- Output tensor of the generator/R network.
        """
        image = Input(shape=input_shape, name='z')
        # Encoder.
        x = Conv2D(filters=self.df_dim * 2, kernel_size=5, strides=2, padding='same', name='g_encoder_h0_conv')(image)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2D(filters=self.df_dim * 4, kernel_size=5, strides=2, padding='same', name='g_encoder_h1_conv')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2D(filters=self.df_dim * 8, kernel_size=5, strides=2, padding='same', name='g_encoder_h2_conv')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # Decoder.
        # TODO: need a flexable solution to select output_padding and padding.
        # x = Conv2DTranspose(self.gf_dim*2, kernel_size = 5, strides=2, activation='relu', padding='same',
        # output_padding=0, name='g_decoder_h0')(x)
        # x = BatchNormalization()(x)
        # x = Conv2DTranspose(self.gf_dim*1, kernel_size = 5, strides=2, activation='relu', padding='same',
        # output_padding=1, name='g_decoder_h1')(x)
        # x = BatchNormalization()(x)
        # x = Conv2DTranspose(self.c_dim,    kernel_size = 5, strides=2, activation='tanh', padding='same',
        # output_padding=1, name='g_decoder_h2')(x)

        x = Conv2D(self.gf_dim * 1, kernel_size=5, activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(self.gf_dim * 1, kernel_size=5, activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(self.gf_dim * 2, kernel_size=3, activation='relu')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(self.c_dim, kernel_size=5, activation='sigmoid', padding='same')(x)
        return Model(image, x, name='R')

    def build_discriminator(self, input_shape):
        """Build the discriminator/D network
        
        Arguments:
            input_shape {list} -- Input tensor shape of the discriminator network, either the real unmodified image
                or the generated image by generator/R network.
        
        Returns:
            [Tensor] -- Network output tensors.
        """

        image = Input(shape=input_shape, name='d_input')
        x = Conv2D(filters=self.df_dim, kernel_size=5, strides=2, padding='same', name='d_h0_conv')(image)
        x = LeakyReLU()(x)

        x = Conv2D(filters=self.df_dim * 2, kernel_size=5, strides=2, padding='same', name='d_h1_conv')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Conv2D(filters=self.df_dim * 4, kernel_size=5, strides=2, padding='same', name='d_h2_conv')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Conv2D(filters=self.df_dim * 8, kernel_size=5, strides=2, padding='same', name='d_h3_conv')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Flatten()(x)
        x = Dense(1, activation='sigmoid', name='d_h3_lin')(x)

        return Model(image, x, name='D')

    def build_model(self):
        image_dims = [self.input_height, self.input_width, self.c_dim]
        optimizer = RMSprop(lr=0.002, clipvalue=1.0, decay=1e-8)
        # Construct discriminator/D network takes real image as input.
        # D - sigmoid and D_logits -linear output.
        self.discriminator = self.build_discriminator(image_dims)

        # Model to train D to discrimate real images.
        self.discriminator.compile(optimizer=optimizer, loss='binary_crossentropy')

        # Construct generator/R network.
        self.generator = self.build_generator(image_dims)
        img = Input(shape=image_dims)

        reconstructed_img = self.generator(img)

        self.discriminator.trainable = False
        validity = self.discriminator(reconstructed_img)

        # Model to train Generator/R to minimize reconstruction loss and trick D to see
        # generated images as real ones.
        self.adversarial_model = Model(img, [reconstructed_img, validity])
        self.adversarial_model.compile(loss=['binary_crossentropy', 'binary_crossentropy'],
                                       loss_weights=[self.r_alpha, 1],
                                       optimizer=optimizer)

        print('\n\rdiscriminator')
        self.discriminator.summary()

        print('\n\adversarial_model')
        self.adversarial_model.summary()

    def train(self, epochs, batch_size=128, sample_interval=500):
        # Get a batch of sample images with attention_label to export as montage.
        sample = self.data[0:batch_size]

        # Export images as montage, sample_input also use later to generate sample R network outputs during training.
        sample_inputs = np.array(sample).astype(np.float32)
        # scipy.misc.imsave('./{}/train_input_samples.jpg'.format(self.sample_dir), montage(sample_inputs[:, :, :, 0]))

        counter = 1
        # Record generator/R network reconstruction training losses.
        plot_epochs = []
        plot_g_recon_losses = []

        # Load traning data, add random noise.
        sample_w_noise = get_noisy_data(self.data)

        # Adversarial ground truths
        ones = np.ones((batch_size, 1))
        zeros = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            print('Epoch ({}/{})-------------------------------------------------'.format(epoch, epochs))
            # Number of batches computed by total number of target data / batch size.
            batch_idxs = len(self.data) // batch_size

            for idx in range(0, batch_idxs):
                # Get a batch of images and add random noise.
                batch = self.data[idx * batch_size:(idx + 1) * batch_size]
                batch_noise = sample_w_noise[idx * batch_size:(idx + 1) * batch_size]
                batch_clean = self.data[idx * batch_size:(idx + 1) * batch_size]
                # Turn batch images data to float32 type.
                batch_images = np.array(batch).astype(np.float32)
                batch_noise_images = np.array(batch_noise).astype(np.float32)
                batch_clean_images = np.array(batch_clean).astype(np.float32)
                batch_fake_images = self.generator.predict(batch_noise_images)
                # Update D network, minimize real images inputs->D-> ones, noisy z->R->D->zeros loss.
                d_loss_real = self.discriminator.train_on_batch(batch_images, ones)
                d_loss_fake = self.discriminator.train_on_batch(batch_fake_images, zeros)

                # Update R network twice, minimize noisy z->R->D->ones and reconstruction loss.
                self.adversarial_model.train_on_batch(batch_noise_images, [batch_clean_images, ones])
                g_loss = self.adversarial_model.train_on_batch(batch_noise_images, [batch_clean_images, ones])
                plot_epochs.append(epoch + idx / batch_idxs)
                plot_g_recon_losses.append(g_loss[1])
                counter += 1
                msg = 'Epoch:[{0}]-[{1}/{2}] --> d_loss: {3:>0.3f}, g_loss:{4:>0.3f}, g_recon_loss:{4:>0.3f}'.format(
                    epoch, idx, batch_idxs, d_loss_real + d_loss_fake, g_loss[0], g_loss[1])
                print(msg)
                logging.info(msg)
                if np.mod(counter, sample_interval) == 0:
                    samples = self.generator.predict(sample_inputs)
                    manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
                    manifold_w = int(np.floor(np.sqrt(samples.shape[0])))
                    # save_images(samples, [manifold_h, manifold_w],
                    #             './{}/train_{:02d}_{:04d}.png'.format(self.sample_dir, epoch, idx))

            # Save the checkpoint end of each epoch.
            self.save(epoch)
        # Export the Generator/R network reconstruction losses as a plot.
        plt.title('Generator/R network reconstruction losses')
        plt.xlabel('Epoch')
        plt.ylabel('training loss')
        plt.grid()
        plt.plot(plot_epochs, plot_g_recon_losses)
        plt.savefig('plot_g_recon_losses.png')

    def save(self, step):
        """Helper method to save model weights.
        
        Arguments:
            step {[type]} -- [description]
        """
        model_name = 'ALOCC_Model_{}.h5'.format(step)
        self.adversarial_model.save_weights(self.checkpoint_path.joinpath(model_name))

    def load_weight(self, weight_path):
        self.adversarial_model.load_weights(weight_path)

    def predict(self, images, weight_path):
        self.load_weight(weight_path)
        assert (self.input_height, self.input_width, self.c_dim) == (images.shape[1:])
        return self.adversarial_model.predict(images)
