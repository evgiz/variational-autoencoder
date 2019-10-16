"""
Author: Sigve Rokenes
Date: February, 2019

Variational autoencoder

"""

import sys
from util import *
import numpy as np
import tensorflow as tf
from batch_manager import BatchManager


# ===================================== #
#                                       #
#      Variational Autoencoder          #
#                                       #
# ===================================== #

class VariationalAutoencoder:

    def __init__(self, sess, batch_size=32, image_size=128, channels=3, latent_size=256):
        self.sess = sess
        self.batch_size = batch_size
        self.latent_size = latent_size
        self.image_size = image_size

        self.image_shape = [None, self.image_size, self.image_size, channels]
        self.encoder_input = tf.placeholder(tf.float32, shape=self.image_shape)
        self.decoder_target = tf.placeholder(tf.float32, shape=self.image_shape)
        self.latent_input = tf.placeholder(tf.float32, shape=[None, self.latent_size])

        self.encoder, mean, stddev = self.make_encoder()
        self.decoder = self.make_decoder()
        self.generator = self.make_decoder(generator=True)

        flat_img = tf.reshape(self.decoder, shape=[-1, self.image_shape[1] * self.image_shape[2] * self.image_shape[3]])
        flat_lbl = tf.reshape(self.decoder_target,
                              shape=[-1, self.image_shape[1] * self.image_shape[2] * self.image_shape[3]])
        rec_loss = tf.reduce_sum(tf.squared_difference(flat_img, flat_lbl), 1)
        lat_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * stddev - tf.square(mean) - tf.exp(2.0 * stddev), 1)

        self.loss = tf.reduce_mean(rec_loss + lat_loss)
        adam = tf.train.AdamOptimizer(0.0005)
        self.optimize = adam.minimize(self.loss)

        print("# ========================= #")
        print("#  Variational Autoencoder  #")
        print("# ========================= #")
        print("Encoder:  ", self.encoder.get_shape())
        print("Latent:   ", self.latent_size)
        print("Decoder:  ", self.decoder.get_shape())
        print("Generator:", self.generator.get_shape())

    def make_encoder(self):
        with tf.variable_scope("encoder"):
            print("# ========================= #")
            print("#       Encoding Model      #")
            print("# ========================= #")
            net = self.encoder_input
            print_tensor(net, "input")

            net = conv(net, 32, kernel=3, strides=1)
            net = tf.layers.dropout(net, 0.15)
            net = tf.layers.max_pooling2d(net, 2, 2)
            print_tensor(net, "conv")

            net = conv(net, 64, kernel=5, strides=1)
            net = tf.layers.dropout(net, 0.15)
            net = tf.layers.max_pooling2d(net, 2, 2)
            print_tensor(net, "conv")

            net = conv(net, 128, kernel=5, strides=3)
            net = tf.layers.dropout(net, 0.15)
            net = tf.layers.max_pooling2d(net, 2, 2)
            print_tensor(net, "conv")

            net = tf.layers.flatten(net)
            print_tensor(net, "flat")

            mean = tf.layers.dense(net, units=self.latent_size)
            stddev = tf.layers.dense(net, units=self.latent_size) / 2.0
            epsilon = tf.random_normal(tf.shape(stddev), 0, 1, dtype=tf.float32)
            latent = mean + tf.multiply(tf.exp(stddev), epsilon)
            print_tensor(latent, "latent")

        return latent, mean, stddev

    def make_decoder(self, generator=False):
        with tf.variable_scope("decoder", reuse=generator):
            print("# ========================= #")
            if generator:
                net = self.latent_input
                print("#      Generative Model     #")
            else:
                net = self.encoder
                print("#       Decoding Model      #")
            print("# ========================= #")
            print_tensor(net, "input")

            net = tf.layers.dense(net, 64, activation=tf.nn.leaky_relu)
            print_tensor(net, "dense")

            net = tf.layers.dense(net, 192, activation=tf.nn.leaky_relu)
            print_tensor(net, "dense")

            net = tf.reshape(net, [-1, 8, 8, 3])
            print_tensor(net, "deconv")

            net = deconv(net, 256, kernel=3, strides=1)
            net = tf.layers.dropout(net, 0.15)
            print_tensor(net, "deconv")

            net = deconv(net, 128, kernel=5, strides=2)
            net = tf.layers.dropout(net, 0.15)
            print_tensor(net, "deconv")

            net = deconv(net, 64, kernel=5, strides=4)
            net = tf.layers.dropout(net, 0.15)
            print_tensor(net, "deconv")

            net = deconv(net, self.image_shape[3], strides=1, activation=tf.nn.sigmoid)
            print_tensor(net, "deconv")

        return net

    def train(self, data):
        _, train_loss = self.sess.run([self.optimize, self.loss], feed_dict={
            self.decoder_target: np.array(data),
            self.encoder_input: np.array(data)
        })
        return train_loss


# ===================================== #
#                                       #
#            Training Script            #
#                                       #
# ===================================== #

if __name__ == "__main__":

    n_epochs = 100
    batch_size = 64

    with tf.Session() as sess:
        vae = VariationalAutoencoder(sess, batch_size, image_size=64, channels=3, latent_size=256)
        batch_manager = BatchManager("data/processed", resize=(64, 64))
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        print("# ========================= #")
        print("#                           #")
        print("#      Training Session     #")
        print("#                           #")
        print("# ========================= #")
        print("n_epochs          =", n_epochs)
        print("batch_size        =", batch_size)
        print("training_examples =", batch_manager.num_examples())
        print("batch_shape       =", np.shape(batch_manager.next_batch(batch_size)))
        print("params            =", np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
        print("   decoder        =", np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables() if "decoder" in v.name]))
        print("   encoder        =", np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables() if "encoder" in v.name]), "\n")

        for epoch in range(n_epochs):

            loss = 0
            epoch_progress = 0
            n_batches = int(batch_manager.num_examples() / batch_size)

            print("# ========================= #")
            print("#        Epoch {:05d}        #\t".format(epoch))
            print("# ========================= #")

            for b in range(n_batches):
                batch = batch_manager.next_batch(batch_size)
                loss += vae.train(batch)

                progress = int(float(b) / n_batches * 100)
                if progress >= epoch_progress + 10:
                    epoch_progress = progress
                    print("{}%".format(progress), end=" ")
                    sys.stdout.flush()

            print("Loss:", loss)

            # ===================================== #
            #              Output Images            #
            # ===================================== #

            sample_batch = batch_manager.sample(10)

            recreated = sess.run(vae.decoder, feed_dict={
                vae.encoder_input: sample_batch
            })
            random_gen = sess.run(vae.generator, feed_dict={
                vae.latent_input: np.random.normal(0, 1, [10, vae.latent_size])
            })
            encoded = sess.run(vae.encoder, feed_dict={
                vae.encoder_input: sample_batch
            })

            for sample in range(10):
                save_image("data/training/{:02d}_a.png".format(sample), sample_batch[sample], resize=[128, 128])
                save_image("data/training/{:02d}_b.png".format(sample), recreated[sample], resize=[128, 128])
                save_image("data/training/gen_{:02d}.png".format(sample), random_gen[sample], resize=[128, 128])

            # ===================================== #
            #               Save Model              #
            # ===================================== #

            saver.save(sess, "model/model_{}".format(epoch))
