import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
import tables
import diffraction_functions
import pickle
import sys


class GetData():
    def __init__(self, batch_size):
        self.batch_counter = 0
        self.batch_index = 0
        self.batch_size = batch_size
        self.train_filename = "zernike3/build/train.hdf5"
        self.test_filename = "zernike3/build/test.hdf5"
        self.hdf5_file_train = tables.open_file(self.train_filename, mode="r")
        self.hdf5_file_validation = tables.open_file(self.test_filename, mode="r")
        self.samples = self.hdf5_file_train.root.object_real.shape[0]
        # shape of the sample
        self.N = self.hdf5_file_train.root.N[0,0]
        print("initializing GetData")
        print("self.N =>", self.N)
        print("self.samples =>", self.samples)

    def next_batch(self):
        # retrieve the next batch of data from the data source
        samples = {}
        samples["object_real_samples"] = self.hdf5_file_train.root.object_real[self.batch_index:self.batch_index + self.batch_size, :]
        samples["object_imag_samples"] = self.hdf5_file_train.root.object_imag[self.batch_index:self.batch_index + self.batch_size, :]
        samples["diffraction_samples"] = self.hdf5_file_train.root.diffraction[self.batch_index:self.batch_index + self.batch_size, :]

        self.batch_index += self.batch_size

        return  samples

    def evaluate_on_train_data(self, n_samples):
        samples = {}
        samples["object_real_samples"] = self.hdf5_file_train.root.object_real[:n_samples, :]
        samples["object_imag_samples"] = self.hdf5_file_train.root.object_imag[:n_samples, :]
        samples["diffraction_samples"] = self.hdf5_file_train.root.diffraction[:n_samples, :]

        return samples

    def evaluate_on_validation_data(self, n_samples):
        samples = {}
        samples["object_real_samples"] = self.hdf5_file_validation.root.object_real[:n_samples, :]
        samples["object_imag_samples"] = self.hdf5_file_validation.root.object_imag[:n_samples, :]
        samples["diffraction_samples"] = self.hdf5_file_validation.root.diffraction[:n_samples, :]
        # samples["imag_scalar_samples"] = self.hdf5_file_train.root.imag_norm_factor[:n_samples, :]

        return samples

    def __del__(self):
        self.hdf5_file_train.close()

class DiffractionNet():
    def __init__(self, name):
        self.name = name
        print("initializing network")
        print(name)

        # initialize get data object
        self.get_data = GetData(batch_size=10)

        # input image
        self.x = tf.placeholder(tf.float32, shape=[None, self.get_data.N , self.get_data.N, 1])
        # self.x = tf.placeholder(tf.float32, shape=[None, 128 , 128, 1])
        # label
        self.imag_actual = tf.placeholder(tf.float32, shape=[None, self.get_data.N, self.get_data.N, 1])
        # self.imag_scalar_actual = tf.placeholder(tf.float32, shape=[None, 1])
        self.real_actual = tf.placeholder(tf.float32, shape=[None, self.get_data.N, self.get_data.N, 1])

        # amplitude mask for generating the output
        _, self.amplitude_mask = diffraction_functions.get_amplitude_mask_and_imagesize(self.get_data.N, int(self.get_data.N/2))
        self.amplitude_mask = np.expand_dims(self.amplitude_mask, axis=-1)
        self.amplitude_mask = np.expand_dims(self.amplitude_mask, axis=0)

        # real retrieval network
        self.nn_nodes = {}
        self.setup_network_2(self.nn_nodes)

        # learning rate
        self.s_LR = tf.placeholder(tf.float32, shape=[])

        # define loss function

        #####################
        # mean squared error
        #####################
        self.nn_nodes["real_loss"] = tf.losses.mean_squared_error(labels=self.real_actual, predictions=self.nn_nodes["real_out"])
        self.nn_nodes["imag_loss"] = tf.losses.mean_squared_error(labels=self.imag_actual, predictions=self.nn_nodes["imag_out"])
        # self.nn_nodes["imag_norm_factor_loss"] = tf.losses.mean_squared_error(labels=self.imag_scalar_actual, predictions=self.nn_nodes["imag_scalar_out"])

        #####################
        # cross entropy loss
        #####################
        # self.nn_nodes["real_loss"] = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.real_actual, logits=self.nn_nodes["real_logits"]))
        # self.nn_nodes["imag_loss"] = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.imag_actual, logits=self.nn_nodes["imag_logits"]))

        #####################
        # reconstruction loss
        #####################
        # self.nn_nodes["recons_loss"] = TODO
        self.nn_nodes["recons_diffraction_pattern"] = diffraction_functions.tf_reconstruct_diffraction_pattern(real_norm=self.nn_nodes["real_out"], imag_norm=self.nn_nodes["imag_out"])
        self.nn_nodes["reconstruction_loss"] = tf.losses.mean_squared_error(labels=self.x, predictions=self.nn_nodes["recons_diffraction_pattern"])

        self.nn_nodes["cost_function"] = self.nn_nodes["real_loss"] + self.nn_nodes["imag_loss"]
        # + self.nn_nodes["imag_norm_factor_loss"]

        optimizer = tf.train.AdamOptimizer(learning_rate=self.s_LR)
        self.nn_nodes["train"] = optimizer.minimize(self.nn_nodes["cost_function"])

        optimizer_u = tf.train.AdamOptimizer(learning_rate=self.s_LR)
        self.nn_nodes["u_train"] = optimizer_u.minimize(self.nn_nodes["reconstruction_loss"])

        # save file
        if not os.path.isdir('./models'):
            os.makedirs('./models')
        shutil.copyfile('./'+ os.path.basename(__file__), './models/'+os.path.basename(__file__).split(".")[0]+'_{}.py'.format(self.name))

        # setup logging
        self.tf_loggers = {}
        self.setup_logging()

        # initialize graph
        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(self.init)
        self.writer = tf.summary.FileWriter("./tensorboard_graph/" + self.name)

        # number of epochs to run
        self.epochs = 90000
        self.i = 0
        self.epoch = None
        self.dots = None

        # intialize saver
        self.saver = tf.train.Saver()

        # if the model already exists, load it into memory
        if os.path.exists('./models/{}.ckpt.index'.format(self.name)):
            self.saver.restore(self.sess, './models/{}.ckpt'.format(self.name))
            with open("models/" + self.name + ".p", "rb") as file:
                obj = pickle.load(file)
                self.i = obj["i"]
            print("restored saved model {}".format(self.name))
            print("model loaded at epoch {}".format(self.i))


    def setup_network_1(self, _nodes):
        # convolutional layer down sampling

        # _nodes["conv1"] = convolutional_layer(self.x, shape=[3,3,1,32], activate='relu', stride=[1,1])
        _nodes["conv1"] = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='SAME', activation='relu')(self.x)

        # _nodes["conv2"] = convolutional_layer(_nodes["conv1"], shape=[3,3,32,32], activate='relu', stride=[1,1])
        _nodes["conv2"] = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='SAME', activation='relu')(_nodes["conv1"])

        # max pooling
        # _nodes["pool3"] = max_pooling_layer(_nodes["conv2"], pool_size_val=[2,2], stride_val=[2,2], pad=True)
        _nodes["pool3"] = tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding='SAME')(_nodes["conv2"])

        # convolutional layer
        # _nodes["conv4"] = convolutional_layer(_nodes["pool3"], shape=[3,3,32,64], activate='relu', stride=[1,1])
        _nodes["conv4"] = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='SAME', activation='relu')(_nodes["pool3"])

        # _nodes["conv5"] = convolutional_layer(_nodes["conv4"], shape=[3,3,64,64], activate='relu', stride=[1,1])
        _nodes["conv5"] = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='SAME', activation='relu')(_nodes["conv4"])

        # max pooling
        # _nodes["pool6"] = max_pooling_layer(_nodes["conv5"], pool_size_val=[2,2], stride_val=[2,2], pad=True)
        _nodes["pool6"] = tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding='SAME')(_nodes["conv5"])

        # convolutional layer
        # _nodes["conv7"] = convolutional_layer(_nodes["pool6"], shape=[3,3,64,128], activate='relu', stride=[1,1])
        _nodes["conv7"] = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='SAME', activation='relu')(_nodes["pool6"])

        # _nodes["conv8"] = convolutional_layer(_nodes["conv7"], shape=[3,3,128,128], activate='relu', stride=[1,1])
        _nodes["conv8"] = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='SAME', activation='relu')(_nodes["conv7"])

        # max pooling
        # _nodes["pool9"] = max_pooling_layer(_nodes["conv8"], pool_size_val=[2,2], stride_val=[2,2], pad=True)
        _nodes["pool9"] = tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding='SAME')(_nodes["conv8"])

        # convolutional layer
        # _nodes["conv10"] = convolutional_layer(_nodes["pool9"], shape=[3,3,128,128], activate='relu', stride=[1,1])
        _nodes["conv10"] = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='SAME', activation='relu')(_nodes["pool9"])

        # _nodes["conv11"] = convolutional_layer(_nodes["conv10"], shape=[3,3,128,128], activate='relu', stride=[1,1])
        _nodes["conv11"] = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='SAME', activation='relu')(_nodes["conv10"])

        # up sampling
        # _nodes["ups12"] = upsample_2d(_nodes["conv11"], 2)
        _nodes["ups12"] = tf.keras.layers.UpSampling2D(size=2)(_nodes["conv11"])

        # convolutional layer
        # _nodes["conv13"] = convolutional_layer(_nodes["ups12"], shape=[3,3,128,64], activate='relu', stride=[1,1])
        _nodes["conv13"] = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='SAME', activation='relu')(_nodes["ups12"])

        # _nodes["conv14"] = convolutional_layer(_nodes["conv13"], shape=[3,3,64,64], activate='relu', stride=[1,1])
        _nodes["conv14"] = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='SAME', activation='relu')(_nodes["conv13"])

        # up sampling
        # _nodes["ups15"] = upsample_2d(_nodes["conv14"], 2)
        _nodes["ups15"] = tf.keras.layers.UpSampling2D(size=2)(_nodes["conv14"])

        # convolutional layer
        # _nodes["conv16"] = convolutional_layer(_nodes["ups15"], shape=[3,3,64,32], activate='relu', stride=[1,1])
        _nodes["conv16"] = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='SAME', activation='relu')(_nodes["ups15"])

        # _nodes["conv17"] = convolutional_layer(_nodes["conv16"], shape=[3,3,32,32], activate='relu', stride=[1,1])
        _nodes["conv17"] = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='SAME', activation='relu')(_nodes["conv16"])

        # up sampling
        # _nodes["ups18"] = upsample_2d(_nodes["conv17"], 2)
        _nodes["ups18"] = tf.keras.layers.UpSampling2D(size=2)(_nodes["conv17"])

        # _nodes["conv19"] = convolutional_layer(_nodes["ups18"], shape=[3,3,32,1], activate='sigmoid', stride=[1,1])
        _nodes["imag_logits"] = tf.keras.layers.Conv2D(filters=1, kernel_size=3, padding='SAME')(_nodes["ups18"])
        _nodes["real_logits"] = tf.keras.layers.Conv2D(filters=1, kernel_size=3, padding='SAME')(_nodes["ups18"])
        # _nodes["out_logits"] = _nodes["conv19"]
        _nodes["imag_out"] = tf.nn.sigmoid(_nodes["imag_logits"])
        _nodes["real_out"] = tf.nn.sigmoid(_nodes["real_logits"])

    def setup_network_2(self, _nodes):

        assert int(self.x.shape[2]) == 128
        assert int(self.x.shape[1]) == 128

        _nodes["conv1"] = tf.keras.layers.Conv2D(filters=64, kernel_size=4, padding='SAME', strides=2)(self.x)
        _nodes["leakyrelu2"] = tf.keras.layers.LeakyReLU(alpha=0.2)(_nodes["conv1"])

        _nodes["conv3"] = tf.keras.layers.Conv2D(filters=128, kernel_size=4, padding='SAME', strides=2)(_nodes['leakyrelu2'])
        _nodes["leakyrelu4"] = tf.keras.layers.LeakyReLU(alpha=0.2)(_nodes["conv3"])
        _nodes["batch_norm5"] = tf.keras.layers.BatchNormalization()(_nodes["leakyrelu4"])

        _nodes["conv6"] = tf.keras.layers.Conv2D(filters=256, kernel_size=4, padding='SAME', strides=2)(_nodes['batch_norm5'])
        _nodes["leakyrelu7"] = tf.keras.layers.LeakyReLU(alpha=0.2)(_nodes["conv6"])
        _nodes["batch_norm8"] = tf.keras.layers.BatchNormalization()(_nodes["leakyrelu7"])

        _nodes["conv9"] = tf.keras.layers.Conv2D(filters=512, kernel_size=4, padding='SAME', strides=2)(_nodes['batch_norm8'])
        _nodes["leakyrelu10"] = tf.keras.layers.LeakyReLU(alpha=0.2)(_nodes["conv9"])
        _nodes["batch_norm11"] = tf.keras.layers.BatchNormalization()(_nodes["leakyrelu10"])
        _nodes["sigmoid12"] = tf.keras.activations.sigmoid(_nodes["batch_norm11"])

        # feature encoded layer
        _nodes["conv_t13"] = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=4, padding='SAME', strides=2, activation='relu')(_nodes["sigmoid12"])
        _nodes["batch_norm14"] = tf.keras.layers.BatchNormalization()(_nodes["conv_t13"])

        _nodes["conv_t15"] = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=4, padding='SAME', strides=2, activation='relu')(_nodes["batch_norm14"])
        _nodes["batch_norm16"] = tf.keras.layers.BatchNormalization()(_nodes["conv_t15"])

        _nodes["conv_t17"] = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=4, padding='SAME', strides=2, activation='relu')(_nodes["batch_norm16"])
        _nodes["batch_norm18"] = tf.keras.layers.BatchNormalization()(_nodes["conv_t17"])

        _nodes["real_out"] = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=4, padding='SAME', strides=2, activation='relu')(_nodes["batch_norm18"])
        _nodes["imag_out"] = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=4, padding='SAME', strides=2, activation='relu')(_nodes["batch_norm18"])
        # output is currently between 0 and 1

        _nodes["real_out"] *=2
        _nodes["imag_out"] *=2
        _nodes["real_out"] -=1
        _nodes["imag_out"] -=1
        # the output is now between -1 and 1

        # constrain the output to the mask
        _nodes["real_out"] = _nodes["real_out"] * self.amplitude_mask
        _nodes["imag_out"] = _nodes["imag_out"] * self.amplitude_mask

    def setup_logging(self):
        self.tf_loggers["real_loss_training"] = tf.summary.scalar("real_loss_training", self.nn_nodes["real_loss"])
        self.tf_loggers["real_loss_validation"] = tf.summary.scalar("real_loss_validation", self.nn_nodes["real_loss"])
        self.tf_loggers["imag_loss_training"] = tf.summary.scalar("imag_loss_training", self.nn_nodes["imag_loss"])
        self.tf_loggers["imag_loss_validation"] = tf.summary.scalar("imag_loss_validation", self.nn_nodes["imag_loss"])
        # self.tf_loggers["imag_norm_factor_loss_training"] = tf.summary.scalar("imag_norm_factor_loss_training", self.nn_nodes["imag_norm_factor_loss"])
        # self.tf_loggers["imag_norm_factor_loss_validation"] = tf.summary.scalar("imag_norm_factor_loss_validation", self.nn_nodes["imag_norm_factor_loss"])
        self.tf_loggers["reconstruction_loss_training"] = tf.summary.scalar("reconstruction_loss_training", self.nn_nodes["reconstruction_loss"])
        self.tf_loggers["reconstruction_loss_validation"] = tf.summary.scalar("reconstruction_loss_validation", self.nn_nodes["reconstruction_loss"])

    def supervised_learn(self):
        while self.i < self.epochs:
        # for self.i in range(self.epochs):
            self.epoch = self.i + 1
            print("Epoch : {}".format(self.epoch))
            self.dots = 0
            while self.get_data.batch_index < self.get_data.samples:
                self.show_loading_bar()

                # retrieve data
                data = self.get_data.next_batch()

                # run training iteration
                object_real_samples = data["object_real_samples"].reshape(-1,self.get_data.N, self.get_data.N, 1)
                object_imag_samples = data["object_imag_samples"].reshape(-1,self.get_data.N, self.get_data.N, 1)
                # imag_scalar_samples = data["imag_scalar_samples"].reshape(-1, 1)
                diffraction_samples = data["diffraction_samples"].reshape(-1,self.get_data.N, self.get_data.N, 1)

                # train network
                self.sess.run(self.nn_nodes["train"], feed_dict={self.x:diffraction_samples,
                                                    self.real_actual:object_real_samples,
                                                    self.imag_actual:object_imag_samples,
                                                    # self.imag_scalar_actual:imag_scalar_samples,
                                                    self.s_LR:0.0001})

            print("add_tensorboard_values")
            self.add_tensorboard_values()
            if self.i % 50 == 0:

                # create directory if it doesnt exist
                check_is_dir("nn_pictures")
                check_is_dir("nn_pictures/"+self.name+"_pictures")
                check_is_dir("nn_pictures/"+self.name+"_pictures/"+str(self.epoch))
                check_is_dir("nn_pictures/"+self.name+"_pictures/"+str(self.epoch)+"/training")
                check_is_dir("nn_pictures/"+self.name+"_pictures/"+str(self.epoch)+"/validation")

                data = self.get_data.evaluate_on_train_data(n_samples=10)
                self.evaluate_performance(data, "training")

                data = self.get_data.evaluate_on_validation_data(n_samples=10)
                self.evaluate_performance(data, "validation")


            # save the network
            self.get_data.batch_index = 0
            self.i+=1
            print("saving network models/" + self.name + ".ckpt")
            self.saver.save(self.sess, "models/" + self.name + ".ckpt")
            training_parameters = {}
            training_parameters["i"] = self.i
            with open("models/" + self.name + ".p", "wb") as file:
                pickle.dump(training_parameters, file)
            print("network saved" + self.name + ".ckpt")


    def add_tensorboard_values(self):
        # # # # # # # # # # # # # # #
        # loss on the training data #
        # # # # # # # # # # # # # # #
        data = self.get_data.evaluate_on_train_data(n_samples=50)
        object_real_samples = data["object_real_samples"].reshape(-1,self.get_data.N, self.get_data.N, 1)
        object_imag_samples = data["object_imag_samples"].reshape(-1,self.get_data.N, self.get_data.N, 1)
        diffraction_samples = data["diffraction_samples"].reshape(-1,self.get_data.N, self.get_data.N, 1)
        # imag_scalar_samples = data["imag_scalar_samples"].reshape(-1, 1)

        # real loss
        loss_value = self.sess.run(self.nn_nodes["real_loss"], feed_dict={self.x:diffraction_samples, self.real_actual:object_real_samples})
        print("real training loss_value =>", loss_value)

        # imag loss
        loss_value = self.sess.run(self.nn_nodes["imag_loss"], feed_dict={self.x:diffraction_samples, self.imag_actual:object_imag_samples})
        print("imag training loss_value =>", loss_value)

        # write to log
        # real
        summ = self.sess.run(self.tf_loggers["real_loss_training"], feed_dict={self.x:diffraction_samples, self.real_actual:object_real_samples})
        self.writer.add_summary(summ, global_step=self.epoch)

        # imag
        summ = self.sess.run(self.tf_loggers["imag_loss_training"], feed_dict={self.x:diffraction_samples, self.imag_actual:object_imag_samples})
        self.writer.add_summary(summ, global_step=self.epoch)

        # imag scalar
        # summ = self.sess.run(self.tf_loggers["imag_norm_factor_loss_training"], feed_dict={self.x:diffraction_samples, self.imag_scalar_actual:imag_scalar_samples})
        self.writer.add_summary(summ, global_step=self.epoch)

        # reconstruction
        summ = self.sess.run(self.tf_loggers["reconstruction_loss_training"], feed_dict={self.x:diffraction_samples, self.imag_actual:object_imag_samples})
        self.writer.add_summary(summ, global_step=self.epoch)


        # # # # # # # # # # # # # # # #
        # loss on the validation data #
        # # # # # # # # # # # # # # # #
        data = self.get_data.evaluate_on_validation_data(n_samples=50)
        object_real_samples = data["object_real_samples"].reshape(-1,self.get_data.N, self.get_data.N, 1)
        object_imag_samples = data["object_imag_samples"].reshape(-1,self.get_data.N, self.get_data.N, 1)
        diffraction_samples = data["diffraction_samples"].reshape(-1,self.get_data.N, self.get_data.N, 1)
        # imag_scalar_samples = data["imag_scalar_samples"].reshape(-1, 1)

        # real loss
        loss_value = self.sess.run(self.nn_nodes["real_loss"], feed_dict={self.x:diffraction_samples, self.real_actual:object_real_samples})
        print("real validation loss_value =>", loss_value)

        # imag loss
        loss_value = self.sess.run(self.nn_nodes["imag_loss"], feed_dict={self.x:diffraction_samples, self.imag_actual:object_imag_samples})
        print("imag validation loss_value =>", loss_value)

        # write to log
        summ = self.sess.run(self.tf_loggers["real_loss_validation"], feed_dict={self.x:diffraction_samples, self.real_actual:object_real_samples})
        self.writer.add_summary(summ, global_step=self.epoch)

        summ = self.sess.run(self.tf_loggers["imag_loss_validation"], feed_dict={self.x:diffraction_samples, self.imag_actual:object_imag_samples})
        self.writer.add_summary(summ, global_step=self.epoch)

        # summ = self.sess.run(self.tf_loggers["imag_norm_factor_loss_validation"], feed_dict={self.x:diffraction_samples, self.imag_scalar_actual:imag_scalar_samples})
        self.writer.add_summary(summ, global_step=self.epoch)

        # reconstruction
        summ = self.sess.run(self.tf_loggers["reconstruction_loss_validation"], feed_dict={self.x:diffraction_samples, self.imag_actual:object_imag_samples})
        self.writer.add_summary(summ, global_step=self.epoch)

        self.writer.flush()

    def __del__(self):
        del self.get_data

    def show_loading_bar(self):
        # display loading bar
        percent = 50 * self.get_data.batch_index / self.get_data.samples
        if percent - self.dots > 1:
            print(".", end="", flush=True)
            self.dots += 1

    def evaluate_performance(self, _data, _set):
        """
            _data: the data set to input to the network
            _set: (validation or training)
            make plots of the output of the network
        """
        object_real_samples = _data["object_real_samples"].reshape(-1,self.get_data.N, self.get_data.N, 1)
        object_imag_samples = _data["object_imag_samples"].reshape(-1,self.get_data.N, self.get_data.N, 1)
        diffraction_samples = _data["diffraction_samples"].reshape(-1,self.get_data.N, self.get_data.N, 1)
        # imag_scalar_samples = _data["imag_scalar_samples"].reshape(-1, 1)

        # plot the output
        real_output = self.sess.run(self.nn_nodes["real_out"], feed_dict={self.x:diffraction_samples})
        imag_output = self.sess.run(self.nn_nodes["imag_out"], feed_dict={self.x:diffraction_samples})
        # imag_scalar_output = self.sess.run(self.nn_nodes["imag_scalar_out"], feed_dict={self.x:diffraction_samples})
        tf_reconstructed_diff = self.sess.run(self.nn_nodes["recons_diffraction_pattern"], feed_dict={self.x:diffraction_samples})

        # test
        #TODO delete this
        # imag_out_raw = self.sess.run(self.nn_nodes["imag_out_raw"], feed_dict={self.x:diffraction_samples})


        # # check the output
        # with open("nn_pictures/"+self.name+"_pictures/"+str(self.epoch)+"/"+_set+"/samples.p", "wb") as file:
            # obj = {}
            # # network output
            # obj["real_output"] = real_output
            # obj["imag_output"] = imag_output
            # # obj["imag_out_raw"] = imag_out_raw
            # # obj["imag_scalar_output"] = imag_scalar_output
            # obj["tf_reconstructed_diff"] = tf_reconstructed_diff

            # # training data
            # obj["object_real_samples"] = object_real_samples
            # obj["object_imag_samples"] = object_imag_samples
            # obj["diffraction_samples"] = diffraction_samples
            # # obj["imag_scalar_samples"] = imag_scalar_samples
            # pickle.dump(obj, file)


        # multiply by real_output

        # imag_output = imag_output * real_output #TODO maybe remove this

        for index in range(0,10):
            axes_obj = PlotAxes("sample "+str(index))

            im = axes_obj.diffraction_samples.pcolormesh(diffraction_samples[index,:,:,0])
            axes_obj.fig.colorbar(im, ax=axes_obj.diffraction_samples)

            im = axes_obj.object_real_samples.pcolormesh(object_real_samples[index,:,:,0])
            axes_obj.fig.colorbar(im, ax=axes_obj.object_real_samples)

            im = axes_obj.real_output.pcolormesh(real_output[index,:,:,0])
            axes_obj.fig.colorbar(im, ax=axes_obj.real_output)

            im = axes_obj.object_imag_samples.pcolormesh(object_imag_samples[index,:,:,0])
            axes_obj.fig.colorbar(im, ax=axes_obj.object_imag_samples)

            im = axes_obj.imag_output.pcolormesh(imag_output[index,:,:,0])
            axes_obj.fig.colorbar(im, ax=axes_obj.imag_output)

            # tensorflow reconstructed diffraction pattern
            im = axes_obj.tf_reconstructed_diff.pcolormesh(tf_reconstructed_diff[index,:,:,0])
            axes_obj.fig.colorbar(im, ax=axes_obj.tf_reconstructed_diff)

            axes_obj.save("nn_pictures/"+self.name+"_pictures/"+str(self.epoch)+"/"+_set+"/sample_"+str(index))
            del axes_obj

def max_pooling_layer(input_x, pool_size_val,  stride_val, pad=False):
    if pad:
        return tf.layers.max_pooling2d(input_x, pool_size=[pool_size_val[0], pool_size_val[1]], strides=[stride_val[0], stride_val[1]], padding="SAME")
    else:
        return tf.layers.max_pooling2d(input_x, pool_size=[pool_size_val[0], pool_size_val[1]], strides=[stride_val[0], stride_val[1]], padding="VALID")

def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
    return tf.Variable(init_random_dist)

def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(init_bias_vals)

def convolutional_layer(input_x, shape, activate, stride):
    W = init_weights(shape)
    b = init_bias([shape[3]])

    if activate == 'relu':
        return tf.nn.relu(conv2d(input_x, W, stride) + b)

    if activate == 'leaky':
        return tf.nn.leaky_relu(conv2d(input_x, W, stride) + b)

    if activate == 'sigmoid':
        return tf.nn.sigmoid(conv2d(input_x, W, stride) + b)

    elif activate == 'none':
        return conv2d(input_x, W, stride) + b

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1], padding='SAME')

def upsample_2d(x, S):
    height = int(x.shape[1])
    width = int(x.shape[2])

    return tf.image.resize_nearest_neighbor(x, (S*height, S*width))

def check_is_dir(directory):
    if not os.path.isdir(directory):
        os.mkdir(directory)

class PlotAxes():
    def __init__(self,fig_title):
        """
        fig_title:(string)
        title for the figure
        """
        # create plot
        self.fig = plt.figure(figsize=(10,10))
        self.gs = self.fig.add_gridspec(3,2)

        self.fig.text(0.5, 0.95,fig_title, fontsize=30, ha='center')

        self.axes = {}
        self.diffraction_samples = self.fig.add_subplot(self.gs[0:1,0:1])
        self.diffraction_samples.set_title("Input Diffraction Pattern")
        self.diffraction_samples.set_xticks([])
        self.diffraction_samples.set_yticks([])

        # self.diffraction_recons = self.fig.add_subplot(self.gs[0:1,1:2])
        # self.diffraction_recons.set_title("diffraction_recons")
        # self.diffraction_recons.set_xticks([])
        # self.diffraction_recons.set_yticks([])

        self.tf_reconstructed_diff = self.fig.add_subplot(self.gs[0:1,1:2])
        self.tf_reconstructed_diff.set_title("Reconstructed Diffration Pattern")
        self.tf_reconstructed_diff.set_xticks([])
        self.tf_reconstructed_diff.set_yticks([])

        self.object_real_samples = self.fig.add_subplot(self.gs[1:2,0:1])
        self.object_real_samples.set_title("Actual Real Object")
        self.object_real_samples.set_xticks([])
        self.object_real_samples.set_yticks([])

        self.real_output = self.fig.add_subplot(self.gs[1:2,1:2])
        self.real_output.set_title("Retrieved Real Object")
        self.real_output.set_xticks([])
        self.real_output.set_yticks([])

        self.imag_output = self.fig.add_subplot(self.gs[2:3,1:2])
        self.imag_output.set_title("Retrieved Imag Object")
        self.imag_output.set_xticks([])
        self.imag_output.set_yticks([])

        self.object_imag_samples = self.fig.add_subplot(self.gs[2:3,0:1])
        self.object_imag_samples.set_title("Actual Imag Object")
        self.object_imag_samples.set_xticks([])
        self.object_imag_samples.set_yticks([])


    def save(self, filename):
        self.fig.savefig(filename)


if __name__ == "__main__":
    # getdata = GetData(batch_size=10)
    # getdata.next_batch()
    # del getdata

    diffraction_net = DiffractionNet(name=sys.argv[1])
    diffraction_net.supervised_learn()
    del diffraction_net
    # pass




