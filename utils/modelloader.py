"""
This material was prepared as an account of work sponsored by an agency of the
United States Government.  Neither the United States Government nor the United
States Department of Energy, nor Battelle, nor any of their employees, nor any
jurisdiction or organization that has cooperated in the development of these
materials, makes any warranty, express or implied, or assumes any legal
liability or responsibility for the accuracy, completeness, or usefulness or
any information, apparatus, product, software, or process disclosed, or
represents that its use would not infringe privately owned rights.

Reference herein to any specific commercial product, process, or service by
trade name, trademark, manufacturer, or otherwise does not necessarily
constitute or imply its endorsement, recommendation, or favoring by the United
States Government or any agency thereof, or Battelle Memorial Institute. The
views and opinions of authors expressed herein do not necessarily state or
reflect those of the United States Government or any agency thereof.

                 PACIFIC NORTHWEST NATIONAL LABORATORY
                              operated by
                                BATTELLE
                                for the
                   UNITED STATES DEPARTMENT OF ENERGY
                    under Contract DE-AC05-76RL01830
"""

import logging
import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, initializers, regularizers
from tensorflow.keras.models import Model, Sequential, load_model
import math



def get_model(args):
    """Loads few-shot model and training callbacks.

    Args:
      args: Command-line arguments.

    Returns:
      Loaded model and callbacks.
    """      
        
    # get number of GPUs
    devices = tf.config.list_physical_devices('GPU')
    logging.info(f"num GPUs used for training: {len(devices)}")
    logging.info(f"...using devices {devices}")


    # load model
    if args.parallel:
        # setup strategy for parallel model
        mirrored_strategy = tf.distribute.MirroredStrategy()
        args.batch_size = args.batch_size * len(devices)
        learning_rate = args.learning_rate_start * len(devices)
        # set up optimizer
        opt = Adam(learning_rate)
        with mirrored_strategy.scope():
            if args.basemodel == 'None':
                model = build_fn(args)
            else:
                # note that model must have same hyperparameters
                model = load_FewShotNetwork(args.basemodel)
            model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    else:
        # set up optimizer
        opt = Adam(args.learning_rate_start)
        if args.basemodel == 'None':
            model = build_fn(args)
        else:
            # note that model must have same hyperparameters
            model = load_FewShotNetwork(args.basemodel)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        
    # set training callbacks: logging, checkpoint, early stopping, adaptive learning rate
    callbacks = [tf.keras.callbacks.CSVLogger(os.path.join(args.savedir, args.model_name, 'log.csv')),
                 tf.keras.callbacks.ModelCheckpoint(os.path.join(args.savedir, args.model_name, 'checkpoint.h5')),
                 tf.keras.callbacks.ModelCheckpoint(os.path.join(args.savedir, args.model_name, 'best_model.h5'), 
                                                    save_best_only=True),
                 tf.keras.callbacks.TerminateOnNaN(),
                 tf.keras.callbacks.ReduceLROnPlateau(min_delta=0.0001, min_lr=args.learning_rate_end),
                 tf.keras.callbacks.EarlyStopping(restore_best_weights=True, patience=10)
                ]

    return model, callbacks



class AlexNet1D(layers.Layer):
    def __init__(self, cutoff, prepool, prepool_size, 
                 pool_size, activation='relu', l2_reg=0., 
                 in_shape=None, **kwargs):
        """Feature extraction model for EDS spectra.
        
        Args:
            cutoff (int): Length of spectra.
            prepool (bool): Choose whether to prepool data.
            prepool_size (int): Size of the max pooling window prior to AlexNet1D.
            pool_size (int): Size of the max pooling window.
            activation (str): Activation function for Conv1D (default = relu)
            l2_reg (float): L2 regularization factor (default = 0.)
            
        """
        super(AlexNet1D, self).__init__(**kwargs)
        self.cutoff = cutoff
        self.prepool_size = prepool_size
        self.pool_size = pool_size
        self.activation = activation
        self.l2_reg = l2_reg
        self.prepool = prepool
               
        if self.prepool:
            # pool raw data
            self.mp0 = layers.MaxPool1D(pool_size=self.prepool_size, padding='valid')
            in_shape = (int((self.cutoff-1)/self.prepool_size), 1)
        else:
            in_shape = (self.cutoff, 1)

        # Layer 1
        self.c1  = layers.Conv1D(96, 11, input_shape=in_shape,
                   padding='same', kernel_regularizer=regularizers.l2(self.l2_reg))
        self.bn1 = layers.BatchNormalization()
        self.a1  = layers.Activation(self.activation)
        self.mp1 = layers.MaxPool1D(pool_size=self.pool_size)

        # Layer 2
        self.c2  = layers.Conv1D(256, 5, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.a2  = layers.Activation(self.activation)
        self.mp2 = layers.MaxPool1D(pool_size=self.pool_size)

        # Layer 3
        self.zp3 = layers.ZeroPadding1D(1)
        self.c3  = layers.Conv1D(512, 3, padding='same')
        self.bn3 = layers.BatchNormalization()
        self.a3  = layers.Activation(self.activation)
        self.mp3 = layers.MaxPool1D(pool_size=self.pool_size)

        # Layer 4
        self.zp4 = layers.ZeroPadding1D(1)
        self.c4  = layers.Conv1D(1024, 3, padding='same')
        self.bn4 = layers.BatchNormalization()
        self.a4  = layers.Activation(self.activation)

        # Layer 5
        self.zp5 = layers.ZeroPadding1D(1)
        self.c5  = layers.Conv1D(1024, 3, padding='same')
        self.bn5 = layers.BatchNormalization()
        self.a5  = layers.Activation(self.activation)
        self.mp5 = layers.MaxPool1D(pool_size=self.pool_size)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            #'in_shape': self.in_shape,
            'prepool_size': self.prepool_size,
            'pool_size': self.pool_size,
            'activation': self.activation,
            'cutoff': self.cutoff,
            'prepool': self.prepool,
            'l2_reg': self.l2_reg
        })
        return config
        
    def call(self, inputs):
        if self.prepool:
            inputs = self.mp0(inputs)
        x = self.c1(inputs)
        x = self.bn1(x)
        x = self.a1(x)
        x = self.mp1(x)
        x = self.c2(x)
        x = self.bn2(x)
        x = self.a2(x)
        x = self.mp2(x)
        x = self.zp3(x)
        x = self.c3(x)
        x = self.bn3(x)
        x = self.a3(x)
        x = self.mp3(x)
        x = self.zp4(x)
        x = self.c4(x)
        x = self.bn4(x)
        x = self.a4(x)
        x = self.zp5(x)
        x = self.c5(x)
        x = self.bn5(x)
        x = self.a5(x)
        x = self.mp5(x)
        return x

class AlexNet2D(layers.Layer):
    def __init__(
        self,
        prepool,
        prepool_size,
        pool_size=(2, 2),
        activation="relu",
        l2_reg=0.0,
        in_shape=(128, 128, 1),
        **kwargs
    ):
        """ Feature extraction model for SEM images.
        
        Args:
            prepool (bool): Choose whether to prepool data.
            prepool_size (int): Size of the max pooling window prior to AlexNet1D.
            pool_size (int): Size of the max pooling window.
            activation (str): Activation function for Conv2D (default = relu)
            l2_reg (float): L2 regularization factor (default = 0.)
            in_shape (tuple of int): Image shape (default = (128, 128, 3))
            
        """
        super(AlexNet2D, self).__init__(**kwargs)
        self.prepool_size = prepool_size 
        self.pool_size = pool_size
        self.activation = activation
        self.l2_reg = l2_reg
        self.prepool = False #prepool 

        if self.prepool:
            # pool raw data
            self.mp0 = layers.MaxPool2D(pool_size=(self.prepool_size, self.prepool_size), padding='valid')
            self.in_shape = (int(in_shape[0]/self.prepool_size), int(in_shape[1]/self.prepool_size), in_shape[2])
        else:
            self.in_shape = in_shape  

        # Layer 1
        self.c1 = layers.Conv2D(
            96,
            (11, 11),
            strides=4,
            padding="valid",
            activation=tf.keras.activations.relu,
            input_shape=self.in_shape,
            kernel_regularizer=regularizers.l2(self.l2_reg),
        )
        self.bn1 = layers.BatchNormalization()
        self.a1 = layers.Activation(self.activation)
        self.mp1 = layers.MaxPool2D(pool_size=(3,3), strides=2, padding="valid")

        # Layer 2
        self.c2 = layers.Conv2D(256, (5, 5), padding="same", strides=1,
        activation=tf.keras.activations.relu)
        self.bn2 = layers.BatchNormalization()
        self.a2 = layers.Activation(self.activation)
        self.mp2 = layers.MaxPool2D(pool_size=(3,3), strides=2, padding="same")

        # Layer 3
        self.zp3 = layers.ZeroPadding2D(1)
        self.c3 = layers.Conv2D(384, (3, 3), padding="same",
        activation=tf.keras.activations.relu, strides=1)
        self.bn3 = layers.BatchNormalization()
        self.a3 = layers.Activation(self.activation)
        self.mp3 = layers.MaxPool2D(pool_size=self.pool_size)

        # Layer 4
        self.zp4 = layers.ZeroPadding2D(1)
        self.c4 = layers.Conv2D(384, (3, 3), strides=1, padding="same", activation=tf.keras.activations.relu)
        self.bn4 = layers.BatchNormalization()
        self.a4 = layers.Activation(self.activation)

        # Layer 5
        #self.zp5 = layers.ZeroPadding2D((1, 1))
        self.zp5 = layers.ZeroPadding2D(1) 
        self.c5 = layers.Conv2D(256, (3, 3), strides=1, padding="same", activation=tf.keras.activations.relu)
        self.bn5 = layers.BatchNormalization()
        self.a5 = layers.Activation(self.activation)
        self.mp5 = layers.MaxPool2D(pool_size=self.pool_size, strides=2, padding="same")

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                #'in_shape': self.in_shape,
                "prepool_size": self.prepool_size,
                "pool_size": self.pool_size,
                "activation": self.activation,
                "prepool": self.prepool,
                "l2_reg": self.l2_reg,
            }
        )
        return config

    def call(self, inputs):
        if self.prepool:
            inputs = self.mp0(inputs)
        x = self.c1(inputs)
        x = self.mp1(x)
        x = self.bn1(x)
        x = self.a1(x)
        x = self.mp1(x)
        x = self.c2(x)
        x = self.mp2(x)
        x = self.bn2(x)
        x = self.a2(x)
        x = self.mp2(x)
        x = self.zp3(x)
        x = self.c3(x)
        x = self.mp3(x)
        x = self.bn3(x)
        x = self.a3(x)
        x = self.mp3(x)
        x = self.zp4(x)
        x = self.c4(x)
        x = self.bn4(x)
        x = self.a4(x)
        x = self.zp5(x)
        x = self.c5(x)
        x = self.mp5(x)
        x = self.bn5(x)
        x = self.a5(x)
        x = self.mp5(x)
        return x


class UQNet(layers.Layer):
    def __init__(self, batch_size, dropout, activation="relu", **kwargs):
        """Uncertainty quantification model.
        
        Args:
            batch_size (int): Batch size.
            dropout (float): Fraction of model nodes to drop.
            activation (str): Activation function for Dense layers.
            
        """
        super(UQNet, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.dropout = dropout
        self.activation = activation

        self.weight_decay_kernel = (1 - self.dropout) / (2 * self.batch_size)
        self.weight_decay_bias = 1 / (2 * self.batch_size)

        self.flat_layer = layers.Flatten()
        self.dense_layer = layers.Dense(128,
                                        activation=self.activation,
                                        kernel_regularizer=regularizers.l2(self.weight_decay_kernel),
                                        bias_regularizer=regularizers.l2(self.weight_decay_bias),)
        self.drop_layer = layers.Dropout(rate=self.dropout)
        self.out_layer = layers.Dense(64,
                                      kernel_regularizer=regularizers.l2(self.weight_decay_kernel),
                                      bias_regularizer=regularizers.l2(self.weight_decay_bias),)
        self.activation_layer = layers.Activation(self.activation)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "batch_size": self.batch_size,
                "dropout": self.dropout,
                "activation": self.activation,
            }
        )
        return config

    def call(self, inputs):
        x = self.flat_layer(inputs)
        x = self.dense_layer(x)
        x = self.drop_layer(x, training=True)
        x = self.out_layer(x)
        x = self.activation_layer(x)
        return x


class Entropy(layers.Layer):
    def __init__(self, entropy="square", **kwargs):
        """Entropy model.
        
        Args:
            entropy (str): Type of entropy function (abs or square).
            
        """
        super(Entropy, self).__init__(**kwargs)

        self.entropy = entropy
        
        if self.entropy == "abs":
            self.layer_l1 = layers.Lambda(
                lambda tensors: tf.abs(tensors[0] - tensors[1])
            )
        elif self.entropy == "square":
            self.layer_l1 = layers.Lambda(
                lambda tensors: tf.square(tensors[0] - tensors[1])
            )
        else:
            logging.info(f'Entropy function ({self.entropy}) not supported. Defaulting to "square".')
            self.layer_l1 = layers.Lambda(
                lambda tensors: tf.square(tensors[0] - tensors[1])
            )
            
        self.out = layers.Dense(
            1, activation="sigmoid", bias_initializer=initializers.RandomUniform()
        )

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "entropy": self.entropy,
            }
        )
        return config

    def call(self, inputs):
        x_l, x_r = inputs
        x = self.layer_l1([x_l, x_r])
        x = self.out(x)
        return x

def build_fn(train_args):
    """Compiles few-shot model.

    Args:
        train_args: Command-line arguments.

    Returns:
        Few-shot model.
    """
    
    if train_args.sem:
        # build multi-modal model
        image_in_shape = (128, 128, 1)
        spectra_in_shape = (train_args.cutoff-train_args.cutoff_start, 1)
        total_flat = math.prod(image_in_shape)
        in_shape = (4, 128, 128)

        inputs = layers.Input(in_shape)
        spectra_input_l, spectra_input_r, image_input_l, image_input_r = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=1))(inputs)

        # remove mask on spectra
        spectra_input_l = layers.Reshape((total_flat,1))(spectra_input_l)
        spectra_input_l = layers.Cropping1D(cropping=(0, total_flat-spectra_in_shape[0]))(spectra_input_l)

        spectra_input_r = layers.Reshape((total_flat,1))(spectra_input_r)
        spectra_input_r = layers.Cropping1D(cropping=(0, total_flat-spectra_in_shape[0]))(spectra_input_r)

        # reshape to correct size
        image_input_l = layers.Reshape(image_in_shape)(image_input_l)
        image_input_r = layers.Reshape(image_in_shape)(image_input_r)


        image_alexnet = AlexNet2D(train_args.prepool, train_args.prepool_size)

        spectra_alexnet = AlexNet1D(
            train_args.cutoff,
            train_args.prepool,
            train_args.prepool_size,
            train_args.pool_size,
            train_args.activation,
        )

        spectra_uqnet = UQNet(
            train_args.batch_size,
            train_args.dropout,
            train_args.activation,
        )

        image_uqnet = UQNet(
            train_args.batch_size,
            train_args.dropout,
            train_args.activation,
        )    

        entropy = Entropy(train_args.entropy)

        # embedding layers
        emb_l = spectra_alexnet(spectra_input_l)
        emb_r = spectra_alexnet(spectra_input_r)
        image_emb_l = image_alexnet(image_input_l)
        image_emb_r = image_alexnet(image_input_r)

        # UQNet
        spectra_seq_l = spectra_uqnet(emb_l)
        spectra_seq_r = spectra_uqnet(emb_r)
        image_seq_l = image_uqnet(image_emb_l)
        image_seq_r = image_uqnet(image_emb_r)

        # concat features after UQNet
        seq_l = tf.concat([spectra_seq_l, image_seq_l], axis=1)
        seq_r = tf.concat([spectra_seq_r, image_seq_r], axis=1)

        pred = entropy([seq_l, seq_r])

        return Model(inputs=inputs, outputs=pred)
    
    else:
        # build single modality (EDS) model
        if train_args.cutoff < 0:
            in_shape = (2, 128, 128, 1)
        else:
            in_shape = (2, train_args.cutoff-train_args.cutoff_start, 1)

        inputs = layers.Input(in_shape)

        input_l, input_r = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=1))(inputs)
        input_l = layers.Lambda(lambda x: tf.squeeze(x, axis=1))(input_l)
        input_r = layers.Lambda(lambda x: tf.squeeze(x, axis=1))(input_r)

        alexnet = AlexNet1D(train_args.cutoff, train_args.prepool, train_args.prepool_size, 
                            train_args.pool_size, train_args.activation)
        uqnet = UQNet(train_args.batch_size, train_args.dropout, train_args.activation)
        entropy = Entropy(train_args.entropy)


        emb_l = alexnet(input_l)
        emb_r = alexnet(input_r)

        seq_l = uqnet(emb_l)
        seq_r = uqnet(emb_r)

        pred = entropy([seq_l, seq_r])

        return Model(inputs=inputs, outputs=pred)


def load_FewShotNetwork(modelpath):
    """Loads pre-trained few-shot model.

    Args:
        modelpath: Path to pretrained model.

    Returns:
        Multi-modal few-shot model.
    """
    model = load_model(modelpath, custom_objects=custom_objects)
    return model


custom_objects = {"AlexNet1D": AlexNet1D,
                  "AlexNet2D": AlexNet2D,
                  "UQNet": UQNet,
                  "Entropy": Entropy,
                 }
