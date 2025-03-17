import tensorflow as tf
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.regularizers import *
import tensorflow.keras.backend as K
from adahessian import AdaHessian
import random



class CreateModels:

    def __init__(self, args, random_hp=False, include_pretrain=False):
        # helper functions for tensorflow compiling
        self.real_accuracy_metric = BinaryAccuracy()
        self.fake_accuracy_metric = BinaryAccuracy()
        self.cross_entropy = BinaryCrossentropy(from_logits=True)
        self.cross_entropy_disc = BinaryCrossentropy(from_logits=True)

        if include_pretrain:
            self.generator_optimizer = AdaHessian(0.002)
            self.discriminator_optimizer = tf.keras.optimizers.Adam(0.0025, beta_1=0.5)
            self.affine_mod = self.pretrain_affine_part(args["features"])
            self.bool_mod = self.pre_train_bool_part(args["features"])
            self.base_model = self.combine(args["features"], self.affine_mod, self.bool_mod)
            self.discriminator = self.disc_with_pretrain(args["features"], self.base_model)
            self.generator = self.generator = self.define_generator(args["dataset_target_dim"], args["features"])
            return
        if random_hp:
            self.hp_d = {
                "disc": random.choice(["bilinear", "dropout"]),
                }
            self.hp_g = {
                "seed": random.randint(0, 1000000000),
                "neurons": random.choice([40, 50, 100, 200, 400]),
                "layers": random.choice([2, 3, 4,5]),
                "activation": random.choice(["elu", "selu", "relu"]),
                "learning_rate": random.choice([ 0.0005, 0.0001, 0.0002, 0.00025, 0.00005]),
                "kernel_initializer": random.choice(
                    ["random_uniform", "he_uniform", "glorot_uniform", "random_normal", "he_normal", "glorot_normal"]),
                "conv_stride": random.choice([8, 16, 32]),
                "lin_activation": random.choice([True, False]),
                "conv": random.choice([True, False]),
                "filter_len": random.choice([8, 16, 32, 64, 128]),
                "init_filters": random.choice([8, 16, 4, 2]),
                "conv_layers": random.choice([1, 2, 3])
            }
            self.generator_optimizer = tf.keras.optimizers.Adam(self.hp_g["learning_rate"], beta_1=0.5)
            self.discriminator_optimizer = tf.keras.optimizers.Adam(0.0025, beta_1=0.5)
            if args["discriminator"] and args["discriminator"] == "bilinear":
                 self.discriminator = self.define_discriminator_split_bilinear(args["features"],
                                                        n_classes=9 if args["leakage_model"] == "HW" else 256)
            else:
                 self.discriminator = self.define_discriminator(args["features"],
                                                        n_classes=9 if args["leakage_model"] == "HW" else 256)

            # create the generator
            self.generator = self.define_mlp_generator_random(args["dataset_target_dim"], args["features"])
            
        else:
            self.generator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
            self.discriminator_optimizer = tf.keras.optimizers.Adam(0.0025, beta_1=0.5)
            # create the discriminator
            classes = 9 if args["leakage_model"] == "HW" else 256
            print(classes)
            # self.discriminator = self.define_discriminator(args["features"],
            #                                             n_classes=9 if args["leakage_model"] == "HW" else 256)
            # # create the generator
            # self.generator = self.define_generator(args["dataset_target_dim"], args["features"])
            #self.best_models_random_search(args["dataset_reference"], args["dataset_target"])
            self.best_models_10000_var_dpa()
            # create the discriminator
            self.discriminator = self.define_discriminator_random(args["features"], n_classes=classes)
            # create the generator
            self.generator = self.define_generator_random(args["dataset_target_dim"], args["features"])
            self.generator = self.define_generator(args["dataset_target_dim"], args["features"])
            self.discriminator = self.define_discriminator(args["features"], n_classes=classes)

    def best_models_10000_var_dpa(self):
        self.hp_d = {
                        'neurons_embed':
                        200,
                        'neurons_dropout':
                        500,
                        'neurons_bilinear':
                        100,
                        'layers_embed':
                        2,
                        'layers_dropout':
                        1,
                        'layers_bilinear':
                        2,
                        'dropout':
                        0.6,
                        }
        self.hp_g = {
                    'neurons_1':
                    200,
                    'layers':
                    1,
                    'activation':
                    'selu',
                    }

    def best_models_random_search(self, reference, target):
        if reference == "ascad-variable":
            if target == "ASCAD":
                # reference ascad-variable (25000) vs target ASCAD (2500) (ID Model, 100 searches):
                self.hp_d = {
                    'neurons_embed': 100, 'neurons_dropout': 200, 'layers_embed': 2, 'layers_dropout': 3, 'dropout': 0.7,
                    'neurons_bilinear': 200, 'layers_bilinear': 1,
                }
                self.hp_g = {
                    'neurons_1': 300, 'layers': 1, 'activation': 'linear'
                }
            if target == "dpa_v42":
                # reference ascad-variable (25000) vs target dpa_v42 (7500) (ID Model, 100 searches):
                self.hp_d = {
                    'neurons_embed': 100, 'neurons_dropout': 200, 'layers_embed': 1, 'layers_dropout': 1, 'dropout': 0.8
                }
                self.hp_g = {
                    'neurons_1': 200, 'layers': 4, 'activation': 'linear', 'neurons_2': 200, 'neurons_3': 200, 'neurons_4': 100
                }
            if target == "ches_ctf":
                self.hp_d = {
                    'neurons_embed': 200, 'neurons_dropout': 200, 'layers_embed': 2, 'layers_dropout': 1, 'dropout': 0.5
                }
                self.hp_g = {
                    'neurons_1': 100, 'layers': 4, 'activation': 'linear', 'neurons_2': 100, 'neurons_3': 100, 'neurons_4': 100
                }
            if target == "eshard":
                self.hp_d = {
                    'neurons_embed': 200, 'neurons_dropout': 200, 'layers_embed': 1, 'layers_dropout': 1, 'dropout': 0.7
                }
                self.hp_g = {
                    'neurons_1': 500, 'layers': 3, 'activation': 'leakyrelu', 'neurons_2': 500, 'neurons_3': 100
                }
        if reference == "ASCAD":
            if target == "ascad-variable":
                # reference ASCAD (10000) vs target ascad-variable (6250) (ID Model, 100 searches):
                self.hp_d = {
                    'neurons_embed': 200, 'neurons_dropout': 200, 'layers_embed': 2, 'layers_dropout': 1, 'dropout': 0.6,
                }
                self.hp_g = {
                    'neurons_1': 200, 'layers': 3, 'activation': 'leakyrelu', 'neurons_2': 200, 'neurons_3': 100
                }
            if target == "dpa_v42":
                # reference ASCAD (10000) vs target dpa_v42 (7500) (ID Model, 100 searches):
                self.hp_d = {
                    'neurons_embed': 200, 'neurons_dropout': 100, 'layers_embed': 2, 'layers_dropout': 2, 'dropout': 0.8
                }
                self.hp_g = {
                    'neurons_1': 300, 'layers': 2, 'activation': 'linear', 'neurons_2': 100
                }
            if target == "ches_ctf":
                # reference ASCAD (10000) vs target eshard (1400) (HW Model, 100 searches):
                self.hp_d = {
                    'neurons_embed': 200, 'neurons_dropout': 200, 'layers_embed': 2, 'layers_dropout': 1, 'dropout': 0.5
                }
                self.hp_g = {
                    'neurons_1': 100, 'layers': 4, 'activation': 'linear', 'neurons_2': 100, 'neurons_3': 100, 'neurons_4': 100
                }
            if target == "eshard":
                # reference ASCAD (10000) vs target eshard (1400) (HW Model, 100 searches):
                self.hp_d = {
                    'neurons_embed': 500, 'neurons_dropout': 200, 'layers_embed': 1, 'layers_dropout': 1, 'dropout': 0.7
                }
                self.hp_g = {
                    'neurons_1': 500, 'layers': 2, 'activation': 'selu', 'neurons_2': 400
                }
        if reference == "dpa_v42":
            if target == "ascad-variable":
                # reference dpa_v42 (15000) vs target ascad-variable (6250) (ID Model, 100 searches):
                self.hp_d = {
                    'neurons_embed': 500, 'neurons_dropout': 100, 'layers_embed': 1, 'layers_dropout': 1, 'dropout': 0.6
                }
                self.hp_g = {
                    'neurons_1': 100, 'layers': 1, 'activation': 'elu'
                }
            if target == "ASCAD":
                # reference dpa_v42 (15000) vs target ASCAD (2500) (ID Model, 100 searches):
                self.hp_d = {
                    'neurons_embed': 100, 'neurons_dropout': 200, 'layers_embed': 1, 'layers_dropout': 1, 'dropout': 0.7
                }
                self.hp_g = {
                    'neurons_1': 500, 'layers': 4, 'activation': 'linear', 'neurons_2': 100, 'neurons_3': 100, 'neurons_4': 100
                }
            if target == "ches_ctf":
                # reference dpa_v42 (15000) vs target eshard (1400) (HW Model, 100 searches):
                self.hp_d = {
                    'neurons_embed': 500, 'neurons_dropout': 500, 'layers_embed': 1, 'layers_dropout': 1, 'dropout': 0.8
                }
                self.hp_g = {
                    'neurons_1': 100, 'layers': 1, 'activation': 'linear'
                }
            if target == "eshard":
                # reference dpa_v42 (15000) vs target eshard (1400) (HW Model, 100 searches):
                self.hp_d = {
                    'neurons_embed': 500, 'neurons_dropout': 500, 'layers_embed': 1, 'layers_dropout': 1, 'dropout': 0.6
                }
                self.hp_g = {
                    'neurons_1': 400, 'layers': 2, 'activation': 'selu', 'neurons_2': 300
                }
        if reference == "eshard":
            if target == "ascad-variable":
                # reference eshard (1400) vs target ascad-variable (6250) (ID Model, 100 searches):
                self.hp_d = {
                    'neurons_embed': 200, 'neurons_dropout': 100, 'layers_embed': 1, 'layers_dropout': 1, 'dropout': 0.7
                }
                self.hp_g = {
                    'neurons_1': 100, 'layers': 4, 'activation': 'linear', 'neurons_2': 100, 'neurons_3': 100, 'neurons_4': 100
                }
            if target == "ASCAD":
                # reference eshard (1400) vs target ASCAD (2500) (ID Model, 100 searches):
                self.hp_d = {
                    'neurons_embed': 100, 'neurons_dropout': 500, 'layers_embed': 1, 'layers_dropout': 1, 'dropout': 0.6
                }
                self.hp_g = {
                    'neurons_1': 400, 'layers': 2, 'activation': 'linear', 'neurons_2': 300
                }
            if target == "dpa_v42":
                # reference eshard (1400) vs target dpa_v42 (7500) (ID Model, 100 searches):
                self.hp_d = {
                    'neurons_embed': 100, 'neurons_dropout': 500, 'layers_embed': 2, 'layers_dropout': 2, 'dropout': 0.8
                }
                self.hp_g = {
                    'neurons_1': 500, 'layers': 1, 'activation': 'linear'
                }

    def new_gen_test(self, input_dim: int, output_dim: int):

        in_traces = Input(shape=(input_dim,))
        out_layer = Dense(output_dim, activation='linear')(in_traces)

        model = Model([in_traces], out_layer)
        model.summary()
        return model

    def discriminator_loss(self, real, fake):
        real_loss = self.cross_entropy_disc(tf.ones_like(real), real)
        fake_loss = self.cross_entropy_disc(tf.zeros_like(fake), fake)
        return real_loss + fake_loss

    def generator_loss(self, fake):
        return self.cross_entropy(tf.ones_like(fake), fake)
    
    def disc_with_pretrain_random(self, features_dim: int,pretrain_model, n_classes: int = 256, kern_init='random_normal'):
        # label input
        in_label = Input(shape=1)
        y = Embedding(n_classes, n_classes)(in_label)
        for l_i in range(self.hp_d["layers_embed"]):
            y = Dense(self.hp_d["neurons_embed"], kernel_initializer=kern_init)(y)
            y = LeakyReLU()(y)
        #y = LeakyReLU()(y)
        y = Flatten()(y)

        in_features = Input(shape=(features_dim,))

        feat_pros = pretrain_model(in_features)
        merge = Concatenate()([y, feat_pros])


        # x = Dense(100, kernel_initializer=kern_init)(merge)
        # x = LeakyReLU()(x)
        # x = Dropout(0.60)(x)
        # x = Dense(100, kernel_initializer=kern_init)(merge)
        # x = LeakyReLU()(x)
        # x = Dropout(0.60)(x)
        x = None
        for l_i in range(self.hp_d["layers_dropout"]):
            x = Dense(self.hp_d["neurons_dropout"], kernel_initializer=kern_init)(merge if l_i == 0 else x)
            x = LeakyReLU()(x)
            x = Dropout(self.hp_d["dropout"])(x)


        # output
        out_layer = Dense(1, activation='sigmoid')(x)

        # model = Model([input_traces, in_label, in_features], out_layer)
        model = Model([in_label, in_features], out_layer)
        # model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=self.hp_d["learning_rate"]), metrics=['accuracy'])
        model.summary()
        return model

    def define_discriminator(self, features_dim: int, n_classes: int = 256, kern_init='random_normal'):
        # label input
        in_label = Input(shape=1)
        y = Embedding(n_classes, n_classes)(in_label)
        y = Dense(200, kernel_initializer=kern_init)(y)
        # y = LeakyReLU()(y)
        y = Flatten()(y)

        in_features = Input(shape=(features_dim,))
        #in_features = Dropout(0.5)(in_features)

        merge = Concatenate()([y, in_features])

        x = Dense(100, kernel_initializer=kern_init)(merge)
        x = LeakyReLU()(x)
        x = Dropout(0.60)(x)

        # output
        out_layer = Dense(1, activation='sigmoid')(x)

        model = Model([in_label, in_features], out_layer)
        model.summary()
        return model

    def define_discriminator_split_bilinear(self, features_dim: int, n_classes: int = 256, kern_init='random_normal'):
        # label input
        in_label = Input(shape=1)
        l1 = Embedding(n_classes, n_classes)(in_label)
        l1 = Dense(200, kernel_initializer=kern_init)(l1)
        l1 = LeakyReLU()(l1)
        l1 = Flatten()(l1)


        in_features = Input(shape=(features_dim,1))
        in_features_reshaped = Reshape((1, features_dim//2))(in_features[:, 0:features_dim//2])
        # # input_traces = Input(shape=self.traces_dim)
        temp = in_features[:, features_dim//2:features_dim]
        dot_lambda = lambda x_arr: tf.multiply(x_arr[0], x_arr[1])
        in_features_dot = Lambda(dot_lambda)([temp,in_features_reshaped])
    #Should be (None, 50, 50)
        # Should be (None, 50, 50)
        print(in_features_dot.shape)
        
        x = Flatten()(in_features_dot)
        # Should be (None, 2500(features_dim^2))
        x = Dense(30, kernel_initializer=kern_init, activation='linear')(x)
        merge = Concatenate()([l1, x])

        x = Dense(10, kernel_initializer=kern_init)(merge)
        x = LeakyReLU()(x)

        # output
        out_layer = Dense(1, activation='sigmoid')(x)

        # model = Model([input_traces, in_label, in_features], out_layer)
        model = Model([in_label, in_features], out_layer)
        # model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=self.hp_d["learning_rate"]), metrics=['accuracy'])
        model.summary()
        return model

    def define_discriminator_split_bilinear_order(self, features_dim: int, n_classes: int = 256, order=2, kern_init='random_normal'):
        # label input
        in_label = Input(shape=1)
        l1 = Embedding(n_classes, n_classes)(in_label)
        l1 = Dense(30, kernel_initializer=kern_init)(l1)
        l1 = LeakyReLU()(l1)
        l1 = Flatten()(l1)

        in_features = Input(shape=(features_dim, 1))
        temp = in_features[:, 0:features_dim // order]
        for i in range(1, order):
            in_features_reshaped = Reshape((1, features_dim // order))(
                in_features[:, (features_dim * i) // order:(features_dim * (i + 1)) // order])
            # # input_traces = Input(shape=self.traces_dim)
            temp = Reshape((features_dim // order, 1))(temp)
            dot_lambda = lambda x_arr: tf.multiply(x_arr[0], x_arr[1])
            temp = Lambda(dot_lambda)([temp, in_features_reshaped])
            # Should be (None, 50, 50)
            temp = Flatten()(temp)
            temp = Dense(features_dim // order, kernel_initializer=kern_init)(temp)

        # Should be (None, 2500(features_dim^2))
        in_features_dot = LeakyReLU()(temp)
        # merge = Concatenate()([input_traces, l1, in_features])
        merge = Concatenate()([l1, in_features_dot])

        x = Dense(10, kernel_initializer=kern_init)(merge)
        x = LeakyReLU()(x)

        # output
        out_layer = Dense(1, activation='sigmoid')(x)

        # model = Model([input_traces, in_label, in_features], out_layer)
        model = Model([in_label, in_features], out_layer)
        # model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=self.hp_d["learning_rate"]), metrics=['accuracy'])
        # model.summary()
        return model

    #define the standalone generator model
    def define_generator_conv(self, input_dim: int, output_dim: int, n_classes=256):
        # input_random_data = Input(shape=(self.traces_target_dim,))
        # rnd = Dense(400, activation='elu')(input_random_data)
    
        in_traces = Input(shape=(input_dim,))
        reshaped = Reshape((input_dim, 1))(in_traces)
        x = Conv1D(16, 32, kernel_initializer='he_uniform', activation='selu', strides=16, padding='same', name='block1_conv1')(reshaped)
        # x = BatchNormalization()(x)
        # x = Conv1D(8, 11, kernel_initializer='he_uniform', activation='relu',strides=2,  padding='same', name='block1_conv2')(x)
        # x = BatchNormalization()(x)
        # x = Conv1D(16, 11, kernel_initializer='he_uniform', activation='relu',strides=2,  padding='same', name='block1_conv3')(x)
        # x = BatchNormalization()(x)
        x = Flatten(name='flatten')(x)
        x = Dense(output_dim, activation='selu')(x)
        # x = Dropout(0.2)(x)
        out_layer = Dense(output_dim, activation='linear')(x)
    
        model = Model([in_traces], out_layer)
        model.summary()
        return model

    def add_gaussian_noise(self, x):
        mean = 0.0  # Mean of the Gaussian distribution
        stddev = 0.01  # Standard deviation of the Gaussian distribution
        noise = K.random_normal(shape=K.shape(x), mean=mean, stddev=stddev)
        return x + noise
    

    def define_mlp_generator_random(self, input_dim: int, output_dim: int):
        tf.random.set_seed(self.hp_g["seed"])
        input_layer = Input(shape=(input_dim,))
        activation = "linear" if self.hp_g["lin_activation"] else self.hp_g["activation"]
        x = None
        if self.hp_g["conv"]:
             reshaped = Reshape((input_dim, 1))(input_layer)
             for i in range(self.hp_g["conv_layers"]):
                 x = Conv1D(self.hp_g["init_filters"]*(2**i), self.hp_g["filter_len"], kernel_initializer=self.hp_g["kernel_initializer"], activation=activation, strides=self.hp_g["conv_stride"], padding='same')(reshaped if x is None else x)
             x = Flatten()(x)
        
        
        for i in range(self.hp_g["layers"]):
            x = Dense( self.hp_g["neurons"], activation=activation, kernel_initializer=self.hp_g["kernel_initializer"])(input_layer if x is None else x)
        out_layer = Dense(output_dim, activation="linear", kernel_initializer=self.hp_g["kernel_initializer"])(x)
        model = Model([input_layer], out_layer)
        model.summary()
        return model
    

    # define the standalone generator model
    def define_generator(self, input_dim: int, output_dim: int, n_classes=256):
        # input_random_data = Input(shape=(self.traces_target_dim,))
        # rnd = Dense(400, activation='elu')(input_random_data)

        in_traces = Input(shape=(input_dim,))
        # x = Lambda(self.add_gaussian_noise)(in_traces)
        x = Dense(100, activation='selu')(in_traces)
        x = Dense(100, activation='selu')(x)
        x = Dense(100, activation='selu')(x)
        out_layer = Dense(output_dim, activation='linear')(x)

        model = Model([in_traces], out_layer)
        model.summary()
        return model

    def define_discriminator_random(self, features_dim: int, n_classes: int = 256, kern_init='random_normal'):
        # label input
        in_label = Input(shape=1)
        y = Embedding(n_classes, n_classes)(in_label)
        for l_i in range(self.hp_d["layers_embed"]):
            y = Dense(self.hp_d["neurons_embed"], kernel_initializer=kern_init)(y)
            y = LeakyReLU()(y)
        y = Flatten()(y)

        in_features = Input(shape=(features_dim,))

        merge = Concatenate()([y, in_features])

        x = None
        for l_i in range(self.hp_d["layers_dropout"]):
            x = Dense(self.hp_d["neurons_dropout"], kernel_initializer=kern_init)(merge if l_i == 0 else x)
            x = LeakyReLU()(x)
            x = Dropout(self.hp_d["dropout"])(x)

        # output
        out_layer = Dense(1, activation='sigmoid')(x)

        model = Model([in_label, in_features], out_layer)
        model.summary()
        return model

    def disc_with_pretrain(self, features_dim: int,pretrain_model, n_classes: int = 256, kern_init='random_normal'):
        # label input
        in_label = Input(shape=1)
        y = Embedding(n_classes, n_classes)(in_label)
        y = Dense(100, kernel_initializer=kern_init)(y)
        #y = LeakyReLU()(y)
        y = Flatten()(y)

        in_features = Input(shape=(features_dim,))

        feat_pros = pretrain_model(in_features)
        merge = Concatenate()([y, feat_pros])

        # x = Dense(100, kernel_initializer=kern_init)(merge)
        # x = LeakyReLU()(x)
        # x = Dropout(0.60)(x)
        # x = Dense(100, kernel_initializer=kern_init)(merge)
        # x = LeakyReLU()(x)
        # x = Dropout(0.60)(x)
        x = Dense(100, kernel_initializer=kern_init)(merge)
        x = LeakyReLU()(x)
        x = Dropout(0.60)(x)

        # output
        out_layer = Dense(1, activation='sigmoid')(x)

        # model = Model([input_traces, in_label, in_features], out_layer)
        model = Model([in_label, in_features], out_layer)
        # model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=self.hp_d["learning_rate"]), metrics=['accuracy'])
        model.summary()
        return model

    # define a random generator model
    def define_generator_random(self, input_dim: int, output_dim: int):

        in_traces = Input(shape=(input_dim,))
        x = None
        for l_i in range(self.hp_g["layers"]):
            x = Dense(self.hp_g[f"neurons_{l_i + 1}"],
                      activation=self.hp_g["activation"] if self.hp_g["activation"] != "leakyrelu" else None)(in_traces if l_i == 0 else x)
            if self.hp_g["activation"] == "leakyrelu":
                x = LeakyReLU()(x)
        out_layer = Dense(output_dim, activation='linear')(x)

        model = Model([in_traces], out_layer)
        model.summary()
        return model
    def pre_train_model(self, input_dim: int):
        in_features = Input(shape=(input_dim, ))
        x = Dense(20, activation='selu')(in_features)
        x = Dense(20, activation='selu')(x)
        out = Dense(20, activation='selu')(x)
        model = Model(inputs= in_features, outputs=out)
        return model

    def pre_train_bool_part(self, input_dim: int):
        in_features = Input(shape=(input_dim, ))
        x = Dense(10, activation='selu')(in_features)
        x = Dense(10, activation='selu')(x)
        out = Dense(8, activation='selu')(x)
        model = Model(inputs= in_features, outputs=out)
        return model

    def pretrain_affine_part(self, input_dim: int):
        in_features = Input(shape=(input_dim, ))
        x = Dense(10, activation='selu')(in_features)
        x = Dense(10, activation='selu')(x)
        out = Dense(8, activation='selu')(x)
        model = Model(inputs= in_features, outputs=out)
        return model

    def combine(self, input_dim: int, affine_mod, rest_mod):
        in_features = Input(shape=(input_dim, ))
        affine_part = affine_mod(in_features)
        rest_part = rest_mod(in_features)
        rest_part = Reshape((1, rest_part.shape[1]))(rest_part)
        affine_part = Reshape((affine_part.shape[1], 1))(affine_part)
        dot_lambda = lambda x_arr: tf.multiply(x_arr[0], x_arr[1])
        x= Lambda(dot_lambda)([affine_part,rest_part])
        x= Flatten()(x)

        # # x = Flatten()(in_features_dot)
        # dot_lambda = lambda x_arr: tf.multiply(x_arr[0], x_arr[1])
        # x = Lambda(dot_lambda)([affine_part, rest_part])
        #x = Concatenate()([affine_part, rest_part])
        print(x.shape)
        x = Dense(20, activation='selu')(x)
        out = Dense(20, activation='selu')(x)
        model = Model(inputs= in_features, outputs=out)
        return model
