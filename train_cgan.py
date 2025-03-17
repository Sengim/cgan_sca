from prepare_datasets import *
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
from utils import *
from profiling_and_attack import *
from sklearn.utils import shuffle
from create_models import *

tf.keras.backend.set_floatx('float64')


class TrainCGAN:

    def __init__(self, args, models, dir_results):
        self.args = args
        self.features_dim = args["features"]
        self.datasets = PrepareDatasets(self.args)
        self.models = CreateModels(self.args, False, True)
        self.create_pretrain_model_affine()
        self.include_gen(self.args["dataset_target_dim"])
        self.dir_results = dir_results

        """ Metrics to assess quality of profiling attack: Max SNR, Guessing entropy, Ntraces_GE = 1, Perceived Information """
        self.max_snr_share_1 = []
        self.max_snr_share_2 = []
        self.max_snr_share_3 = []

        self.ge_fake = []
        self.nt_fake = []
        self.pi_fake = []

        self.ge_real = []
        self.nt_real = []
        self.pi_real = []

        self.ge_real_original = []
        self.nt_real_original = []
        self.pi_real_original = []

        self.ge_real_ta = []
        self.nt_real_ta = []
        self.pi_real_ta = []

        """ Just for plot """
        self.x_axis_epochs = []

        """ Accuracy for real and synthetic data """
        self.real_acc = []
        self.fake_acc = []
        self.real_acc_per_epoch = []
        self.fake_acc_per_epoch = []

        """ Generator and Discriminator Losses """
        self.g_loss = []
        self.d_loss = []

    def generate_reference_samples(self, batch_size):
        rnd = np.random.randint(0, self.datasets.dataset_reference.n_profiling - batch_size)
        features = self.datasets.features_reference_profiling[rnd:rnd + batch_size]
        labels = self.datasets.dataset_reference.profiling_labels[rnd:rnd + batch_size]
        return [features, labels]

    def generate_target_samples(self, batch_size):
        rnd = np.random.randint(0, self.datasets.dataset_target.n_profiling - batch_size)
        traces = self.datasets.dataset_target.x_profiling[rnd:rnd + batch_size]
        # ns = len(traces[0])
        # for trace_index in range(batch_size):
        #     shift = random.randint(-50, 50)
        #     if shift > 0:
        #         traces[trace_index][0:ns - shift] = traces[trace_index][shift:ns]
        #         traces[trace_index][ns - shift:ns] = traces[trace_index][0:shift]
        #     else:
        #         traces[trace_index][0:abs(shift)] = traces[trace_index][ns - abs(shift):ns]
        #         traces[trace_index][abs(shift):ns] = traces[trace_index][0:ns - abs(shift)]
        labels = self.datasets.dataset_target.profiling_labels[rnd:rnd + batch_size]
        #labels = np.array([self.datasets.dataset_target.profiling_plaintexts[i, self.datasets.dataset_target.target_byte] for i in range(rnd,rnd + batch_size)])
        return [traces, labels]

    @tf.function
    def train_step(self, traces_batch, label_traces, features, label_features):

        with tf.GradientTape() as disc_tape:
            fake_features = self.models.generator(traces_batch, training=True)
            real_output = self.models.discriminator([label_features, features], training=True)
            fake_output = self.models.discriminator([label_traces, fake_features], training=True)
            disc_loss = self.models.discriminator_loss(real_output, fake_output)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.models.discriminator.trainable_variables)
        self.models.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.models.discriminator.trainable_variables))

        with tf.GradientTape() as gen_tape:
            fake_features = self.models.generator(traces_batch, training=True)
            fake_output = self.models.discriminator([label_traces, fake_features], training=True)
            gen_loss = self.models.generator_loss(fake_output)
        gradients_of_generator = gen_tape.gradient(gen_loss, self.models.generator.trainable_variables)
        self.models.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.models.generator.trainable_variables))

        self.models.fake_accuracy_metric.update_state(tf.zeros_like(fake_features), fake_output)
        self.models.real_accuracy_metric.update_state(tf.ones_like(features), real_output)

        return gen_loss, disc_loss

    def compute_snr_reference_features(self):
        batch_size_reference = 3000
        rnd_reference = random.randint(0, len(self.datasets.dataset_reference.x_profiling) - batch_size_reference)
        features_reference = self.datasets.features_reference_profiling[rnd_reference:rnd_reference + batch_size_reference]
        snr_reference_features_share_1, snr_reference_features_share_2 = [], []
        if self.args["dataset_reference"] == "simulate":

            snr_reference_features_share_1 = snr_fast(features_reference,
                                                    self.datasets.dataset_reference.profiling_shares[
                                                    rnd_reference:rnd_reference + batch_size_reference, 0])
            snr_reference_features_share_2 = snr_fast(features_reference,
                                                    self.datasets.dataset_reference.profiling_shares[
                                                    rnd_reference:rnd_reference + batch_size_reference, 1])
            if self.args["dataset_target"] in ["spook_sw3", "ascadv2"]:
                    snr_reference_features_share_3 = snr_fast(features_reference,
                                                    self.datasets.dataset_reference.profiling_shares[
                                                    rnd_reference:rnd_reference + batch_size_reference, 2])
        # prepare traces from target dataset
        else:

            snr_reference_features_share_1 = snr_fast(features_reference,
                                                    self.datasets.dataset_reference.share1_profiling[self.datasets.target_byte_reference,
                                                    rnd_reference:rnd_reference + batch_size_reference])
            snr_reference_features_share_2 = snr_fast(features_reference,
                                                    self.datasets.dataset_reference.share2_profiling[self.datasets.target_byte_reference,
                                                    rnd_reference:rnd_reference + batch_size_reference])
        plt.rc('axes', labelsize=16)
        plt.rc('xtick', labelsize=12)
        plt.rc('ytick', labelsize=12)
        plt.plot(snr_reference_features_share_1)
        plt.plot(snr_reference_features_share_2)
        if self.args["dataset_target"] in ["spook_sw3", "ascadv2"]:
            plt.plot(snr_reference_features_share_3)
        plt.xlim([1, self.datasets.features_dim])
        plt.xlabel("Features")
        plt.ylabel("SNR")
        
        plt.savefig(f"{self.dir_results}/snr_reference_features.png")
        plt.close()

    def shuffle_datasets(self):
        self.datasets.dataset_target.x_profiling, self.datasets.dataset_target.profiling_labels, self.datasets.dataset_target.y_profiling = shuffle(self.datasets.dataset_target.x_profiling, self.datasets.dataset_target.profiling_labels, self.datasets.dataset_target.y_profiling)
        self.datasets.dataset_reference.x_profiling, self.datasets.dataset_reference.profiling_labels = shuffle(self.datasets.dataset_reference.x_profiling, self.datasets.dataset_reference.profiling_labels)


    def compute_snr_target_features(self, epoch, synthetic_traces=True):
        batch_size_target = 8000

        # prepare traces from target dataset
        rnd_target = random.randint(0, len(self.datasets.dataset_target.x_attack) - batch_size_target)



        if synthetic_traces:
            rnd_target = random.randint(0, len(self.datasets.dataset_target.x_validation) - batch_size_target) if self.datasets.dataset_target.name=="spook_sw3" else rnd_target
            traces_target =  self.datasets.dataset_target.x_validation[rnd_target:rnd_target + batch_size_target] if self.datasets.dataset_target.name=="spook_sw3" else self.datasets.dataset_target.x_attack[rnd_target:rnd_target + batch_size_target]
            features_target = self.models.generator.predict([traces_target])
        else:
            #features_target = self.datasets.features_target_attack[rnd_target:rnd_target + batch_size_target]
            return
        snr_target_features_share_1, snr_target_features_share_2, snr_target_features_share_3 = [],[],[]
        if not self.datasets.dataset_target.name == "spook_sw3" and not self.datasets.dataset_target.name == "ascadv2":
            snr_target_features_share_1 = snr_fast(features_target, self.datasets.dataset_target.share1_attack[self.datasets.target_byte_target, rnd_target:rnd_target + batch_size_target]).tolist()
            snr_target_features_share_2 = snr_fast(features_target, self.datasets.dataset_target.share2_attack[self.datasets.target_byte_target, rnd_target:rnd_target + batch_size_target]).tolist()
        elif self.datasets.dataset_target.name == "spook_sw3":
            snr_target_features_share_1 = snr_fast(features_target, self.datasets.dataset_target.val_shares[rnd_target:rnd_target + batch_size_target, self.datasets.target_byte_target, 0]).tolist()
            snr_target_features_share_2 = snr_fast(features_target, self.datasets.dataset_target.val_shares[ rnd_target:rnd_target + batch_size_target, self.datasets.target_byte_target, 1]).tolist()
            snr_target_features_share_3 = snr_fast(features_target, self.datasets.dataset_target.val_shares[rnd_target:rnd_target + batch_size_target, self.datasets.target_byte_target, 2]).tolist()
        elif self.datasets.dataset_target.name == "ascadv2":
            snr_target_features_share_1 = snr_fast(features_target, self.datasets.dataset_target.share1_attack[rnd_target:rnd_target + batch_size_target]).tolist()
            snr_target_features_share_2 = snr_fast(features_target, self.datasets.dataset_target.share2_attack[rnd_target:rnd_target + batch_size_target]).tolist()

            snr_target_features_share_3= snr_fast(features_target, self.datasets.dataset_target.share3_attack[ rnd_target:rnd_target + batch_size_target, self.datasets.target_byte_target]).tolist()
        plt.rc('axes', labelsize=16)
        plt.rc('xtick', labelsize=12)
        plt.rc('ytick', labelsize=12)
        plt.plot(snr_target_features_share_1, label="Share 1")
        plt.plot(snr_target_features_share_2, label="Share 2")
        if synthetic_traces and (self.datasets.dataset_target.name == "spook_sw3" or self.datasets.dataset_target.name == "ascadv2"):
            plt.plot(snr_target_features_share_3, label="Share 3")
            self.max_snr_share_3.append(np.max(snr_target_features_share_3))

        plt.xlim([1, self.datasets.features_dim])
        plt.xlabel("Features")
        plt.ylabel("SNR")
        plt.legend()
        if synthetic_traces:
            if (epoch + 1) % 200 == 0:
                plt.savefig(f"{self.dir_results}/snr_target_features_fake_{epoch}.png")
        else:
            plt.savefig(f"{self.dir_results}/snr_target_features_real_{epoch}.png")
        plt.close()

        if synthetic_traces:
            self.max_snr_share_1.append(np.max(snr_target_features_share_1))
            self.max_snr_share_2.append(np.max(snr_target_features_share_2))

            self.real_acc_per_epoch.append(self.models.real_accuracy_metric.result())
            self.fake_acc_per_epoch.append(self.models.fake_accuracy_metric.result())
            fig, ax1 = plt.subplots()
            ax1.set_xlabel("Epochs")
            ax1.set_ylabel("SNR")
            ax1.plot(self.max_snr_share_1, label="Max SNR Share 1")
            ax1.plot(self.max_snr_share_2, label="Max SNR Share 2")
            if self.datasets.dataset_target.name == "spook_sw3" or self.datasets.dataset_target.name == "ascadv2":
                ax1.plot(self.max_snr_share_3, label="Max SNR Share 3")
            ax2 = ax1.twinx()
            ax2.set_ylabel("Accuracy")
            ax2.plot(self.real_acc_per_epoch,linestyle='dotted', label="Real Accuracy")
            ax2.plot(self.fake_acc_per_epoch,linestyle='dotted', label="Fake Accuracy")
            handles, labels = [(a + b) for a, b in zip(ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels())]
            fig.legend(handles, labels)
            fig.tight_layout()
            
            plt.savefig(f"{self.dir_results}/max_snr_share_2.png")
            plt.close()
            if self.datasets.dataset_target.name == "spook_sw3" or self.datasets.dataset_target.name == "ascadv2":
                np.savez(f"{self.dir_results}/max_snr_shares.npz", max_snr_share_1=self.max_snr_share_1, max_snr_share_2=self.max_snr_share_2, max_snr_share_3=self.max_snr_share_3)
                return
            np.savez(f"{self.dir_results}/max_snr_shares.npz", max_snr_share_1=self.max_snr_share_1, max_snr_share_2=self.max_snr_share_2)

    def attack_eval(self, epoch):
        #ge_real_ta, nt_real_ta, pi_real_ta, ge_vector_real_ta = template_attack(self.datasets)
        ge_fake, nt_fake, pi_fake, ge_vector_fake = attack(self.datasets, self.models.generator, self.datasets.features_dim)
        #ge_real, nt_real, pi_real, ge_vector_real = attack(self.datasets, self.models.generator, self.datasets.features_dim,
          #                                                 synthetic_traces=False)
        #ge_real_original, nt_real_original, pi_real_original, ge_vector_real_original = attack(self.datasets, self.models.generator,
                                                                                            #    self.datasets.dataset_target.ns,
                                                                                            #    synthetic_traces=False, original_traces=True)
        self.ge_fake.append(ge_fake)
        self.nt_fake.append(nt_fake)
        self.pi_fake.append(pi_fake)
        # self.ge_real.append(ge_real)
        # self.nt_real.append(nt_real)
        # self.pi_real.append(pi_real)
        # self.ge_real_original.append(ge_real_original)
        # self.nt_real_original.append(nt_real_original)
        # self.pi_real_original.append(pi_real_original)

        self.x_axis_epochs.append(epoch + 1)

        plt.plot(self.x_axis_epochs, self.ge_fake, label="fake")
        # plt.plot(self.x_axis_epochs, self.ge_real, label="real")
        # plt.plot(self.x_axis_epochs, self.ge_real_original, label="real raw")
        #plt.plot(self.x_axis_epochs, self.ge_real_ta, label="real TA (LDA)")
        plt.legend()
        plt.xlabel("CGAN Training Epoch")
        plt.ylabel("Guessing Entropy")
        plt.savefig(f"{self.dir_results}/ge.png")
        plt.close()

        plt.plot(self.x_axis_epochs, self.nt_fake, label="fake")
        # plt.plot(self.x_axis_epochs, self.nt_real, label="real")
        # plt.plot(self.x_axis_epochs, self.nt_real_original, label="real raw")
        #plt.plot(self.x_axis_epochs, self.nt_real_ta, label="TA (LDA)")
        # plt.legend()
        # plt.xlabel("CGAN Training Epoch")
        # plt.ylabel("Number of Traces for GE=1")
        # plt.yscale('log')
        # #plt.savefig(f"{self.dir_results}/nt.png")
        # plt.close()

        # plt.plot(self.x_axis_epochs, self.pi_fake, label="fake")
        # plt.plot(self.x_axis_epochs, self.pi_real, label="real")
        # plt.plot(self.x_axis_epochs, self.pi_real_original, label="real raw")
        #plt.plot(self.x_axis_epochs, self.pi_real_ta, label="TA (LDA)")
        # plt.legend()
        # plt.xlabel("CGAN Training Epoch")
        # plt.ylabel("Perceived Information")
        # plt.savefig(f"{self.dir_results}/pi.png")
        # plt.close()

        np.savez(f"{self.dir_results}/metrics.npz",
                 ge_fake=self.ge_fake,
                 nt_fake=self.nt_fake,
                 pi_fake=self.pi_fake,
                 ge_real=self.ge_real,
                 nt_real=self.nt_real,
                 pi_real=self.pi_real,
                 ge_real_original=self.ge_real_original,
                 nt_real_original=self.nt_real_original,
                 pi_real_original=self.pi_real_original,
                 ge_real_ta=self.ge_real_ta,
                 nt_real_ta=self.nt_real_ta,
                 pi_real_ta=self.pi_real_ta
                 )

        # plt.plot(ge_vector_fake, label="fake")
        # # plt.plot(ge_vector_real, label="real")
        # # plt.plot(ge_vector_real_original, label="real raw")
        # plt.legend()
        # plt.xscale('log')
        # plt.xlabel("Attack Traces")
        # plt.ylabel("Guessing Entropy")
        # plt.savefig(f"{self.dir_results}/ge.png")
        # plt.close()

        np.savez(f"{self.dir_results}/ge_vector_epoch_{epoch}.npz",
                 ge_vector_fake=ge_vector_fake,
                #  ge_vector_real=ge_vector_real,
                #  ge_vector_real_original=ge_vector_real_original,
                 )

    # def attack_eval_synthetc(self, epoch):
    #     ge, nt, pi, ge_vector = attack(self.datasets, self.models.generator, self.datasets.features_dim)
    #     plt.xscale('log')
    #     plt.plot(ge_vector, label="50000 traces")
    #     for n_profiling in [100000, 250000, 500000]:
    #         ge, nt, pi, ge_vector = attack(self.datasets, self.models.generator, self.datasets.features_dim, n_profiling=n_profiling)
    #         plt.xscale('log')
    #         plt.plot(ge_vector, label=f"{n_profiling} traces")
    #     plt.legend()
    #     plt.savefig(f"{self.dir_results}/ge_vector_epoch_{epoch}.png")
    #     plt.close()

    def train(self):
        training_set_size = max(self.datasets.dataset_reference.n_profiling, self.datasets.dataset_target.n_profiling)

        # determine half the size of one batch, for updating the discriminator
        batch_size = self.args["batch_size"]
        n_batches = int(training_set_size / batch_size)

        # manually enumerate epochs
        for e in range(self.args["epochs"]):
            for b in range(n_batches):
                [features_reference, labels_reference] = self.generate_reference_samples(batch_size)
                [traces_target, labels_target] = self.generate_target_samples(batch_size)
                
                # Custom training step for speed and versatility
                g_loss, d_loss = self.train_step(traces_target, labels_target, features_reference, labels_reference)
                self.full_class.train_on_batch(traces_target,to_categorical(labels_target, num_classes=256) )
                if (b + 1) % 100 == 0:
                    self.real_acc.append(self.models.real_accuracy_metric.result())
                    self.fake_acc.append(self.models.fake_accuracy_metric.result())
                    self.g_loss.append(g_loss)
                    self.d_loss.append(d_loss)

                    plt.plot(self.real_acc, label="Real")
                    plt.plot(self.fake_acc, label="Fake")
                    plt.axhline(y=0.5, linestyle="dashed", color="black")
                    plt.legend()
                    plt.savefig(f"{self.dir_results}/acc.png")
                    plt.close()

                    plt.plot(self.g_loss, label="g_loss")
                    plt.plot(self.d_loss, label="d_Loss")
                    plt.legend()
                    plt.savefig(f"{self.dir_results}/loss.png")
                    plt.close()

                    print(
                        f"epoch: {e}, batch: {b}, d_loss: {d_loss}, g_loss: {g_loss}, real_acc: {self.models.real_accuracy_metric.result()}, fake_acc: {self.models.fake_accuracy_metric.result()}")



            # Split eval steps up as attacking takes significant time while snr computation is fast
            self.shuffle_datasets()
            if e == 0 and not self.datasets.dataset_target.name == "ches_ctf":

                self.compute_snr_reference_features()
                self.compute_snr_target_features(e, synthetic_traces=False)
            if (e + 1) % 1== 0 and not self.datasets.dataset_target.name == "ches_ctf":
                #Redo shuffling 

                #self.datasets.dataset_target.x_profiling =  augment_renew(self.datasets.dataset_target.x_profiling, 15, self.args["n_profiling_target"])
                self.compute_snr_target_features(e)
                # self.attack_eval_synthetic(e)
            if (e + 1) % 200 == 0 and not self.datasets.dataset_target.name == "spook_sw3":
                self.attack_eval(e)
                # self.models.generator.save(
                #     f"{self.dir_results}/generator_{self.datasets.traces_target_dim}_{self.datasets.traces_reference_dim}_epoch_{e}.h5")
        np.savez(f"{self.dir_results}/acc_and_loss.npz",
                             g_loss=self.g_loss, d_loss=self.d_loss,
                             real_acc=self.real_acc, fake_acc=self.fake_acc)
        self.models.generator.save(
            f"{self.dir_results}/generator_{self.datasets.traces_target_dim}_{self.datasets.traces_reference_dim}_epoch_{self.args['epochs'] - 1}.h5")
    
    def create_pretrain_model_affine(self):
        affine_mod = self.models.affine_mod
        affine_class = self.classification(self.features_dim, affine_mod, classes=255)
        y_label = self.datasets.dataset_reference.profiling_masks[:, 0] -1
        print(y_label.__contains__(255), y_label.__contains__(0))
        y_attack = self.datasets.dataset_reference.attack_masks[:, 0] -1
        optim = tf.keras.optimizers.Adam(0.01,  beta_1 =0.5)
        affine_class.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
        # optim = tf.keras.optimizers.Adam(0.00025)
        affine_class.fit(x = self.datasets.dataset_reference.x_profiling, y =to_categorical(y_label, num_classes=255), validation_data=(self.datasets.dataset_reference.x_attack, to_categorical(y_attack, num_classes=255)),  epochs=5, shuffle=True, batch_size=200)

        bool_part = self.models.bool_mod
        class_model = self.classification(self.features_dim, bool_part)
        optim = tf.keras.optimizers.Adam(0.01)
        class_model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
        y_label = self.datasets.dataset_reference.profiling_shares[:, 1].copy()
        y_attack= self.datasets.dataset_reference.attack_shares[:, 1].copy()
        print(y_label.shape, self.datasets.dataset_reference.x_profiling.shape)
        class_model.fit(x = self.datasets.dataset_reference.x_profiling, y =to_categorical(y_label), validation_data=(self.datasets.dataset_reference.x_attack, to_categorical(y_attack)),  epochs=5, shuffle=True, batch_size=200)

        for i in range(self.datasets.dataset_reference.order, 1, -1):
            print(f"-------------------{i}-----------------------")
            y_label = y_label ^ self.datasets.dataset_reference.profiling_shares[:, i]
            y_attack = y_attack ^ self.datasets.dataset_reference.attack_shares[:, i]
            class_model.fit(x = self.datasets.dataset_reference.x_profiling, y =to_categorical(y_label), validation_data=(self.datasets.dataset_reference.x_attack, to_categorical(y_attack)),  epochs=30, shuffle=True, batch_size=200)
        

        print(f"-------------------AFFINE-----------------------")
        self.models.bool_mod.trainable=False
        self.models.affine_mod.trainable=False
        base_model = self.models.base_model
        self.class_model = self.classification(self.features_dim, base_model)
        optim = tf.keras.optimizers.Adam(0.001)
        self.class_model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
        y_label = self.datasets.dataset_reference.profiling_masks[:, -1]
        y_attack = self.datasets.dataset_reference.attack_masks[:, -1]
        self.class_model.fit(x = self.datasets.dataset_reference.x_profiling, y =to_categorical(y_label), validation_data=(self.datasets.dataset_reference.x_attack, to_categorical(y_attack)),  epochs=15, shuffle=True, batch_size=400)
        #self.models.base_model.trainable = False
        return
    
    
    def classification(self, input_dim, pretrain_model, classes=256):
        in_features = Input(shape=(input_dim, ))
        
        x = pretrain_model(in_features)
        out = Dense(classes, activation='softmax')(x)
        print(out.shape)
        model = Model(inputs= in_features, outputs=out)
        return model
    
    def include_gen(self, input_dim):
        in_features = Input(shape=(input_dim, ))
        x = self.models.generator(in_features)
        out = self.class_model(x)
        self.full_class = Model(inputs= in_features, outputs=out)
        optim = tf.keras.optimizers.Adam(0.001,  beta_1 =0.5)
        self.full_class.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
        #y_label = self.datasets.dataset_reference.profiling_masks[:, -1]
        return 