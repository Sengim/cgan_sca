from profiling_and_attack import *
from prepare_datasets import *
import matplotlib.pyplot as plt


def define_generator(input_dim: int, output_dim: int):
    in_traces = Input(shape=(input_dim,))
    x = Dense(400, activation='linear')(in_traces)
    x = Dense(200, activation='linear')(x)
    x = Dense(100, activation='linear')(x)
    out_layer = Dense(output_dim, activation='linear')(x)
    model = Model([in_traces], out_layer)
    model.summary()
    return model


def ta(features_profiling, features_attack, classes, profiling_labels, attack_labels, labels_key_hypothesis_attack, correct_key_attack):
    """ Template Attack """
    mean_v, cov_v, template_classes = template_training(features_profiling, profiling_labels, pool=False)
    predictions = template_attacking(mean_v, cov_v, features_attack[:2000], template_classes, attack_labels[:2000], classes)

    """ Check if we are able to recover the key from the target/attack measurements """
    ge, ge_vector, nt = guessing_entropy(predictions, labels_key_hypothesis_attack[:, :2000], correct_key_attack, 2000)
    pi = information(predictions, attack_labels[:2000], classes)
    return ge, nt, pi, ge_vector


dataset_loaders = {
    "ascad-variable": ReadASCADr,
    "ASCAD": ReadASCADf,
    "eshard": ReadEshard,
    "dpa_v42": ReadDPAV42,
    "aes_sim_reference": ReadAESSimReference,
    "aes_hd_mm": ReadAESHDMM
}

# dpa_v42_vs_ASCAD_15_06_2023_21_49_25_6323044 2500
# dpa_v42_vs_ASCAD_15_06_2023_21_49_25_5136069 5000
# dpa_v42_vs_ASCAD_15_06_2023_21_49_25_7604445 10000
# dpa_v42_vs_ASCAD_15_06_2023_22_15_51_2811283 2500
# dpa_v42_vs_ASCAD_15_06_2023_22_43_21_4159611 5000
# dpa_v42_vs_ASCAD_15_06_2023_23_21_47_4249103 10000

reference_dataset_name = "dpa_v42"
target_dataset_name = "ASCAD"
n_prof = 50000
n_val = 0
n_attack = 10000
target_byte = 2
leakage_model = "ID"
dataset_root_path = "/tudelft.net/staff-umbrella/dlsca/Guilherme"
results_root_path = "/tudelft.net/staff-umbrella/dlsca/Guilherme/paper_9_gan_results"
str_datatime = "15_06_2023_23_21_47_4249103"
target_dim = 10000
reference_dim = 30000
features = 100

args = np.load(f"{results_root_path}/{reference_dataset_name}_vs_{target_dataset_name}_{str_datatime}/args.npz", allow_pickle=True)
print(args["args"])


dataset_file = get_dataset_filepath(dataset_root_path, target_dataset_name, target_dim, leakage_model)

loader = dataset_loaders[target_dataset_name]
dataset_target = loader(n_prof, n_val, n_attack, target_byte, leakage_model, dataset_file, number_of_samples=target_dim)
scaler = StandardScaler()
dataset_target.x_profiling = scaler.fit_transform(dataset_target.x_profiling)
dataset_target.x_attack = scaler.transform(dataset_target.x_attack)

features_target_profiling, features_target_attack = get_features(dataset_target, target_byte, features)

results_folder = f"{results_root_path}/{reference_dataset_name}_vs_{target_dataset_name}_{str_datatime}"
generator_trained_model = f"{results_folder}/generator_{target_dim}_{reference_dim}_epoch_199.h5"
generator = define_generator(target_dim, features)
generator.load_weights(generator_trained_model)

ge_cgan_sca_profiling = []
nt_cgan_sca_profiling = []
pi_cgan_sca_profiling = []

ge_whiteboxmlp_profiling = []
nt_whiteboxmlp_profiling = []
pi_whiteboxmlp_profiling = []

ge_blackboxmlp_profiling = []
nt_blackboxmlp_profiling = []
pi_blackboxmlp_profiling = []

ge_whiteboxta_profiling = []
nt_whiteboxta_profiling = []
pi_whiteboxta_profiling = []

profiling_steps = 20
step = int((n_prof - 10000) / profiling_steps)  # we will run from 10k:n_prof profiling traces

for attack_type in ["CGAN-SCA MLP", "White-Box MLP", "Black-Box MLP", "White-Box TA"]:

    if attack_type == "White-Box TA":
        features_profiling, features_attack = get_features(dataset_target, target_byte, n_poi=10)
        for n_p in range(10000, n_prof + step, step):
            ge, nt, pi, ge_vector = ta(features_profiling[:n_p], features_attack, dataset_target.classes,
                                       dataset_target.profiling_labels[:n_p], dataset_target.attack_labels,
                                       dataset_target.labels_key_hypothesis_attack, dataset_target.correct_key_attack)
            ge_whiteboxta_profiling.append(ge)
            nt_whiteboxta_profiling.append(nt)
            pi_whiteboxta_profiling.append(pi)
    else:
        if attack_type == "CGAN-SCA MLP":
            attack_traces = np.array(generator.predict([dataset_target.x_attack]))
            profiling_traces = np.array(generator.predict([dataset_target.x_profiling]))
        if attack_type == "White-Box MLP":
            attack_traces = np.array(features_target_attack)
            profiling_traces = np.array(features_target_profiling)
        if attack_type == "Black-Box MLP":
            attack_traces = dataset_target.x_attack
            profiling_traces = dataset_target.x_profiling
        else:
            attack_traces = np.array(features_target_attack)
            profiling_traces = np.array(features_target_profiling)

        for n_p in range(10000, n_prof + step, step):
            model = mlp(dataset_target.classes, profiling_traces.shape[1])
            model.fit(
                x=profiling_traces[:n_p],
                y=to_categorical(dataset_target.profiling_labels[:n_p], num_classes=dataset_target.classes),
                batch_size=400,
                verbose=2,
                epochs=50,
                shuffle=True,
                validation_data=(
                    attack_traces, to_categorical(dataset_target.attack_labels, num_classes=dataset_target.classes)),
                callbacks=[])

            """ Predict the trained MLP with target/attack measurements """
            predictions = model.predict(attack_traces)
            """ Check if we are able to recover the key from the target/attack measurements """
            ge, ge_vector, nt = guessing_entropy(predictions, dataset_target.labels_key_hypothesis_attack,
                                                 dataset_target.correct_key_attack, 2000)
            pi = information(predictions, dataset_target.attack_labels, dataset_target.classes)

            if attack_type == "CGAN-SCA MLP":
                ge_cgan_sca_profiling.append(ge)
                nt_cgan_sca_profiling.append(nt)
                pi_cgan_sca_profiling.append(pi)
            if attack_type == "White-Box MLP":
                ge_whiteboxmlp_profiling.append(ge)
                nt_whiteboxmlp_profiling.append(nt)
                pi_whiteboxmlp_profiling.append(pi)
            if attack_type == "Black-Box MLP":
                ge_blackboxmlp_profiling.append(ge)
                nt_blackboxmlp_profiling.append(nt)
                pi_blackboxmlp_profiling.append(pi)

x_axis_values = list(range(10000, n_prof + step, step))

plt.plot(x_axis_values, ge_cgan_sca_profiling, label="CGAN-SCA MLP")
plt.plot(x_axis_values, ge_whiteboxmlp_profiling, label="White-Box MLP")
plt.plot(x_axis_values, ge_blackboxmlp_profiling, label="Black-Box MLP")
plt.plot(x_axis_values, ge_whiteboxta_profiling, label="White-Box TA (+LDA)")
plt.legend()
plt.xlabel("Profiling Traces")
plt.ylabel("Guessing Entropy")
plt.savefig(f"{results_folder}/ge_profiling.png")
plt.close()
np.savez(f"{results_folder}/ge_profiling.npz",
         x_axis_values=x_axis_values,
         ge_cgan_sca_profiling=ge_cgan_sca_profiling,
         ge_whiteboxmlp_profiling=ge_whiteboxmlp_profiling,
         ge_blackboxmlp_profiling=ge_blackboxmlp_profiling,
         ge_whiteboxta_profiling=ge_whiteboxta_profiling)

plt.plot(x_axis_values, nt_cgan_sca_profiling, label="CGAN-SCA MLP")
plt.plot(x_axis_values, nt_whiteboxmlp_profiling, label="White-Box MLP")
plt.plot(x_axis_values, nt_blackboxmlp_profiling, label="Black-Box MLP")
plt.plot(x_axis_values, nt_whiteboxta_profiling, label="White-Box TA (+LDA)")
plt.legend()
plt.xlabel("Profiling Traces")
plt.ylabel("Number of Traces for GE=1")
plt.yscale('log')
plt.savefig(f"{results_folder}/nt_profiling.png")
plt.close()
np.savez(f"{results_folder}/nt_profiling.npz",
         x_axis_values=x_axis_values,
         nt_cgan_sca_profiling=nt_cgan_sca_profiling,
         nt_whiteboxmlp_profiling=nt_whiteboxmlp_profiling,
         nt_blackboxmlp_profiling=nt_blackboxmlp_profiling,
         nt_whiteboxta_profiling=nt_whiteboxta_profiling)

plt.plot(x_axis_values, pi_cgan_sca_profiling, label="CGAN-SCA MLP")
plt.plot(x_axis_values, pi_whiteboxmlp_profiling, label="White-Box MLP")
plt.plot(x_axis_values, pi_blackboxmlp_profiling, label="Black-Box MLP")
plt.plot(x_axis_values, pi_whiteboxta_profiling, label="White-Box TA (+LDA)")
plt.legend()
plt.xlabel("Profiling Traces")
plt.ylabel("Perceived Information")
plt.savefig(f"{results_folder}/pi_profiling.png")
plt.close()
np.savez(f"{results_folder}/pi_profiling.npz",
         x_axis_values=x_axis_values,
         pi_cgan_sca_profiling=pi_cgan_sca_profiling,
         pi_whiteboxmlp_profiling=pi_whiteboxmlp_profiling,
         pi_blackboxmlp_profiling=pi_blackboxmlp_profiling,
         pi_whiteboxta_profiling=pi_whiteboxta_profiling)
