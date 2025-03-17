from src.datasets.load_ascadr import *
from src.datasets.load_ascadf import *
from src.datasets.load_dpav42 import *
from src.datasets.load_aes_sim_reference import *
from src.datasets.load_eshard import *
from src.datasets.load_spook_sw3 import *
from src.datasets.load_aes_hd_mm import *
from src.datasets.simulate_higher_order import *
from src.datasets.load_ascadv2 import *
from src.datasets.paths import *
import tensorflow as tf
from os.path import exists
from utils import *


class PrepareDatasets:

    def __init__(self, args):
        self.features_dim = args["features"]
        self.target_byte_reference = args["target_byte_reference"]
        self.target_byte_target = args["target_byte_target"]
        self.path = args["dataset_root_path"]

        self.traces_reference_dim = args["dataset_reference_dim"]
        self.traces_target_dim = args["dataset_target_dim"]

        self.dataset_reference = self.load_dataset(args, args["dataset_reference"], self.target_byte_reference, self.traces_reference_dim,
                                                   args["n_profiling_reference"], 0, args["n_attack_reference"], reference=True)
        self.dataset_target = self.load_dataset(args, args["dataset_target"], self.target_byte_target, self.traces_target_dim,
                                                args["n_profiling_target"], args["n_validation_target"], args["n_attack_target"])

        self.add_gaussian_noise(args)

        self.dataset_reference.x_profiling, self.dataset_reference.x_attack = self.scale_dataset(self.dataset_reference.x_profiling,
                                                                                                 self.dataset_reference.x_attack,
                                                                                                 StandardScaler())
        if not self.dataset_target.name =="spook_sw3":        
            self.dataset_target.x_profiling, self.dataset_target.x_attack = self.scale_dataset(self.dataset_target.x_profiling,
                                                                                                 self.dataset_target.x_attack,
                                                                                                 StandardScaler())
        self.features_reference_profiling, self.features_reference_attack = self.dataset_reference.x_profiling, self.dataset_reference.x_attack
        if self.dataset_target.name == "ascadv2":
            root_path = args["dataset_root_path"]
            dif_folder ="diffusion_ascadv2_22_11_2023_07_22_04_1057297"
            dif_mod = tf.keras.models.load_model(f"{root_path}/paper_10_diffussion_results/{dif_folder}/trained_model.keras")
            dif_mod = tf.keras.models.Model(inputs=dif_mod.inputs, outputs=dif_mod.layers[-3].output)

            self.dataset_target.x_profiling = dif_mod.predict([self.dataset_target.x_profiling, np.ones(self.dataset_target.x_profiling.shape[0])*0])
            self.dataset_target.x_attack = dif_mod.predict([self.dataset_target.x_attack, np.ones(self.dataset_target.x_attack.shape[0])*0])
            args["dataset_target_dim"] = self.dataset_target.x_profiling.shape[1]
            self.dataset_target.x_profiling, self.dataset_target.x_attack = self.scale_dataset(self.dataset_target.x_profiling,
                                                                                                 self.dataset_target.x_attack,
                                                                                                 StandardScaler())

        """ the following is used only for verification, not in the CGAN training """
        if self.dataset_target.name == "ches_ctf" or self.dataset_target.name=="ascadv2":
            
            pass
        else:
            self.features_target_profiling, self.features_target_attack = get_features(self.dataset_target,
                                                                                   self.target_byte_target,
                                                                                   n_poi=self.features_dim)

    def load_dataset(self, args, identifier, target_byte, traces_dim, n_prof, n_val, n_attack, reference=False):

        implement_reference_feature_selection = False
        reference_features_shortcut = ""
        num_features = args["features"]
        dataset_file = get_dataset_filepath(args["dataset_root_path"], identifier, traces_dim, args["leakage_model"])
        if reference and (not identifier =="simulate"):
            """ If features were already computed for this dataset, target key byte, 
            and leakage model, there is no need to compute it again"""
            reference_features_shortcut = f'{args["features_root_path"]}/selected_{args["features"]}_features_snr_{args["dataset_reference"]}_{self.traces_reference_dim}_target_byte_{self.target_byte_reference}_{args["feature_select"]}.h5'
            if exists(reference_features_shortcut):
                print("Reference features already created.")
                dataset_file = reference_features_shortcut
                traces_dim = num_features
            else:
                implement_reference_feature_selection = True

        dataset = None
        if identifier == "ascad-variable":
            dataset = ReadASCADr(n_prof, n_val, n_attack, target_byte, args["leakage_model"], dataset_file, number_of_samples=traces_dim)
        if identifier == "simulate":
            order = 2 if args["dataset_target"] in ["spook_sw3", "ascadv2"] else 1
            informative = args["features"]//(order+1) if args["feature_select"]== "snr" else 100
            #informative = informative if not args["dataset_target"]=="ascadv2" else 50
            total =  args["features"] if args["feature_select"]== "snr" else (100 * (order+1))
            print(informative, total)
            dataset = SimulateHigherOrder(args, order, n_prof, n_attack, informative, total)
            if args["feature_select"]== "pca":
                dataset.x_profiling, dataset.x_attack = get_pca_features(dataset, 0, num_features)
            elif args["feature_select"]== "lda":
                dataset.x_profiling, dataset.x_attack = get_lda_features(dataset, 0, num_features)
        if identifier == "ASCAD":
            dataset = ReadASCADf(n_prof, n_val, n_attack, target_byte, args["leakage_model"], dataset_file, number_of_samples=traces_dim)
        if identifier == "eshard":
            dataset = ReadEshard(n_prof, n_val, n_attack, target_byte, args["leakage_model"], dataset_file, number_of_samples=traces_dim)
        if identifier == "dpa_v42":
            dataset = ReadDPAV42(n_prof, n_val, n_attack, target_byte, args["leakage_model"], dataset_file, number_of_samples=traces_dim)
        if identifier == "aes_sim_reference":
            dataset = ReadAESSimReference(n_prof, n_val, n_attack, target_byte, args["leakage_model"], dataset_file,
                                          number_of_samples=traces_dim)
        if identifier == "aes_hd_mm":
            dataset = ReadAESHDMM(n_prof, n_val, n_attack, target_byte, args["leakage_model"], dataset_file, number_of_samples=traces_dim)
        if identifier == "spook_sw3":
                 return ReadSpookSW3(n_prof, n_val, n_attack, target_byte, args["leakage_model"], dataset_file,
                                   number_of_samples=traces_dim)
        if identifier == "ches_ctf":
                 return ReadCHESCTF(n_prof, n_val, n_attack, target_byte, args["leakage_model"], dataset_file,
                                   number_of_samples=traces_dim)
        if identifier == "ascadv2":
                 return ReadASCADv2(n_prof, n_val, n_attack, target_byte, args["leakage_model"], dataset_file,
                                   number_of_samples=traces_dim)

        if implement_reference_feature_selection:
            self.generate_features_h5(dataset, target_byte, reference_features_shortcut, args["feature_select"])
            dataset_file = reference_features_shortcut
            traces_dim = num_features

            if identifier == "ascad-variable":
                return ReadASCADr(n_prof, n_val, n_attack, target_byte, args["leakage_model"], dataset_file,
                                  number_of_samples=traces_dim)
            if identifier == "ASCAD":
                return ReadASCADf(n_prof, n_val, n_attack, target_byte, args["leakage_model"], dataset_file,
                                  number_of_samples=traces_dim)
            if identifier == "eshard":
                return ReadEshard(n_prof, n_val, n_attack, target_byte, args["leakage_model"], dataset_file,
                                  number_of_samples=traces_dim)
            if identifier == "dpa_v42":
                return ReadDPAV42(n_prof, n_val, n_attack, target_byte, args["leakage_model"], dataset_file,
                                  number_of_samples=traces_dim)
            if identifier == "aes_sim_reference":
                return ReadAESSimReference(n_prof, n_val, n_attack, target_byte, args["leakage_model"], dataset_file,
                                           number_of_samples=traces_dim)
            if identifier == "aes_hd_mm":
                return ReadAESHDMM(n_prof, n_val, n_attack, target_byte, args["leakage_model"], dataset_file,
                                   number_of_samples=traces_dim)
            
            if identifier == "spook_sw3":
                 return ReadSpookSW3(n_prof, n_val, n_attack, target_byte, args["leakage_model"], dataset_file,
                                   number_of_samples=traces_dim)

        else:
            
            return dataset

    def scale_dataset(self, prof_set, attack_set, scaler):
        prof_new = scaler.fit_transform(prof_set)
        if attack_set is not None:
            attack_new = scaler.transform(attack_set)
        else:
            attack_new = None
        return prof_new, attack_new

    def add_gaussian_noise(self, args):
        if args["std_gaussian_noise_reference"] > 0.0:
            print(f"adding gaussian noise of {args['std_gaussian_noise_reference']}")
            noise = np.random.normal(0, args["std_gaussian_noise_reference"], np.shape(self.dataset_reference.x_profiling))
            self.dataset_reference.x_profiling = np.add(self.dataset_reference.x_profiling, noise)
            noise = np.random.normal(0, args["std_gaussian_noise_reference"], np.shape(self.dataset_reference.x_attack))
            self.dataset_reference.x_attack = np.add(self.dataset_reference.x_attack, noise)

        if args["std_gaussian_noise_target"] > 0.0:
            print(f"adding gaussian noise of {args['std_gaussian_noise_target']}")
            noise = np.random.normal(0, args["std_gaussian_noise_target"], np.shape(self.dataset_target.x_profiling))
            self.dataset_target.x_profiling = np.add(self.dataset_target.x_profiling, noise)
            noise = np.random.normal(0, args["std_gaussian_noise_target"], np.shape(self.dataset_target.x_attack))
            self.dataset_target.x_attack = np.add(self.dataset_target.x_attack, noise)

    def generate_features_h5(self, dataset, target_byte, save_file_path, feature_select):
        profiling_traces_rpoi, attack_traces_rpoi = None, None
        """spook features generation doesnt work"""
        if feature_select== "snr":
            profiling_traces_rpoi, attack_traces_rpoi = get_features(dataset, target_byte, self.features_dim)
        elif feature_select == "pca":
            profiling_traces_rpoi, attack_traces_rpoi = get_pca_features(dataset, target_byte, self.features_dim)
        elif feature_select == "lda":
        #profiling_traces_rpoi, attack_traces_rpoi = get_features_bit(dataset, target_byte, self.features_dim)
            profiling_traces_rpoi, attack_traces_rpoi = get_lda_features(dataset, target_byte, self.features_dim)
        #profiling_traces_rpoi, attack_traces_rpoi = get_pca_features(dataset, target_byte, self.features_dim)
        
        out_file = h5py.File(save_file_path, 'w')

        profiling_index = [n for n in range(dataset.n_profiling)]
        attack_index = [n for n in range(dataset.n_attack)]

        profiling_traces_group = out_file.create_group("Profiling_traces")
        attack_traces_group = out_file.create_group("Attack_traces")

        profiling_traces_group.create_dataset(name="traces", data=profiling_traces_rpoi, dtype=profiling_traces_rpoi.dtype)
        attack_traces_group.create_dataset(name="traces", data=attack_traces_rpoi, dtype=attack_traces_rpoi.dtype)

        metadata_type_profiling = np.dtype([("plaintext", dataset.profiling_plaintexts.dtype, (len(dataset.profiling_plaintexts[0]),)),
                                            ("key", dataset.profiling_keys.dtype, (len(dataset.profiling_keys[0]),)),
                                            ("masks", dataset.profiling_masks.dtype, (len(dataset.profiling_masks[0]),))
                                            ])
        metadata_type_attack = np.dtype([("plaintext", dataset.attack_plaintexts.dtype, (len(dataset.attack_plaintexts[0]),)),
                                         ("key", dataset.attack_keys.dtype, (len(dataset.attack_keys[0]),)),
                                         ("masks", dataset.attack_masks.dtype, (len(dataset.attack_masks[0]),))
                                         ])

        profiling_metadata = np.array(
            [(dataset.profiling_plaintexts[n], dataset.profiling_keys[n], dataset.profiling_masks[n]) for n in profiling_index],
            dtype=metadata_type_profiling)
        profiling_traces_group.create_dataset("metadata", data=profiling_metadata, dtype=metadata_type_profiling)

        attack_metadata = np.array([(dataset.attack_plaintexts[n], dataset.attack_keys[n], dataset.attack_masks[n]) for n in attack_index],
                                   dtype=metadata_type_attack)
        attack_traces_group.create_dataset("metadata", data=attack_metadata, dtype=metadata_type_attack)

        out_file.flush()
        out_file.close()
