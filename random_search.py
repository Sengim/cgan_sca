import argparse
from RandomCGANSCA import *
from datetime import datetime
import os

# dataset_root_path = "/tudelft.net/staff-umbrella/dlsca/Guilherme"
# results_root_path = "/tudelft.net/staff-umbrella/dlsca/Guilherme/paper_9_gan_results"
# features_root_path = "/tudelft.net/staff-umbrella/dlsca/Guilherme/cgan_reference_features"

# dataset_root_path = "D:/traces"
# results_root_path = "D:/postdoc/paper_cgan_features/results"
# results_root_path = "D:/postdoc/paper_cgan_features/cgan_reference_features"

dataset_root_path = "C:/Users/Sengim/Datasets"
results_root_path = "C:/Users/Sengim/Datasets/paper_9_gan_results"
features_root_path = "C:/Users/Sengim/Datasets/paper_9_gan_features"

# dataset_root_path = "/data1/karayalcins/datasets"
# results_root_path = "/data1/karayalcins/datasets/paper_9_gan_results"
# features_root_path = "/data1/karayalcins/datasets/paper_9_gan_features"


def get_arguments():
    parser = argparse.ArgumentParser(add_help=False)

    """ root path for datasets """
    parser.add_argument("-dataset_root_path", "--dataset_root_path", default=dataset_root_path)

    """ root path for results """
    parser.add_argument("-results_root_path", "--results_root_path", default=results_root_path)

    """ root path for reference features """
    parser.add_argument("-features_root_path", "--features_root_path", default=features_root_path)

    """ dataset_reference: name of reference dataset (possible values: ascad-variable, ASCAD, dpa_v42, eshard, aes_hd_mm) """
    parser.add_argument("-dataset_reference", "--dataset_reference", default="ascad-variable")

    """ dataset_reference_dim: number of features (samples) in reference dataset """
    parser.add_argument("-dataset_reference_dim", "--dataset_reference_dim", default=1400)

    """ dataset_target: name of target dataset (possible values: ascad-variable, ASCAD, dpa_v42, eshard, aes_hd_mm) """
    parser.add_argument("-dataset_target", "--dataset_target", default="ASCAD")

    """ dataset_target_dim: number of features (samples) in target dataset """
    parser.add_argument("-dataset_target_dim", "--dataset_target_dim", default=700)

    """ features: number of features extracted from reference dataset and output dimension of generator """
    parser.add_argument("-features", "--features", default=100)

    """ n_profiling_reference: number of profiling traces from the reference dataset (always profiling set from .h5 datasets) """
    parser.add_argument("-n_profiling_reference", "--n_profiling_reference", default=200000)

    """ n_attack_reference: number of profiling traces from the reference dataset (always profiling set from .h5 datasets) """
    parser.add_argument("-n_attack_reference", "--n_attack_reference", default=5000)

    """ n_profiling_target: number of profiling traces from the target dataset """
    parser.add_argument("-n_profiling_target", "--n_profiling_target", default=50000)

    """ n_validation_target: number of validation traces from the target dataset """
    parser.add_argument("-n_validation_target", "--n_validation_target", default=0)

    """ n_attack_target: number of attack traces from the target dataset """
    parser.add_argument("-n_attack_target", "--n_attack_target", default=10000)

    """ n_attack_ge: number of attack traces for guessing entropy calculation """
    parser.add_argument("-n_attack_ge", "--n_attack_ge", default=2000)

    """ target_byte_reference: key byte index in the reference dataset """
    parser.add_argument("-target_byte_reference", "--target_byte_reference", default=2)

    """ target_byte_target: key byte index in the target dataset """
    parser.add_argument("-target_byte_target", "--target_byte_target", default=2)

    """ leakage_model: leakage model type (ID or HW) """
    parser.add_argument("-leakage_model", "--leakage_model", default="ID")

    """ epochs: number of training epochs for CGAN """
    parser.add_argument("-epochs", "--epochs", default=200)

    """ batch_size: batch size for training the CGAN """
    parser.add_argument("-batch_size", "--batch_size", default=400)

    """ std_gaussian_noise_reference: standard deviation for Gaussian noise artificially added to reference dataset (default mean is 0) """
    parser.add_argument("-std_gaussian_noise_reference", "--std_gaussian_noise_reference", default=0.0)

    """ std_gaussian_noise_target: standard deviation for Gaussian noise artificially added to target dataset (default mean is 0) """
    parser.add_argument("-std_gaussian_noise_target", "--std_gaussian_noise_target", default=0.0)

    """ num_models: number of random models to train (default is 10) """
    parser.add_argument("-num_models", "--num_models", default=10)

    return parser.parse_args()


if __name__ == "__main__":
    arg_list = get_arguments()

    arguments = {
        "dataset_root_path": arg_list.dataset_root_path,
        "results_root_path": arg_list.results_root_path,
        "features_root_path": arg_list.features_root_path,
        "dataset_reference": arg_list.dataset_reference,
        "dataset_reference_dim": int(arg_list.dataset_reference_dim),
        "dataset_target": arg_list.dataset_target,
        "dataset_target_dim": int(arg_list.dataset_target_dim),
        "n_profiling_reference": int(arg_list.n_profiling_reference),
        "n_attack_reference": int(arg_list.n_attack_reference),
        "n_profiling_target": int(arg_list.n_profiling_target),
        "n_validation_target": int(arg_list.n_validation_target),
        "n_attack_target": int(arg_list.n_attack_target),
        "n_attack_ge": int(arg_list.n_attack_ge),
        "target_byte_reference": int(arg_list.target_byte_reference),
        "target_byte_target": int(arg_list.target_byte_target),
        "features": int(arg_list.features),
        "leakage_model": arg_list.leakage_model,
        "epochs": int(arg_list.epochs),
        "batch_size": int(arg_list.batch_size),
        "std_gaussian_noise_reference": float(arg_list.std_gaussian_noise_reference),
        "std_gaussian_noise_target": float(arg_list.std_gaussian_noise_target),
        "num_models": int(arg_list.num_models),
    }
    now = datetime.now()
    results_collection= f"{results_root_path}/random_search_{now.strftime('%d_%m_%Y_%H')}"
    if not os.path.exists(results_collection):
        os.mkdir(results_collection)
    cgan = RandomCGANSCA(args=arguments, main_path=results_collection)
    best_nt_result = cgan.train_cgan()
    best_model_folder = cgan.dir_results
    print(best_model_folder)
    for i in range(arguments["num_models"]-1):
        cgan = RandomCGANSCA(args=arguments, main_path= results_collection)
        nt_result = cgan.train_cgan()
        if nt_result < best_nt_result:
            best_model_folder = cgan.dir_results

    print(f"The results from the best model can be found in {cgan.dir_results}")

