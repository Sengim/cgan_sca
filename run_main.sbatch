#!/bin/bash -l

#####################
# job-array example #
#####################

#SBATCH --job-name=cgan_features
#SBATCH --qos=medium
#SBATCH --time=08:00:00

# 500MB memory per core
# this is a hard limit
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=jobspernode:four:1
#SBATCH --mem-per-cpu=125000MB
#SBATCH --gres=gpu:turing:1

# you may not place bash commands before the last SBATCH directive
module use /opt/insy/modulefiles
module load cuda/11.0 cudnn/11.0-8.0.3.33
module list

cd /home/nfs/gperin/AISY_Framework
source env/bin/activate
cd /home/nfs/gperin/paper_9_gan/cgan_sca_release


# default values
dataset_reference=ascad-variable
dataset_reference_dim=1400
n_profiling_reference=200000
dataset_target=ASCAD
dataset_target_dim=700
n_profiling_target=50000
target_byte_reference=2
target_byte_target=2
leakage_model=ID
features=100
epochs=200
std_gaussian_noise_reference=0
std_gaussian_noise_target=0

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        --dataset_reference)
        dataset_reference="$2"
        shift
        ;;
        --dataset_reference_dim)
        dataset_reference_dim="$2"
        shift
        ;;
        --n_profiling_reference)
        n_profiling_reference="$2"
        shift
        ;;
        --dataset_target)
        dataset_target="$2"
        shift
        ;;
        --dataset_target_dim)
        dataset_target_dim="$2"
        shift
        ;;
        --n_profiling_target)
        n_profiling_target="$2"
        shift
        ;;
        --target_byte_reference)
        target_byte_reference="$2"
        shift
        ;;
        --target_byte_target)
        target_byte_target="$2"
        shift
        ;;
        --leakage_model)
        leakage_model="$2"
        shift
        ;;
        --features)
        features="$2"
        shift
        ;;
        --epochs)
        epochs="$2"
        shift
        ;;
        --std_gaussian_noise_reference)
        std_gaussian_noise_reference="$2"
        shift
        ;;
        --std_gaussian_noise_target)
        std_gaussian_noise_target="$2"
        shift
        ;;
        *)
        # unknown option
        ;;
    esac
    shift
done

# traces
echo "dataset_reference is set to: $dataset_reference"
echo "dataset_reference_dim is set to: $dataset_reference_dim"
echo "n_profiling_reference is set to: $n_profiling_reference"
echo "dataset_target is set to: $dataset_target"
echo "dataset_target_dim is set to: $dataset_target_dim"
echo "n_profiling_target is set to: $n_profiling_target"
echo "target_byte_reference is set to: $target_byte_reference"
echo "target_byte_target is set to: $target_byte_target"
echo "leakage_model is set to: $leakage_model"
echo "features is set to: $features"
echo "epochs is set to: $epochs"
echo "std_gaussian_noise_reference is set to: $std_gaussian_noise_reference"
echo "std_gaussian_noise_target is set to: $std_gaussian_noise_target"

srun python main.py --dataset_reference $dataset_reference --dataset_reference_dim $dataset_reference_dim --n_profiling_reference $n_profiling_reference --dataset_target $dataset_target --dataset_target_dim $dataset_target_dim --n_profiling_target $n_profiling_target --target_byte_reference $target_byte_reference --target_byte_target $target_byte_target --features $features --epochs $epochs --leakage_model $leakage_model --std_gaussian_noise_reference $std_gaussian_noise_reference --std_gaussian_noise_target $std_gaussian_noise_target
exit 0


