#!/bin/bash -l

#####################
# job-array example #
#####################

#SBATCH --job-name=cgan_sca_p_a
#SBATCH --qos=short
#SBATCH --time=04:00:00

# 500MB memory per core
# this is a hard limit
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=jobspernode:four:1
#SBATCH --mem-per-cpu=12000MB
#SBATCH --gres=gpu:turing:1

# you may not place bash commands before the last SBATCH directive
module use /opt/insy/modulefiles
module load cuda/11.0 cudnn/11.0-8.0.3.33
module list

cd /home/nfs/gperin/AISY_Framework
source env/bin/activate
cd /home/nfs/gperin/paper_9_gan/cgan_sca_release


srun python run_profiling_and_attack.py
exit 0


