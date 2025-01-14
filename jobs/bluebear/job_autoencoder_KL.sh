#!/bin/bash
#SBATCH --qos=bbgpu
#SBATCH --account=chenhp-dpmodel
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=300G
module purge; module load bluebear
module load bear-apps/2022b/live
module load Miniforge3/24.1.2-0


script_name="acdc_autoencoder_test_kl"

source activate /rds/projects/c/chenhp-dpmodel/conda_envs
cd /rds/projects/c/chenhp-dpmodel/2024-miccai-dgm-daum

export PYTHONPATH=$PYTHONPATH:$(pwd)
python core/Main.py --config_path projects/cardiac_autoencoder/configKL.yaml >output/$script_name.out
