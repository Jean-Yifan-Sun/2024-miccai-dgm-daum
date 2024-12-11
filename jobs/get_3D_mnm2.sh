#!/bin/bash
#SBATCH --qos=bbdefault
#SBATCH --account=chenhp-dpmodel
#SBATCH --time=10:00:00
#SBATCH --mem=200G
module purge; module load bluebear
module load bear-apps/2022b/live
module load Miniforge3/24.1.2-0


script_name="acdc_autoencoder_test"

source activate /rds/projects/c/chenhp-dpmodel/conda_envs
cd /rds/projects/c/chenhp-dpmodel/2024-miccai-dgm-daum

export PYTHONPATH=$PYTHONPATH:$(pwd)

python /rds/projects/c/chenhp-dpmodel/2024-miccai-dgm-daum/data/MnM2/prepare_mnm2_3D.py