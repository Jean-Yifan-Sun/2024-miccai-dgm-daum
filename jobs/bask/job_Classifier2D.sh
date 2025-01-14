#!/bin/bash
#SBATCH --account=qingjiem-heart-tte
#SBATCH --qos=bham
#SBATCH --time=10:00:00
#SBATCH --nodes 1
#SBATCH --gres gpu:1
#SBATCH --gpus-per-task 1
#SBATCH --tasks-per-node 1
#SBATCH --constraint=a100_40
#SBATCH --mem=64G  # 请求内存

script_name="acdc_classifier"
set -e
module purge
module load baskerville

# 运行 Python 命令
source /bask/projects/q/qingjiem-heart-tte/yifansun/conda/miniconda/etc/profile.d/conda.sh
conda init
conda activate miccai24
conda info --envs
cd /bask/projects/q/qingjiem-heart-tte/yifansun/project/2024-miccai-dgm-daum
export PYTHONPATH=$PYTHONPATH:$(pwd)
nohup python core/Main.py --config_path projects/cardiac_classifier/config2d.yaml >output/$script_name.out