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


set -e
module purge
module load baskerville

# 运行 Python 命令
source /bask/homes/s/sxz363/wangsu-transfer/miniconda3/etc/profile.d/conda.sh
conda init
conda activate "/bask/homes/s/sxz363/wangsu-transfer/miniconda3/envs/unitime"
conda info --envs
nohup python run.py --gpu 0 --training_list execute_list/train_all.csv --max_token_num 27 >>'log1.txt'