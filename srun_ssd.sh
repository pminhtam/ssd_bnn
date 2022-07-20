#!/bin/bash
#SBATCH --job-name=bnn_coco
#SBATCH --output=slurm_bnn_ssd_coco_%A.out
#SBATCH --error=slurm_bnn_ssd_coco_%A.err
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=70
#SBATCH --partition=research

nvidia-smi
module purge
module load python/miniconda3/miniconda3
eval "$(conda shell.bash hook)"
export HTTP_PROXY=http://proxytc.vingroup.net:9090/
export HTTPS_PROXY=http://proxytc.vingroup.net:9090/
export http_proxy=http://proxytc.vingroup.net:9090/
export https_proxy=http://proxytc.vingroup.net:9090/
rm -rf /home/tampm2/.conda/envs/bnn_ssd_coco
conda create --name bnn_ssd_coco python=3.7 --force
conda activate bnn_ssd_coco
pip install -r requirements.txt
pip install wandb
wandb login c5a6f4c212b00734d9517784e3e892155a301de0
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install opencv-python
pip install pycocotools
#mkdir -p data/coco/images/
#unzip /vinai-public-dataset/COCO/train2014.zip -d data/coco/images/
#unzip /vinai-public-dataset/COCO/val2014.zip -d data/coco/images/
#unzip /vinai-public-dataset/COCO/annotations_trainval2014.zip -d data/coco/
#unzip /vinai-public-dataset/COCO/instances_minival2014.json.zip -d data/coco/annotations/
python train_bidet_ssd.py --data_root /lustre/scratch/client/vinai/users/tampm2/ssd.pytorch/data/coco  --dataset COCO --num_workers 64 --batch_size 64
