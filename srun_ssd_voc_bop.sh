#!/bin/bash
#SBATCH --job-name=bnnvocBOP
#SBATCH --output=slurm_bnn_ssd_voc_bop_%A.out
#SBATCH --error=slurm_bnn_ssd_voc_bop_%A.err
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem-per-cpu=5G
#SBATCH --cpus-per-task=32
#SBATCH --partition=research

nvidia-smi
module purge
module load python/miniconda3/miniconda3
eval "$(conda shell.bash hook)"
export HTTP_PROXY=http://proxytc.vingroup.net:9090/
export HTTPS_PROXY=http://proxytc.vingroup.net:9090/
export http_proxy=http://proxytc.vingroup.net:9090/
export https_proxy=http://proxytc.vingroup.net:9090/
rm -rf /home/tampm2/.conda/envs/bnn_ssd_voc_bop
conda create --name bnn_ssd_voc_bop python=3.7 --force
conda activate bnn_ssd_voc_bop
pip install -r requirements.txt
pip install wandb
wandb login c5a6f4c212b00734d9517784e3e892155a301de0
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install opencv-python
pip install pycocotools
#tar -xvf /vinai-public-dataset/VOC2012/VOCtrainval_11-May-2012.tar -C data/
#tar -xvf /vinai-public-dataset/VOC2007/VOCtrainval_06-Nov-2007.tar -C data/
#tar -xvf /vinai-public-dataset/VOC2007/VOCtest_06-Nov-2007.tar -C data/
python train_bidet_ssd.py --data_root /lustre/scratch/client/vinai/users/tampm2/ssd.pytorch/data/VOCdevkit/ --dataset VOC --num_workers 32 --batch_size 32 --opt bop  --add_name_wandb BOP
#python train_bidet_ssd.py --data_root ../arp_data/VOC/VOCdevkit/ --dataset VOC --num_workers 32 --batch_size 64 --opt bop  --add_name_wandb BOP
