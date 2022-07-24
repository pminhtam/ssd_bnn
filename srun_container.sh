#!/bin/bash
#SBATCH --job-name=container-job
#SBATCH --output=slurm_container_%A.out
#SBATCH --error=slurm_container_%A.err
#SBATCH --gpus=0
#SBATCH --mem=8000MB
#SBATCH --cpus-per-task=10
#SBATCH --partition=research

srun --partition=research --container-image nvcr.io/nvidia/pytorch:20.03-py3  --no-container-entrypoint \
--container-mounts=/lustre/scratch/client/vinai/users/tampm2/ssd_bnn:~/ \
pwd \
ls \
nvidia-smi \
df -h \
ls -la /
