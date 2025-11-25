#!/bin/bash

#SBATCH --time=4:00:00               
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1                 # any GPU
#SBATCH --mem-per-cpu=8G             
#SBATCH --partition=gpu.4h           
#SBATCH --job-name=semlaflow-test
#SBATCH --output=semlaflow-%j.out
#SBATCH --error=semlaflow-%j.err
#SBATCH --mail-user=jobaumann@ethz.ch
#SBATCH --mail-type=END,FAIL
#SBATCH --account=es_jorner

module load gcc/8.2.0
module load cuda/12.1
module load eth_proxy

source ~/.bashrc
mamba activate semlaflow

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
nvidia-smi

cd /cluster/project/jorner/jobaumann/semla-flow

python -m semlaflow.train \
    --data_path ../datasets/qmugs/smol \
    --dataset geom-drugs \
    --batch_cost 128 \
    --epochs 10

echo "Job finished at: $(date)"
