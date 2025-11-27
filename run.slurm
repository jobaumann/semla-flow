#!/bin/bash

#SBATCH --time=4:00:00               
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1            # Changed from 8 to 1
#SBATCH --gpus=rtx_4090:1                 
#SBATCH --mem-per-cpu=32G            # Increased since we have fewer CPUs
#SBATCH --partition=gpu.4h           
#SBATCH --job-name=semlaflow-test
#SBATCH --output=semlaflow-%j.out
#SBATCH --error=semlaflow-%j.err
#SBATCH --mail-user=jobaumann@ethz.ch
#SBATCH --mail-type=END,FAIL

module load eth_proxy

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
nvidia-smi

cd /cluster/project/jorner/jobaumann/semla-flow

mamba run -n semlaflow python -m semlaflow.train \
    --data_path ../datasets/qmugs/smol \
    --dataset geom-drugs \
    --batch_cost 128 \
    --epochs 10

echo "Job finished at: $(date)"