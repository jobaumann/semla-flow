srun --time=4:00:00 \
       --nodes=1 \
       --ntasks-per-node=1 \
       --cpus-per-task=8 \
       --gpus=rtx_4090:1 \
       --mem-per-cpu=16G \
       --partition=gpu.4h \
       --pty bash