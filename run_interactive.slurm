srun --time=4:00:00 \
       --nodes=1 \
       --ntasks-per-node=1 \
       --cpus-per-task=8 \
       --gpus=1 \
       --mem-per-cpu=8G \
       --partition=gpu.4h \
       --pty bash