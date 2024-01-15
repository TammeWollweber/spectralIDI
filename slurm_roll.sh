#!/bin/bash

#!/bin/bash

#SBATCH --nodes=1
#SBATCH --nodelist=mpsd-hpc-gpu-003
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=36:00:00
#SBATCH --export=ALL
#SBATCH -J roll
#SBATCH -o .%j.out
#SBATCH -e .%j.out
#SBATCH --partition=gpu-ayyer
#SBATCH --gpus=1

#hostname
#nvidia-smi
#sshfs laptop:/media/wittetam/Expansion/ /home/wittetam/mount/
srun python roll_mem.py 
