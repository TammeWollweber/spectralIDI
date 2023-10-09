#!/bin/bash

#!/bin/bash

#SBATCH --nodes=1
#SBATCH --nodelist=mpsd-hpc-gpu-004
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=36:00:00
#SBATCH --export=ALL
#SBATCH -J sim
#SBATCH -o .%j.out
#SBATCH -e .%j.out
#SBATCH --partition=gpu-ayyer
#SBATCH --gpus=4

#hostname
#nvidia-smi
#sshfs laptop:/media/wittetam/Expansion/ /home/wittetam/mount/
srun python simulate_nofilter.py -n 10000 -N 100000 -m 1 -i 1 -f 1 -t 0 -a 2
