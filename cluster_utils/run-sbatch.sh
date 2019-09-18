#!/usr/bin/env bash

#SBATCH -J HRL # name of the project
#SBATCH -p high # priority
#SBATCH -N 1 # number of nodes
#SBATCH -n 4 # number of cpu
#SBATCH --mem 40gb
#SBATCH --workdir=/homedtic/lsteccanella/RL/ # working directory project
#SBATCH -C intel #request intel node (those have infiniband) # intel node
#SBATCH -o /homedtic/lsteccanella/RL/cluster_utils/jobs/%N.%J.out # STDOUT # output to number of node number of job
#SBATCH -e /homedtic/lsteccanella/RL/cluster_utils/jobs/%N.%j.err # STDERR # output of the error

# set -x # output verbose
source /homedtic/lsteccanella/RL/cluster_utils/modules.sh
source /homedtic/lsteccanella/RL/cluster_utils/RL-cluster-venv/bin/activate
python -u /homedtic/lsteccanella/RL/. "$@"

