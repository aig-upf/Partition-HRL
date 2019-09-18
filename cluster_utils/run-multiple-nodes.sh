for var in "$@"
do
    sbatch /homedtic/lsteccanella/RL/cluster_utils/run-sbatch.sh "$var"
    sleep 5
done
