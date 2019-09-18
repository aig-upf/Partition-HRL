#!/usr/bin/env bash

source /homedtic/lsteccanella/RL/cluster_utils/modules.sh
source /homedtic/lsteccanella/RL/cluster_utils/RL-cluster-venv/bin/activate
python -u /homedtic/lsteccanella/RL/. "$@"

