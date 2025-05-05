#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --partition=all.q
#SBATCH --account=proj_simscience
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G

ENV_PATH=$1
REPO_DIR=$2

source ${ENV_PATH}/bin/activate
cd ${REPO_DIR}

pytest --runslow .