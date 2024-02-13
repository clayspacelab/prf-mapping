#!/bin/bash

#SBATCH --job-name=slurmprf_mapFIT
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=30:00:00
#SBATCH --mem=4GB
#SBATCH --output=/scratch/ac8586/prf-mapping/code/logs/%x_%A_%a.out
##SBATCH --error=/scratch/ac8586/prf-mapping/code/logs/%x_%A_%a.err


module purge

echo "Date: $DATE"
printf "Log file: $LOG_FILE\n\n"

singularity exec \
            --overlay /scratch/ac8586/prf-mapping/myenv.ext3:ro \
            /scratch/work/public/singularity/ubuntu-22.04.1.sif \
            /bin/bash -c "source /ext3/env.sh; python prf_css.py"