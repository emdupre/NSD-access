#!/bin/bash
#SBATCH --job-name=nsd-shared1000
#SBATCH --array=0,1,2,3
#SBATCH --mem=15G
#SBATCH --time=00:05:00
#SBATCH --account=def-pbellec

SCRATCH=/scratch/emdupre
PROJECT=/scratch/emdupre/NSD
subjects=("subj01" "subj02" "subj05" "subj07")

source $PROJECT/nsd.env/bin/activate

echo ${subjects[$SLURM_ARRAY_TASK_ID]}

python $PROJECT/src/compile-shared1000.py \
    --subj_name=${subjects[$SLURM_ARRAY_TASK_ID]} \
    --data_dir=$PROJECT
