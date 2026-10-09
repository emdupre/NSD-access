#!/bin/bash
#SBATCH --job-name=nsd-niftis
#SBATCH --array=1-7
#SBATCH --mem=50G
#SBATCH --cpus-per-task=25
#SBATCH --time=16:00:00
#SBATCH --account=def-pbellec

SCRATCH=/scratch/emdupre
PROJECT=/scratch/emdupre/NSD
subjects=("subj01" "subj02" "subj03" "subj04" "subj05" "subj06" "subj07" "subj08")

source $PROJECT/nsd.env/bin/activate

echo ${subjects[$SLURM_ARRAY_TASK_ID]}

python $PROJECT/src/gen-stim-niftis.py \
    --subj_name=${subjects[$SLURM_ARRAY_TASK_ID]} \
    --data_dir=$PROJECT
