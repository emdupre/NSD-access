#!/bin/bash
#SBATCH --job-name=nsd-niftis
#SBATCH --array=0-7
#SBATCH --mem=100G
#SBATCH --cpus-per-task=50
#SBATCH --time=00:30:00
#SBATCH --account=def-pbellec

SCRATCH=/scratch/emdupre
PROJECT=/scratch/emdupre/NSD
subjects=("subj01" "subj02" "subj03" "subj04" "subj05" "subj06" "subj07" "subj08")

source $PROJECT/nsd.env/bin/activate

echo ${subjects[$SLURM_ARRAY_TASK_ID]}

python $PROJECT/src/gen-stim-niftis.py \
    --subj_name=${subjects[$SLURM_ARRAY_TASK_ID]} \
    --data_dir=$PROJECT