#!/bin/bash
#SBATCH --job-name=nsd-niftis
#SBATCH --array=0
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --account=def-pbellec

SCRATCH=/scratch/emdupre
PROJECT=/scratch/emdupre/NSD
subjects=("subj01" "subj02" "subj03" "subj04" "subj05" "subj06" "subj07" "subj08")

source $PROJECT/nsd.env/bin/activate

echo ${subjects[$SLURM_ARRAY_TASK_ID]}

python $PROJECT/src/gen-alignment-niftis.py \
    --subj_name=${subjects[$SLURM_ARRAY_TASK_ID]} \
    --data_dir=$PROJECT