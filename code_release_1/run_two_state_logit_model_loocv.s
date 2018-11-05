#!/bin/bash
#
#SBATCH --job-name=stanTest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --mem=50GB
#SBATCH --mail-type=END
#SBATCH --mail-user=david.halpern@nyu.edu
#SBATCH --output=slurm_two_state_logit_loocv_%A_%a.out

module purge

SRCDIR=$HOME/behav_analysis/analysis

module load anaconda3/4.3.1
source activate $HOME/.conda/envs/py3.6.3

python $SRCDIR/run_stan_model.py two_state_logit_model $SLURM_ARRAY_TASK_ID cluster
## run call: sbatch --array=0-31 run_two_state_logit_model_loocv.s
