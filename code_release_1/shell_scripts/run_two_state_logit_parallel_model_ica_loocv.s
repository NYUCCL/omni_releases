#!/bin/bash
#
#SBATCH --job-name=stanTest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --mem=50GB
#SBATCH --mail-type=END
#SBATCH --mail-user=david.halpern@nyu.edu
#SBATCH --output=slurm_two_state_logit_parallel_loocv_%A_%a.out

module purge

SRCDIR=$HOME/behav_analysis/analysis

module load python3/intel/3.6.3
source $HOME/py3.6.3/bin/activate

python $SRCDIR/run_stan_model.py two_state_logit_parallel_model $SLURM_ARRAY_TASK_ID cluster --shards 4
## run call: sbatch --array=0-19 run_bkt_model_loocv.s
