import pystan
import _pickle as pickle
import pandas as pd
import argparse
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument('model_name')
parser.add_argument('sub_id', type=int)
parser.add_argument('run_loc')
# stan parameters
parser.add_argument('--n_samps', dest='n_samps', default=200, type=int)
parser.add_argument('--max_treedepth', dest='max_treedepth', default=10, type=int)
parser.add_argument('--adapt_delta', dest='adapt_delta', default=.95, type=float)
parser.add_argument('--chains', dest='chains', default=0, type=int)
parser.add_argument('--n_jobs', dest='n_jobs', default=1, type=int)
# cv options:
# loocv = leave one (fmri) subject out, 
# all_kfcv = leave one of k-folds out, folds stratified over delay groups, 
# fmri_kfcv = leave one of k-folds of fmri subject data out, folds stratified over subject and response correctness
parser.add_argument('--cv', dest='cv', default='loocv') 
# n splits for cross validation
parser.add_argument('--n_splits', dest='n_splits', default=10, type=int)
# number of shards for parallelization
parser.add_argument('--shards', dest='shards', default=1, type=int)
args = vars(parser.parse_args())
# sample calls to run_stan_model.py
# python run_stan_model.py test 1 laptop
# python run_stan_model.py two_state_logit_model 1 laptop --n_samps 1
# python run_stan_model.py two_state_logit_parallel_model 1 laptop --n_samps 1 --shards 2

sub_id = args['sub_id']
n_samps = args['n_samps']
n_splits = args['n_splits']
max_treedepth = args['max_treedepth']
adapt_delta = args['adapt_delta']
cv = args['cv']
run_loc = args['run_loc']
model_name = args['model_name']
chains = args['chains']
n_jobs = args['n_jobs']
shards = args['shards']

print(pystan.__version__)
print(model_name)
print(n_samps)
print(cv)
print('n_jobs', n_jobs)

# FILE INFO
if run_loc == 'laptop':
  #base_dir = '/Users/davidhalpern/Documents/GitHub/omni_releases/'
  base_dir = '/path/to/omni_releases'
  output_dir = base_dir
  data_dir = base_dir + 'data_release_1/'
  code_dir = base_dir + 'code_release_1/'
  model_dir = code_dir + 'models/'
  if chains == 0:
    chains = 1
elif run_loc == 'cluster':
  #base_dir = '/home/djh389/'
  base_dir = '/path/to/omni_releases'
  scratch_dir = '/scratch/' #to store model fits in scratch storage
  output_dir = scratch_dir + 'model_fits/'
  model_dir = base_dir + 'models/'
  if chains == 0:
    chains = 4
  data_dir = base_dir + 'data/'

sys.path.append(base_dir)
import data_utils as du

raw_df = pd.read_csv(data_dir + 'omni_behav_data_release_April-2-2018.csv')
df = du.preprocess_df(raw_df)

#Remove rows missing study trial
df = df[~df[['lag1', 'lag2', 'lag3', 'lag4', 'study_test_lag']].isnull().any(axis=1)]

heldout_subs = df[df['delay_group'] == 'PRISMAF'].participant_id.unique()
print(heldout_subs)

if 'two_state' in model_name:
  S = 2
else:
  S = 3

dat, heldout_sub = du.get_stan_dat(df, split_id=sub_id, n_splits=n_splits, heldout_subs=heldout_subs, cv=cv, S=S, shards=shards)

print(heldout_sub)

model_file = model_dir + model_name + '.stan'

#extra_compile_args = ['-g', '-O0']
extra_compile_args = None
if shards > 1:
  os.environ['STAN_NUM_THREADS'] = "-1"
  extra_compile_args = ['-pthread', '-DSTAN_THREADS']

model = pystan.StanModel(file=model_file, extra_compile_args=extra_compile_args)
print('done compiling')
print(extra_compile_args)
if 'laptop' in run_loc:
  fit = model.sampling(data=dat, iter=n_samps, chains=chains, control={'adapt_delta': adapt_delta, 'max_treedepth': max_treedepth}, verbose=True, n_jobs=n_jobs, check_hmc_diagnostics=False)
else:
  fit = model.sampling(data=dat, iter=n_samps, chains=chains, control={'adapt_delta': adapt_delta, 'max_treedepth': max_treedepth}, verbose=True)

samples = fit.extract(permuted=False)
sampler_params = fit.get_sampler_params()
labels = (fit.sim['pars_oi'], fit.sim['dims_oi'], fit.sim['fnames_oi'])
control_params = (n_samps, adapt_delta, max_treedepth)

out_file = output_dir + 'fit_' + args['model_name'] + '_' + str(cv) + '_' + str(heldout_sub) + '_' + str(n_samps) + '.pkl'

with open(out_file, 'wb') as f:
  pickle.dump((samples, labels, sampler_params, control_params, heldout_sub, dat), f)
