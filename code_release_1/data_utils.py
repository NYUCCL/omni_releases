import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import StratifiedKFold

def get_dat_from_df(df, time_norm, jol_norm):
    #train
    l_seconds = np.array(df[['lag1', 'lag2', 'lag3', 'lag4', 'study_test_lag']])
    l = l_seconds/time_norm
    T = l.shape[0]
    N = len(df['participant_id'].unique())
    jol_raw = np.array(df['jol_value'], dtype='int')
    jol = jol_raw/jol_norm
    
    sub = np.array(df['participant_num'], dtype='int') ##need to change
    r = np.array(df['recall_accuracy'], dtype='int')
    w = np.array(df['word_num'], dtype='int')

    dat = {'l': l,
            'T': T,
            'N': N,
            'jol': jol,
            'sub': sub,
            'r': r,
            'w': w}
    return dat

def get_mri_splits(df, split_id, n_splits, random_state=1234):
    #make split stratified on combination of participant and correct trial (pid_rs below)
    mri_df = df[df['delay_group'] == 'PRISMAF']
    beh_df = df[~(df['delay_group'] == 'PRISMAF')]

    pids = np.array(mri_df['participant_id'])
    rs = np.array(mri_df['recall_accuracy'])
    pid_rs = [pid + '_' + str(int(r)) for pid, r in zip(pids, rs)]

    if split_id < n_splits:
        heldout_sub = split_id
        cv_split = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        #StratifiedKFold only splits on ys so Xs can be anything of the same length
        train, holdout = next(itertools.islice(cv_split.split(pid_rs, pid_rs), split_id, None))

        df_t = beh_df.append(mri_df.iloc[train])
        df_h = mri_df.iloc[holdout]
    else:
        heldout_sub = 'All'
        df_t = df
        df_h = df
    return df_t, df_h, heldout_sub

def get_all_splits(df, split_id, n_splits, random_state=1234):
    #make split stratified on group (delay period) of subjects
    pid_dg_combs = df[['participant_id', 'delay_group']].drop_duplicates()
    delay_group_ids = np.array(pid_dg_combs['delay_group'])
    pids = np.array(pid_dg_combs['participant_id'])
    print(pids)
    
    if split_id < n_splits:
        heldout_sub = split_id
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1234)
        train, holdout = next(itertools.islice(cv.split(pids, delay_group_ids), split_id, None))

        df_t = df[df['participant_id'].isin(pids[train])]
        df_h = df[df['participant_id'].isin(pids[holdout])]
    else:
        heldout_sub = 'All'
        df_t = df
        df_h = df
    return df_t, df_h, heldout_sub
    
def get_stan_dat(df, split_id=0, n_splits=10, heldout_subs=[], cv=None, random_state=1234, S=3, shards=1):
    W = df['eng_word_studied'].unique().shape[0] #total nunmber of words
    max_L = 5 # length of protocols
    time_norm = float(100000) #normalized so roughly 50% change of forgetting per time step on average
    jol_norm = float(100) #normalized so on scale from 0 to 1
    N = df['participant_id'].unique().shape[0] #total number of subjects
    T = df.shape[0] #total number of trials

    if cv == 'all_kfcv':
        df_t, df_h, heldout_sub = get_all_splits(df, split_id, n_splits, random_state=random_state)
    elif cv == 'fmri_subs_kfcv':
        df_t, df_h, heldout_sub = get_mri_splits(df, split_id, n_splits, random_state=random_state)
    elif cv == 'loocv':
        if split_id not in range(len(heldout_subs)):
            heldout_sub = 'All'
        else:
            heldout_sub = heldout_subs[split_id]

        df_t = df[df['participant_id'] != heldout_sub]
        if heldout_sub == 'All':
            df_h = df_t
        else:
            df_h = df[df['participant_id'] == heldout_sub]
    else:
        print('no cv')
        heldout_sub = 'All'
        df_t = df
        df_h = df
    print(cv)
    print(heldout_sub)

    state_prior = [0.99, 0.005, 0.005] # priors for U, T, P
    recall_prob = [.01, .9, .9] # recall prob for U, T, P
    if S == 2:
        state_prior = [0.99, 0.01]
        recall_prob = recall_prob[:S]

    #train
    dat_t = get_dat_from_df(df_t, time_norm, jol_norm)

    #holdout
    dat_h = get_dat_from_df(df_h, time_norm, jol_norm)
    
    if heldout_sub == 'All':
        heldout = 0
    else:
        heldout = 1

    dat = {'S': S, # n states
           'W': W, # n words
           'L': max_L, # length of protocol (all L are the same)
           'N': N, # total n subs
           'T': T, # total n trials
           'shards': shards, # n shards for parallelization

           'heldout': heldout,

           'state_prior': state_prior, # priors for U, T, P
           'recall_prob': recall_prob, # recall prob for U, T, P
           }

    for key in dat_t.keys():
        key_t = '{}_t'.format(key)
        dat[key_t] = dat_t[key]

    for key in dat_h.keys():
        key_h = '{}_h'.format(key) 
        dat[key_h] = dat_h[key]

    return dat, heldout_sub

def preprocess_df(df):
    new_df = df
    new_df['word_num'] = pd.factorize(new_df['eng_word_studied'])[0] + 1
    new_df['participant_num'] = pd.factorize(new_df['participant_id'])[0] + 1
    
    new_df.replace({'study1_start': {-999: np.nan},
                    'study2_start': {-999: np.nan},
                    'study3_start': {-999: np.nan},
                    'study4_start': {-999: np.nan},
                    'study5_start': {-999: np.nan},
                    'jol_start': {-999: np.nan},
                    'recall_start': {-999: np.nan}}, inplace=True)
    
    new_df['lag1'] = new_df.study2_start - new_df.study1_start
    new_df['lag2'] = new_df.study3_start - new_df.study2_start
    new_df['lag3'] = new_df.study4_start - new_df.study3_start
    new_df['lag4'] = new_df.study5_start - new_df.study4_start
    new_df['study_test_lag'] = new_df.recall_start - new_df.study5_start
    new_df['study_jol_lag'] = new_df.jol_start - new_df.study5_start

    return new_df
