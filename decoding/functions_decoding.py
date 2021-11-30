#!/usr/bin/env python3

""" Functions to call for decoding """

import os
import pandas as pd
import numpy as np
from nilearn.image import mean_img
from nilearn.image import load_img, index_img
from nilearn.plotting import view_img, plot_roi, plot_stat_map
from nilearn.decoding import Decoder
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import LeaveOneGroupOut

# define functions
def load_data(strategy: str, preprocessing: str, data_path: str, plot: bool=False):
    # get conditions data from txt file
    fname_onsets = data_path + 'Onsets/' + os.listdir(data_path + 'Onsets')[0]  # get filename of text file with onsets (output from opennft)
    onsets_all = pd.read_csv(fname_onsets, delimiter='\t', index_col=False, skiprows=4)  # load txt file with onsets
    onsets = onsets_all[onsets_all['strategie'] == strategy]
    cond_names = [' neutre', ' regulation']  # set names of two main conditions (like in txt file) ! pay attention to space bar before word (coming from text file)
    tr = 2  # set TR (in seconds)
    conds_fmri = get_conds_from_txt(onsets, cond_names, tr)  # convert onset data to conditions file (['block', 'condition', 'TR'] with one row per TR)
    # load brain data
    fmri_data, fname_anat = load_brain_data(preprocessing, data_path)  # load fmri data and T1
    # align brain data and onset data
    conds_fmri = conds_fmri.iloc[list(range(fmri_data.shape[3]))]  # cut onsets to brain data (cut off last scale tr's that are variable)
        # todo: check back with Pamela, if that is okay
    # import mask (needs to be binarized)
    fname_mask = os.listdir(data_path + 'Mask_ROI_emo')[0]  # get filename of first mask in folder
    mask = load_img(img=data_path + 'Mask_ROI_emo/' + fname_mask)  # load binarized mask
    # select only stimulus trials (in brain and behavioral data)
    conditions_all = conds_fmri['condition']
    condition_mask = conditions_all.isin(cond_names)  # index to restrict data to negative or neutral stimuli
    fmri_niimgs = index_img(fmri_data, condition_mask)  # select only neutral/negative trials from brain data
    conditions_trials = conditions_all[condition_mask]  # select only neutral/negative trials from conditions
    conditions = conditions_trials.values  # Convert to numpy array
    # plot results for checking
    if plot:
        print('fMRI data shape: ', fmri_data.shape)  # print shape of fmri data
        p1 = view_img(mean_img(fmri_data), threshold=None)
        p1.open_in_browser()
        plot_roi(mask, bg_img=fname_anat, cmap='Paired')  # plot mask
    return fmri_niimgs, fname_anat, mask, conds_fmri, condition_mask, conditions, cond_names

def get_conds_from_txt(onsets, cond_names, tr):
    onsets_tr_selected = onsets[['condition', 'onsets_seconds']]  # select condition and onsets from txt file
    # convert onsets (in seconds) to TR
    onsets_tr = onsets_tr_selected.copy()  # copy data frame, due to pandas rules
    onsets_tr.loc[:, 'dur_TR'] = np.nan  # add row to save duration in TR instead of seconds
    for row, onset in enumerate(onsets_tr['onsets_seconds']):
        onsets_tr.at[row, 'dur_TR'] = int(round(onset / tr))  # convert seconds to TR  # todo: change indexing - row names are 1-based; quick change: onsets_tr.at[row+1, 'dur_TR']
    n_TR = int(list(onsets_tr['dur_TR'])[-1] + 1)  # total number of TR (1 added because of base 0)
    # set up and fill final conditions data frame with n_rows = n_TR
    colnames = ['block', 'condition', 'TR']  # define variable names
    conds_fmri = pd.DataFrame(index=range(n_TR), columns=colnames)  # set up data frame
    conds_fmri['TR'] = range(n_TR)  # fill in one TR per row
    # set conditions
    for TR in conds_fmri['TR']:
        # find (closest previous) condition for each TR
        diff_to_tr = np.array([TR - dur_TR for dur_TR in onsets_tr['dur_TR']])  # compare TR to all TRs in original onsets table
        diff_to_tr_pos = np.where(diff_to_tr >= 0, diff_to_tr, np.inf)  # set negative values to infinity (to avoid getting condition of subsequent trial, following the onset time)
        i_closest = diff_to_tr_pos.argmin()  # find condition of closest positive diff.
        conds_fmri.iat[TR, 1] = onsets_tr['condition'][i_closest]  # set closest condition in conds_fmri['condition']
    # set block number from condition
    i_block = 0  # set block index to 0
    for row, cond in enumerate(conds_fmri['condition']):
        conds_fmri.at[row, 'block'] = i_block  # set block number
        # find last trial of each block
        if row < (n_TR - 1):  # for every trial except very last trial
            cond_next_trial = conds_fmri['condition'][row + 1]  # get condition of subsequent trial
            if cond == cond_names[1] and cond_next_trial != cond_names[1]:
                # if condition equals 'regulation' and subsequent trial does not (-> final regulation trial)
                # comment: block index for scale trials can be ignored
                i_block = i_block + 1  # increase block index by 1
    for i in range(20): # todo: relevant for pilots 3 - 5, can be removed later: append 20 rows at the end of the data frame, to correct for the unsaved final onset
        row_to_append = pd.Series([i_block, 'break/end', np.nan], index=colnames)
        conds_fmri = conds_fmri.append(row_to_append, ignore_index=True)
    return conds_fmri

def load_brain_data(preprocessing, data_path):
    # enter preprocessing arg as 'r', 'sr', or 'swr
    # import fmri data
    fnames_fmri = os.listdir(data_path + 'EPIs_baseline')  # get list of all file names in fMRI brain data folder
    fname_fmri_preprocessed = [item for item in fnames_fmri if item.startswith(preprocessing)][0]  # select file that matches preprocessing
    fmri_data = load_img(data_path + 'EPIs_baseline/' + fname_fmri_preprocessed)  # load brain data
    # import anatomical data (T1)
    fnames_anat = os.listdir(data_path + 'T1')  # get list of all file names in T1 brain data folder
    fname_anat = data_path + 'T1/' + [item for item in fnames_anat if item.startswith('2') and item.endswith('.nii')][0]  # select .nii file without prefix
    return fmri_data, fname_anat

def perform_decoding_cv(conditions, fmri_niimgs, mask, conds_fmri, condition_mask, random_state: int, cv_type: str, n_folds: int, anova: bool):
    # perform feature reduction via anova
    if anova:
        smoothing_fwhm = 8
        screening_percentile = 5
    else:
        smoothing_fwhm = None
        screening_percentile = 20
    # determine cv method
    if cv_type == 'k_fold':
        cv = RepeatedKFold(n_splits=n_folds, n_repeats=5, random_state=random_state)
        scoring = 'accuracy'
        groups = None
    elif cv_type == 'block_out':
        cv = LeaveOneGroupOut()
        scoring = 'roc_auc'
        groups = conds_fmri[condition_mask]['block']
    else:
        print('Input error "cv_type": Please indicate either as "k_fold" or as "block_out"')
        return
    # build decoder
    decoder = Decoder(estimator='svc', mask=mask, cv=cv, screening_percentile=screening_percentile,
                      scoring=scoring, smoothing_fwhm=smoothing_fwhm, standardize=True)
    # fit decoder
    decoder.fit(fmri_niimgs, conditions, groups=groups)
    return decoder

def plot_weights(decoder, fname_anat, condition):
    # plot model weights
    #coef_ = decoder.coef_
    #print(coef_.shape)
    weigth_img = decoder.coef_img_[condition]
    plot_stat_map(weigth_img, bg_img=fname_anat, title='SVM weights')
    #p2 = view_img(weigth_img, bg_img=fname_anat, title="SVM weights", dim=-1)  # todo: seems to select false T1?
    #p2.open_in_browser()
    return

def save_accs_to_txt(mean_score, scores, data_path):
    with open(data_path + 'W1/decoding_accuracies.txt', 'w') as f:
        f.write('mean accuracy across folds:')
        f.write('\n')
        f.write(str(mean_score))
        f.write('\n')
        f.write('accuracies per cv fold and repetition:')
        f.write('\n')
        f.write(str(scores))
    return

def compare_cvs(strategy, data_path, random_state):
    # set different preprocessing and cv types
    preprocessing_types = ['r', 'sr', 'swr', 'swr_anova']  # 'r' (realigned), 'sr' (realigned + smoothed), 'swr' (sr + normalization)
    cv_types = ['k_fold', 'block_out']  # cross-validation type
    kfold_types = [5, 10]  # number of folds to perform in k-fold cross-validation; only makes a difference if cv_type == 'k_fold'

    # create table to store results
    row_names = [cv + str(fold) for cv in cv_types for fold in
                 kfold_types]  # create row names (comb. of cv and fold, to make 2D)
    avg_scores = pd.DataFrame(index=row_names, columns=preprocessing_types)  # df for mean scores
    avg_stds_scores = pd.DataFrame(index=row_names, columns=preprocessing_types)  # df for standard deviation of scores

    # loop over different preprocessing and cv types
    for i_prep, preprocessing in enumerate(preprocessing_types):
        # set anova variable
        if preprocessing == 'swr_anova':
            anova = True
            preprocessing = 'swr'
        else:
            anova = False
        # loop over cv_types
        for cv_type in cv_types:
            for n_folds in kfold_types:
                # load data
                fmri_niimgs, fname_anat, mask, conds_fmri, condition_mask, conditions, cond_names = \
                    load_data(strategy, preprocessing, data_path, plot=False)
                # build and fit decoder in cv
                decoder = perform_decoding_cv(conditions, fmri_niimgs, mask, conds_fmri,
                                              condition_mask, random_state, cv_type=cv_type, n_folds=n_folds,
                                              anova=anova)
                # save decoder scores in table
                avg_scores.at[cv_type + str(n_folds), preprocessing_types[i_prep]] = np.mean(decoder.cv_scores_[cond_names[1]])
                avg_stds_scores.at[cv_type + str(n_folds), preprocessing_types[i_prep]] = np.std(decoder.cv_scores_[cond_names[1]])
                print(preprocessing_types[i_prep])
                print(cv_type)
                print(n_folds)
    return avg_scores, avg_stds_scores
