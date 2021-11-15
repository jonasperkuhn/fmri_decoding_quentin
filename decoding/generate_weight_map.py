#!/usr/bin/env python3

""" Script to decode image type (negative vs. neutral) from fmri brain activity """

import os
import pandas as pd
import numpy as np
from nilearn import plotting
from nilearn.image import mean_img
from nilearn.image import math_img
from nilearn.image import load_img, index_img
from nilearn.decoding import Decoder
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import LeaveOneGroupOut
from nilearn.plotting import plot_stat_map

# define functions
def define_conds(n_sessions):
    # set params
    n_scale_per_ses = 2
    n_blocks_per_cond = 3
    n_tr_per_block = 15
    n_tr_scale = 6
    conds = ['neutral', 'negative']
    colnames = ['session', 'condition', 'block', 'TR']
    # construct table
    conds_fmri = pd.DataFrame(columns=colnames)
    cols_df = conds_fmri.columns
    for i_ses in range(n_sessions):
        i_block = 0  # set block counter to 0
        # append first two instr tr of session
        for i in range(2):
            row_to_append = pd.Series([i_ses, 'instr', i_block, np.nan], index=cols_df)
            conds_fmri = conds_fmri.append(row_to_append, ignore_index=True)
        # append half a session, until scale tr
        for i_scale in range(n_scale_per_ses):
            # append block of both conditions
            for i_block_cond in range(n_blocks_per_cond):
                # append both conditions into separate blocks
                for i_cond, cond in enumerate(conds):
                    i_block += 1  # add one to block counter
                    # append instruction tr at beginning of block
                    row_to_append = pd.Series([i_ses, 'instr', i_block, np.nan], index=cols_df)
                    conds_fmri = conds_fmri.append(row_to_append, ignore_index=True)
                    for i_tr in range(n_tr_per_block):
                        row_to_append = pd.Series([i_ses, cond, i_block, np.nan], index=cols_df)
                        conds_fmri = conds_fmri.append(row_to_append, ignore_index=True)
            # append scale tr's
            for i in range(n_tr_scale):
                row_to_append = pd.Series([i_ses, 'scale', i_block, np.nan], index=cols_df)
                conds_fmri = conds_fmri.append(row_to_append, ignore_index=True)
    # add TR values
    conds_fmri['TR'] = range(len(conds_fmri))
    return conds_fmri

def load_data(preprocessing: str, data_path: str, n_sessions: int, plot: bool=False):
    # generate conditions file
    conds_fmri = define_conds(n_sessions)
    # load brain data
    fmri_data, fname_anat = load_brain_data(preprocessing, data_path)
    # import mask (and binarize, if specified)
    fname_mask = os.listdir(data_path + 'Mask_ROI_emo/')[0]  # get filename of first mask in folder
    mask = load_img(img=data_path + 'Mask_ROI_emo/' + fname_mask)  # load binarized mask
    #mask = math_img('img > 5', img=data_path + 'Mask_ROI_emo/' + fname_mask)  # load mask and binarize
    #mask_bin.to_filename(data_path + roi + '_mask_bin.nii')  # save binarized mask
    # select only stimulus trials (in brain and behavioral data)
    conditions_all = conds_fmri['condition']
    condition_mask = conditions_all.isin(['neutral', 'negative'])  # index to restrict data to negative or neutral stimuli
    fmri_niimgs = index_img(fmri_data, condition_mask)
    conditions_trials = conditions_all[condition_mask]
    conditions = conditions_trials.values  # Convert to numpy array
    # plot results for checking
    if plot:
        print(fmri_data.shape)  # print shape of fmri data
        p1 = plotting.view_img(mean_img(fmri_data), threshold=None)  # todo: sth. wrong with brain data?
        p1.open_in_browser()
        plotting.plot_roi(mask, bg_img=fname_anat, cmap='Paired')  # plot mask
    return fmri_niimgs, fname_anat, mask, conds_fmri, condition_mask, conditions

def load_brain_data(preprocessing, data_path):
    # enter preprocessing arg as 'r', 'sr', or 'swr
    # import fmri data
    fnames_fmri = os.listdir(data_path + 'EPIs_baseline')
    fname_fmri = [item for item in fnames_fmri if item.startswith(preprocessing)][0]
    fmri_data = load_img(data_path + 'EPIs_baseline/' + fname_fmri)  # concatenate brain data
    # import anatomical data (T1)
    fnames_anat = os.listdir(data_path + 'T1')
    fname_anat = data_path + 'T1/' + [item for item in fnames_anat if item.startswith('2') and item.endswith('.nii')][0]
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
        cv = RepeatedKFold(n_splits=n_folds, n_repeats=5, random_state=random_state)  # todo: add n_repeats as input options?
        scoring = 'accuracy'
        groups = None
    elif cv_type == 'block_out':
        cv = LeaveOneGroupOut()
        scoring = 'roc_auc'
        groups = conds_fmri['block'][condition_mask]  # todo: change from session to block
    else:
        print('Input error "cv_type": Please indicate either as "k_fold" or as "block_out"')
        return
    # build decoder
    decoder = Decoder(estimator='svc', mask=mask, cv=cv, screening_percentile=screening_percentile,
                      scoring=scoring, smoothing_fwhm=smoothing_fwhm, standardize=True) # todo: discuss settings with Pauline
    # fit decoder
    decoder.fit(fmri_niimgs, conditions, groups=groups)
    return decoder

def plot_weights(decoder, fname_anat, condition):
    # plot model weights
    #coef_ = decoder.coef_
    #print(coef_.shape)
    weigth_img = decoder.coef_img_[condition]
    plot_stat_map(weigth_img, bg_img=fname_anat, title='SVM weights')
    p2 = plotting.view_img(weigth_img, bg_img=fname_anat, title="SVM weights", dim=-1)
    p2.open_in_browser()
    return


# define path to project folder and main params
data_path = "C:/Users/Jonas/PycharmProjects/fmri_decoding_quentin/decoding/data/SESSION 1/"  # set path to data folder of current set
preprocessing = "sr"  # specify as 'r' (realigned), 'sr' (realigned + smoothed), or 'swr' (sr + normalization)
random_state = 8

# load data
fmri_niimgs, fname_anat, mask, conds_fmri, condition_mask, conditions = \
    load_data(preprocessing, data_path, n_sessions=1, plot=False)  # todo: remove session param from script

# build and fit decoder in cv
decoder = perform_decoding_cv(conditions, fmri_niimgs, mask, conds_fmri,
                              condition_mask, random_state, cv_type='k_fold', n_folds=5, anova=False)
# evaluate decoder
print(np.mean(decoder.cv_scores_['negative']))  # todo: understand score (misclassif.?, why so good with motor?, try with other unrelated masks?)

# plot decoder weights
plot_weights(decoder, fname_anat, condition='negative')

# save decoder weights
weigth_img = decoder.coef_img_['negative']
weigth_img.to_filename(data_path + 'W1/weights.nii.gz')
