#!/usr/bin/env python3
""" Script to decode image type (negative vs. neutral) from fmri brain activity """

import pandas as pd
import numpy as np
from nilearn import plotting
from nilearn.image import mean_img
from nilearn.image import math_img
from nilearn.image import concat_imgs, index_img
from nilearn.decoding import Decoder
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import LeaveOneGroupOut
from nilearn.plotting import plot_stat_map

# define functions
def load_data(mask, project_path: str):
    # import behavioral data
    behavioral = pd.read_csv(project_path + '/data/onsets_decoding_pilot01_ses12.csv', delimiter=';')
    conditions = behavioral['Condition']
    condition_mask = conditions.isin(['Negatif', 'Neutre'])
    # import and concatenate brain data
    fmri_ses1 = project_path + '/data/brain_data/Session_1_epi3mm_MB2_TE30_TR2000_IRMf_20210727114430_4.nii'
    fmri_ses2 = project_path + '/data/brain_data/Session_2_epi3mm_MB2_TE30_TR2000_IRMf_20210727114430_6.nii'
    fmri_ses12 = concat_imgs([fmri_ses1, fmri_ses2])
    print(fmri_ses12.shape)
    p1 = plotting.view_img(mean_img(fmri_ses12), threshold=None)
    p1.open_in_browser()
    # import mask and t1 data
    anat = '/data/brain_data/wDicom_baseline_mprage_sag_T1_160sl_iPAT2_20210727114430_2.nii'
    mask_bin = '/mask.nii'
    #plotting.plot_roi(mask_bin, bg_img=anat, cmap='Paired')
    # select only stimulus trials
    fmri_niimgs = index_img(fmri_ses12, condition_mask)
    conditions = conditions[condition_mask]
    conditions = conditions.values  # Convert to numpy array
    return conditions, fmri_niimgs, anat, mask_bin, behavioral, condition_mask

def perform_decoding_cv(cv_type: str, anova: bool, conditions, fmri_niimgs, mask_bin, behavioral, condition_mask, random_state):
    # perform feature reduction via anova
    if anova:
        print('Anova code still missing')
    # build decoder
    if cv_type == 'k_fold':
        cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=random_state)
        decoder = Decoder(
            estimator='svc', mask=mask_bin,
            standardize=True, cv=cv,
            scoring='accuracy')
        groups = None
    elif cv_type == 'session_out':
        cv = LeaveOneGroupOut()
        decoder = Decoder(estimator='svc', mask=mask_bin, standardize=True, cv=cv)
        groups = behavioral['Session'][condition_mask]
    else:
        print('Input error "cv_type": Please indicate either as "k_fold" or as "session_out"')
        return
    # fit decoder
    decoder.fit(fmri_niimgs, conditions, groups=groups)
    return decoder

def plot_weights(decoder, anat):
    # plot model weights
    coef_ = decoder.coef_
    print(coef_)
    print(coef_.shape)
    weigth_img = decoder.coef_img_['Negatif']
    weigth_img.to_filename('pilot01_svc_weights_ses12.nii.gz')
    plot_stat_map(weigth_img, bg_img=anat, title='SVM weights')
    p2 = plotting.view_img(weigth_img, bg_img=anat, title="SVM weights", dim=-1)
    p2.open_in_browser()
    return

# define path to project folder and main params
mask = "whole_brain"
project_path = "C:/Users/Jonas/PycharmProjects/fmri_decoding_quentin/decoding"
random_state = 8
# load data
conditions, fmri_niimgs, anat, mask_bin, behavioral, condition_mask = \
    load_data(mask=mask, project_path=project_path)
# build and fit decoder in cv
decoder = \
    perform_decoding_cv(cv_type='k_fold', anova=False, conditions=conditions, fmri_niimgs=fmri_niimgs,
        mask_bin=mask_bin, behavioral=behavioral, condition_mask=condition_mask, random_state=random_state)
# evaluate decoder
print(np.mean(decoder.cv_scores_['Negatif']))
# plot decoder weights
plot_weights(decoder, anat)
