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
def load_data(preprocessing: str, roi: str, data_path: str, bin_mask: bool=False, plot: bool=False):
    # load brain data
    fmri_ses12 = load_brain_data(preprocessing=preprocessing, data_path=data_path)
    # import mask (and binarize, if specified)
    if bin_mask:
        binarize_mask(roi, data_path)
    mask_bin = data_path + roi + 'mask_bin.nii'  # import mask
    # import behavioral data
    behavioral = pd.read_csv(data_path + 'onsets_decoding_pilot01_ses12.csv', delimiter=';')
    # select only stimulus trials (in brain and behavioral data)
    conditions = behavioral['Condition']
    condition_mask = conditions.isin(['Negatif', 'Neutre'])  # index to restrict data to negative or neutral stimuli
    fmri_niimgs = index_img(fmri_ses12, condition_mask)
    conditions = conditions[condition_mask]
    conditions = conditions.values  # Convert to numpy array
    # plot results for checking
    if plot:
        print(fmri_ses12.shape)
        p1 = plotting.view_img(mean_img(fmri_ses12), threshold=None)
        p1.open_in_browser()
        plotting.plot_roi(mask_bin, bg_img=anat, cmap='Paired')
    return fmri_niimgs, anat, mask_bin, behavioral, condition_mask, conditions

def perform_decoding_cv(cv_type: str, anova: bool, conditions, fmri_niimgs, mask_bin, behavioral, condition_mask, random_state):
    # perform feature reduction via anova
    if anova:
        smoothing_fwhm = 8
        screening_percentile = 5
    else:
        smoothing_fwhm = None
        screening_percentile = 20
    # determine cv method
    if cv_type == 'k_fold':
        cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=random_state)
        scoring = 'accuracy'
        groups = None
    elif cv_type == 'session_out':
        cv = LeaveOneGroupOut()
        scoring = 'roc_auc'
        groups = behavioral['Session'][condition_mask]
    else:
        print('Input error "cv_type": Please indicate either as "k_fold" or as "session_out"')
        return
    # build decoder
    decoder = Decoder(estimator='svc', mask=mask_bin, cv=cv, screening_percentile=screening_percentile,
                      scoring=scoring, smoothing_fwhm=smoothing_fwhm, standardize=True)
    # fit decoder
    decoder.fit(fmri_niimgs, conditions, groups=groups)
    return decoder

def plot_weights(decoder, anat, data_path):
    # plot model weights
    coef_ = decoder.coef_
    print(coef_.shape)
    weigth_img = decoder.coef_img_['Negatif']
    weigth_img.to_filename(data_path + 'Pilote02_weights_ses12_wholebrain_c1_anova.nii.gz')
    plot_stat_map(weigth_img, bg_img=anat, title='SVM weights')
    p2 = plotting.view_img(weigth_img, bg_img=anat, title="SVM weights", dim=-1)
    p2.open_in_browser()
    return

# helper functions:
def load_brain_data(preprocessing, data_path):
    if preprocessing == 'normalize':
        print('todo: adjust for different batch options')
    else:
        # import brain data
        fmri_ses1 = data_path + 'ses1/' + 'swrBaseline_epi3mm_MB2_TE30_TR2000_IRMf_20211005094519_5.nii'
        fmri_ses2 = data_path + '/data/brain_data/Session_2_epi3mm_MB2_TE30_TR2000_IRMf_20210727114430_6.nii'
        anat = '/data/brain_data/wDicom_baseline_mprage_sag_T1_160sl_iPAT2_20210727114430_2.nii'
        fmri_ses12 = concat_imgs([fmri_ses1, fmri_ses2])  # concatenate brain data
    return fmri_ses12

def binarize_mask(roi, data_path):
    ##Create mask
    mask_neuroquery = data_path + roi + '_mask.nii'
    plotting.plot_roi(mask_neuroquery, bg_img=anat, cmap='Paired')
    mask_bin = math_img('img > 5', img=mask_neuroquery)
    plotting.plot_roi(mask_bin, bg_img=anat, cmap='Paired')
    mask_bin.to_filename(data_path + roi + '_mask_bin.nii')


# define path to project folder and main params
preprocessing = "normalize"
roi = "whole_brain"
project_path = "C:/Users/Jonas/PycharmProjects/fmri_decoding_quentin/decoding"
data_path = "C:/Private/Studium/Studium Leipzig/Praktika/2021/Paris Pauline Favre/Motor neurofeedback/data_pilot_02"
random_state = 8
# load data
conditions, fmri_niimgs, anat, mask_bin, behavioral, condition_mask = \
    load_data(preprocessing=preprocessing, roi=roi, data_path=data_path, bin_mask=False, plot=False)
# build and fit decoder in cv
decoder = \
    perform_decoding_cv(cv_type='k_fold', anova=False, conditions=conditions, fmri_niimgs=fmri_niimgs,
        mask_bin=mask_bin, behavioral=behavioral, condition_mask=condition_mask, random_state=random_state)
# evaluate decoder
print(np.mean(decoder.cv_scores_['Negatif']))
# plot decoder weights
plot_weights(decoder, anat, data_path)

# get univariate stats ###
stat_img = data_path + 'stats_baseline_event_subspace/spmT_0003.nii' # stat_img is just the name of the file that we downloaded
#print(stat_img)
#MNI_152 = "/usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz"
html_view = plotting.view_img(stat_img,bg_img=anat, threshold=4.36,symmetric_cmap=False, vmin=0,
                                     title="Negative > Neutral")
html_view.open_in_browser()
#html_view.save_as_html(os.path.join(directory,'viewer.html'))
