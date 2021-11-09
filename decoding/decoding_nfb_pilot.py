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
def load_data(preprocessing: str, roi: str, fname_fmri_ses1: str, fname_fmri_ses2: str, fname_t1: str, data_path: str, bin_mask: bool=False, plot: bool=False):
    # load brain data
    fmri_ses12, anat = load_brain_data(preprocessing, fname_fmri_ses1, fname_fmri_ses2, fname_t1, data_path)
    # import mask (and binarize, if specified)
    if bin_mask:
        binarize_mask(roi, data_path)
    mask_bin = data_path + 'masks_bin/' + roi + 'mask_bin.nii'  # set path to binarized mask
    # import behavioral data
    behavioral = pd.read_csv(data_path + 'conditions/data_behav.csv', delimiter=';')  # todo: automatize reading data from excel file
    # select only stimulus trials (in brain and behavioral data)
    conditions_all = behavioral['condition']
    condition_mask = conditions_all.isin(['regulation', 'neutre'])  # index to restrict data to negative or neutral stimuli todo: unify names
    fmri_niimgs = index_img(fmri_ses12, condition_mask)
    conditions_trials = conditions_all[condition_mask]
    conditions = conditions_trials.values  # Convert to numpy array
    # plot results for checking
    if plot:
        print(fmri_ses12.shape)
        p1 = plotting.view_img(mean_img(fmri_ses12), threshold=None)  # todo: sth. wrong with brain data?
        p1.open_in_browser()
        plotting.plot_roi(mask_bin, bg_img=anat, cmap='Paired')
    return fmri_niimgs, anat, mask_bin, behavioral, condition_mask, conditions

def load_brain_data(preprocessing, fname_fmri_ses1, fname_fmri_ses2, fname_t1, data_path):
    # enter preprocessing arg as 'r', 'sr', or 'swr
    # import brain data
    fmri_ses1 = data_path + 'ses1/' + preprocessing + fname_fmri_ses1
    fmri_ses2 = data_path + 'ses2/' + preprocessing + fname_fmri_ses2
    anat = data_path + 'T1/' + fname_t1
    fmri_ses12 = concat_imgs([fmri_ses1, fmri_ses2])  # concatenate brain data
    return fmri_ses12, anat

def binarize_mask(roi, data_path):
    ##Create mask
    mask_raw = data_path + 'masks/' + roi + '_mask.nii'
    plotting.plot_roi(mask_raw, bg_img=anat, cmap='Paired')
    mask_bin = math_img('img > 5', img=mask_raw)
    plotting.plot_roi(mask_bin, bg_img=anat, cmap='Paired')
    mask_bin.to_filename(data_path + roi + '_mask_bin.nii')
    return

def perform_decoding_cv(conditions, fmri_niimgs, mask_bin, behavioral, condition_mask, random_state, cv_type: str, anova: bool):
    # perform feature reduction via anova
    if anova:
        smoothing_fwhm = 8
        screening_percentile = 5
    else:
        smoothing_fwhm = None
        screening_percentile = 20
    # determine cv method
    if cv_type == 'k_fold':
        cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=random_state)  # todo: add n_splits and n_repeats as input options?
        scoring = 'accuracy'
        groups = None
    elif cv_type == 'session_out':
        cv = LeaveOneGroupOut()
        scoring = 'roc_auc'
        groups = behavioral['session'][condition_mask]  # todo: change from session to block
    else:
        print('Input error "cv_type": Please indicate either as "k_fold" or as "session_out"')
        return
    # build decoder
    decoder = Decoder(estimator='svc', mask=mask_bin, cv=cv, screening_percentile=screening_percentile,
                      scoring=scoring, smoothing_fwhm=smoothing_fwhm, standardize=True) # todo: discuss settings with Pauline
    # fit decoder
    decoder.fit(fmri_niimgs, conditions, groups=groups)
    return decoder

def plot_weights(decoder, anat):
    # plot model weights
    coef_ = decoder.coef_
    print(coef_.shape)
    weigth_img = decoder.coef_img_['regulation']
    plot_stat_map(weigth_img, bg_img=anat, title='SVM weights')
    p2 = plotting.view_img(weigth_img, bg_img=anat, title="SVM weights", dim=-1)
    p2.open_in_browser()
    return


# define path to project folder and main params
preprocessing = "swr"  # specify as 'r' (realigned), 'sr' (realigned + smoothed), or 'swr' (sr + normalization)
roi = "bin_iw_sma_nquery_"  # specify name of mask in data folder (e.g., 'whole_brain' or 'emot_reg') # todo: check which masks are working
project_path = "C:/Users/Jonas/PycharmProjects/fmri_decoding_quentin/decoding/"
data_path = project_path + "data/pilot_02/"  # set path to data folder of current set
fname_fmri_ses1 = "Baseline_epi3mm_MB2_TE30_TR2000_IRMf_20211005094519_5.nii"  # enter raw name of frmi baseline scan of session
fname_fmri_ses2 = "Baseline_epi3mm_MB2_TE30_TR2000_IRMf_20211005094519_7.nii"  # same for ses 2
fname_t1 = "20211005.Test_JH_05102021.Test_JH_05102021_mprage_sag_T1_160sl_iPAT2_20211005094519_2.nii"
random_state = 8


# load data
fmri_niimgs, anat, mask_bin, behavioral, condition_mask, conditions = \
    load_data(preprocessing, roi, fname_fmri_ses1, fname_fmri_ses2, fname_t1, data_path, bin_mask=False, plot=True)

# build and fit decoder in cv
decoder = perform_decoding_cv(conditions, fmri_niimgs, mask_bin, behavioral,
                              condition_mask, random_state, cv_type='k_fold', anova=False)
# evaluate decoder
print(np.mean(decoder.cv_scores_['regulation']))

# plot decoder weights
plot_weights(decoder, anat)

# save decoder weights
weigth_img = decoder.coef_img_['regulation']
weigth_img.to_filename(data_path + 'weights/weights_' + preprocessing + '_' + roi + '.nii.gz')



""" DEV PART """
# get univariate stats
stat_img = data_path + 'stats_baseline_event_subspace/spmT_0003.nii' # stat_img is just the name of the file that we downloaded
#print(stat_img)
#MNI_152 = "/usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz"
html_view = plotting.view_img(stat_img, bg_img=anat, threshold=4.36, symmetric_cmap=False, vmin=0,
                                     title="Negative > Neutral")
html_view.open_in_browser()
#html_view.save_as_html(os.path.join(directory,'viewer.html'))

# directly read onsets and conditions from text file
onsets = pd.read_csv(data_path + 'conditions/Stimuli_NF_BD_Baseline_pilot02_05-Oct-2021.txt', delimiter='\t', index_col=False, skiprows=4)
onsets_TR = onsets.loc[:, ['condition', 'onsets_seconds']]
onsets_TR['']
