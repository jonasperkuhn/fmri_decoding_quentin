#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 12:38:34 2021

@author: paulinefavre
"""
import pandas as pd
import numpy as np
from nilearn import plotting
from nilearn.image import mean_img
from nilearn.image import math_img
from nilearn.image import concat_imgs, index_img
from nilearn.decoding import Decoder
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneGroupOut
from nilearn.plotting import plot_stat_map



### Import data pilot01 ###
basepath = '/Users/paulinefavre/Neurofeedback/Pilote02/'
fmri_ses1 = basepath + 'ses1/' + 'swrBaseline_epi3mm_MB2_TE30_TR2000_IRMf_20211005094519_5.nii' #data normalisées ?? #et smoothées ??? 
fmri_ses2 = basepath + 'ses2/' + 'swrBaseline_epi3mm_MB2_TE30_TR2000_IRMf_20211005094519_7.nii'
anat = basepath + 'T1/' + 'w20211005.Test_JH_05102021.Test_JH_05102021_mprage_sag_T1_160sl_iPAT2_20211005094519_2.nii'

fmri_ses12 = concat_imgs([fmri_ses1,fmri_ses2])
print(fmri_ses12.shape)
p1 = plotting.view_img(mean_img(fmri_ses12), threshold=None)
p1.open_in_browser()

# ##Create mask
# mask_neuroquery = '/Users/paulinefavre/Neurofeedback/emotion_regulation.nii'
# plotting.plot_roi(mask_neuroquery, bg_img=anat, cmap='Paired')
# mask_bin = math_img('img > 5', img=mask_neuroquery)
# plotting.plot_roi(mask_bin, bg_img=anat, cmap='Paired')
# mask_bin.to_filename(basepath+'mask_bin.nii')
mask_bin = '/Users/paulinefavre/Neurofeedback/masks_neuroquery/emotion_regulation_bin.nii'
plotting.plot_roi(mask_bin, bg_img=anat, cmap='Paired')

# # Import mask in subject's space
# iw_mask_bin = '/Users/paulinefavre/Neurofeedback/Pilote02/iw_p2_emotion_regulation_bin.nii' #Rentrer ici le mask dans l'espace du sujet 
# plotting.plot_roi(iw_mask_bin, bg_img=anat, cmap='Paired')

##Load behavioral data
behavioral = pd.read_csv('/Users/paulinefavre/Neurofeedback/onsets_decoding_ses12.csv', delimiter=';')
print(behavioral)
conditions = behavioral['Condition']
print(conditions)
condition_mask = conditions.isin(['Negatif', 'Neutre'])

#Restrcit analysis to negative and neutral stimuli
fmri_niimgs = index_img(fmri_ses12, condition_mask) 
print(fmri_niimgs.shape)
conditions = conditions[condition_mask]
conditions = conditions.values # Convert to numpy array
print(conditions.shape)


### Decoding with SVM ###
# decoder = Decoder(estimator='svc', mask=mask_bin, standardize=True)
# decoder.fit(fmri_niimgs, conditions)
# prediction = decoder.predict(fmri_niimgs)
# print(prediction)
# print((prediction == conditions).sum() / float(len(conditions)))

# ##Manually leaving out data
# #Leave out the 60 last data points during training, and test the prediction on these 60 last points
# fmri_niimgs_train = index_img(fmri_niimgs, slice(0, -60))
# fmri_niimgs_test = index_img(fmri_niimgs, slice(-60, None))
# conditions_train = conditions[:-60]
# conditions_test = conditions[-60:]

# decoder.fit(fmri_niimgs_train, conditions_train)
# prediction = decoder.predict(fmri_niimgs_test)
# print("Prediction Accuracy: {:.3f}".format(
#     (prediction == conditions_test).sum() / float(len(conditions_test))))

# ##Kfold cross-validation
# cv = KFold(n_splits=5)
# fold = 0
# for train, test in cv.split(conditions):
#     fold += 1
#     decoder = Decoder(estimator='svc', mask=mask_bin, standardize=True)
#     decoder.fit(index_img(fmri_niimgs, train), conditions[train])
#     prediction = decoder.predict(index_img(fmri_niimgs, test))
#     print("CV Fold {:01d} | Prediction Accuracy: {:.3f}".format(fold,
#             (prediction == conditions[test]).sum() / float(len(conditions[test]))))

##Cross-validation with the decoder
n_folds = 5
decoder = Decoder(
    estimator='svc', mask=mask_bin,
    standardize=True, cv=n_folds,
    scoring='accuracy'
)
decoder.fit(fmri_niimgs, conditions)
print(decoder.cv_scores_)
print(decoder.cv_params_['Negatif'])
print(np.mean(decoder.cv_scores_['Negatif']))

# ##Leave-one-session-out
# session_label = behavioral['Session'][condition_mask]
# cv = LeaveOneGroupOut()
# decoder = Decoder(estimator='svc', mask=mask_bin, standardize=True, cv=cv)
# decoder.fit(fmri_niimgs, conditions, groups=session_label)
# print(decoder.cv_scores_)
# print(np.mean(decoder.cv_scores_['Negatif']))
 
##Model weights
coef_ = decoder.coef_
print(coef_)
print(coef_.shape)
weigth_img = decoder.coef_img_['Negatif']
weigth_img.to_filename(basepath+'Pilote02_svc_weights_ses12.nii.gz')
plot_stat_map(weigth_img, bg_img=anat, title='SVM weights')
p2 = plotting.view_img(weigth_img, bg_img=anat, title="SVM weights", dim=-1)
p2.open_in_browser()

# ##Does the model perform better than chance?
# dummy_decoder = Decoder(estimator='dummy_classifier', mask=mask_bin, cv=cv) #Error: dummy_classifier unknown
# dummy_decoder.fit(fmri_niimgs, conditions, groups=session_label)
# print(dummy_decoder.cv_scores_)




### Trial with ANOVA ####
# Load realigned data
fmri_ses1 = basepath + 'ses1/' + 'rBaseline_epi3mm_MB2_TE30_TR2000_IRMf_20211005094519_5.nii' #data normalisées ?? #et smoothées ??? 
fmri_ses2 = basepath + 'ses2/' + 'rBaseline_epi3mm_MB2_TE30_TR2000_IRMf_20211005094519_7.nii'
anat = basepath + 'T1/' + '20211005.Test_JH_05102021.Test_JH_05102021_mprage_sag_T1_160sl_iPAT2_20211005094519_2.nii'
fmri_ses12 = concat_imgs([fmri_ses1,fmri_ses2])
fmri_niimgs = index_img(fmri_ses12, condition_mask) 

# Create whole_brain GM mask
mask_GM = '/Users/paulinefavre/Neurofeedback/Pilote02/T1/rc120211005.Test_JH_05102021.Test_JH_05102021_mprage_sag_T1_160sl_iPAT2_20211005094519_2.nii'
mask_bin_GM = math_img('img > 0.5', img=mask_GM)
# #resample image
# from nilearn.image import resample_img
# rmask_bin_GM = resample_img(mask_bin_GM, target_shape=np.eye(3))
# plotting.plot_roi(rmask_bin_GM, bg_img=anat, cmap='Paired')
# rmask_bin_GM.to_filename(basepath+'rmask_bin_GM.nii')

decoder = Decoder(estimator='svc', mask=mask_bin_GM,smoothing_fwhm=8,cv=5,   
                  standardize=True, screening_percentile=5, scoring='accuracy')

decoder.fit(fmri_niimgs, conditions)

print(decoder.cv_scores_)
print(decoder.cv_params_['Negatif'])
print(np.mean(decoder.cv_scores_['Negatif']))

##Model weights
coef_ = decoder.coef_
print(coef_.shape)
weigth_img = decoder.coef_img_['Negatif']
weigth_img.to_filename(basepath+'Pilote02_weights_ses12_wholebrain_c1_anova.nii.gz')
plot_stat_map(weigth_img, bg_img=anat, title='SVM weights')
p2 = plotting.view_img(weigth_img, bg_img=anat, title="SVM weights", dim=-1)
p2.open_in_browser()


### Univariate stats ###
import os
directory='/Users/paulinefavre/Neurofeedback/Pilote02/stats_baseline_event_subspace'
stat_img = os.path.join(directory,'spmT_0003.nii')
# stat_img is just the name of the file that we downloded
print(stat_img)
#MNI_152 = "/usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz"
html_view = plotting.view_img(stat_img,bg_img=anat, threshold=4.36,symmetric_cmap=False, vmin=0,
                                     title="Negative > Neutral")
html_view.open_in_browser()   
#html_view.save_as_html(os.path.join(directory,'viewer.html'))








