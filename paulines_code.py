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
#basepath = '/Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT'
fmri_ses1 = '/Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/baseline_session1/wrSession_1_epi3mm_MB2_TE30_TR2000_IRMf_20210727114430_4 4.nii'
fmri_ses2 = '/Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/baseline_session2/wrSession_2_epi3mm_MB2_TE30_TR2000_IRMf_20210727114430_6.nii'
anat = '/Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/T1/wDicom_baseline_mprage_sag_T1_160sl_iPAT2_20210727114430_2.nii'
fmri_ses12 = concat_imgs([fmri_ses1,fmri_ses2])
print(fmri_ses12.shape)
p1 = plotting.view_img(mean_img(fmri_ses12), threshold=None)
p1.open_in_browser()
##Create mask
mask_neuroquery = '/Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/RESULTAT_ALL_SESSION/beta_0001.nii'
plotting.plot_roi(mask_neuroquery, bg_img=anat, cmap='Paired')
mask_bin = math_img('img > 5', img=mask_neuroquery)
plotting.plot_roi(mask_bin, bg_img=anat, cmap='Paired')
#mask_bin.to_filename(basepath+'mask_bin.nii')
##Load behavioral data
behavioral = pd.read_csv('/Users/quentingallet/Downloads/onsets_decoding_pilot01_ses12.csv', delimiter=';')
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
## Decoding with SVM ###
decoder = Decoder(estimator='svc', mask=mask_bin, standardize=True)
decoder.fit(fmri_niimgs, conditions)
prediction = decoder.predict(fmri_niimgs)
print(prediction)
print((prediction == conditions).sum() / float(len(conditions)))
##Manually leaving out data
#Leave out the 60 last data points during training, and test the prediction on these 60 last points
fmri_niimgs_train = index_img(fmri_niimgs, slice(0, -60))
fmri_niimgs_test = index_img(fmri_niimgs, slice(-60, None))
conditions_train = conditions[:-60]
conditions_test = conditions[-60:]
#decoder.fit(fmri_niimgs_train, conditions_train)
decoder = Decoder(estimator='svc', mask=mask_bin, smoothing_fwhm=4,
                  standardize=True, screening_percentile=10, scoring='accuracy')
decoder.fit(fmri_niimgs_train, conditions_train)
y_pred = decoder.predict(fmri_niimgs_train)
prediction = decoder.predict(fmri_niimgs_test)
print("Prediction Accuracy: {:.3f}".format(
    (prediction == conditions_test).sum() / float(len(conditions_test))))
#Kfold cross-validation
cv = KFold(n_splits=5)
fold = 0
for train, test in cv.split(conditions):
    fold += 1
    decoder = Decoder(estimator='svc', mask=mask_bin, standardize=True)
    decoder.fit(index_img(fmri_niimgs, train), conditions[train])
    prediction = decoder.predict(index_img(fmri_niimgs, test))
    print("CV Fold {:01d} | Prediction Accuracy: {:.3f}".format(fold,
            (prediction == conditions[test]).sum() / float(len(conditions[test]))))
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
##Leave-one-session-out
session_label = behavioral['Session'][condition_mask]
cv = LeaveOneGroupOut()
decoder = Decoder(estimator='svc', mask=mask_bin, standardize=True, cv=cv)
decoder.fit(fmri_niimgs, conditions, groups=session_label)
print(decoder.cv_scores_)
print(np.mean(decoder.cv_scores_['Negatif']))
##Model weights
coef_ = decoder.coef_
print(coef_)
print(coef_.shape)
weigth_img = decoder.coef_img_['Negatif']
weigth_img.to_filename('pilot01_svc_weights_ses12.nii.gz')
plot_stat_map(weigth_img, bg_img=anat, title='SVM weights')
p2 = plotting.view_img(weigth_img, bg_img=anat, title="SVM weights", dim=-1)
p2.open_in_browser()
# ##Does the model perform better than chance?
# dummy_decoder = Decoder(estimator='dummy_classifier', mask=mask_bin, cv=cv) #Error: dummy_classifier unknown
# dummy_decoder.fit(fmri_niimgs, conditions, groups=session_label)
# print(dummy_decoder.cv_scores_)
# from nilearn.decoding import Decoder
# # Here screening_percentile is set to 5 percent
# mask_img = '/Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/RESULTAT_ALL_SESSION/mask.nii'
# decoder = Decoder(estimator='svc', mask=mask_img, smoothing_fwhm=4,
#                   standardize=True, screening_percentile=5, scoring='accuracy')
# #############################################################################
# # Fit the decoder and predict
# # ----------------------------
# decoder.fit(fmri_niimgs,conditions)
# y_pred = decoder.predict(fmri_niimgs)
# #############################################################################
# # Obtain prediction scores via cross validation
# # -----------------------------------------------
# # Define the cross-validation scheme used for validation. Here we use a
# # LeaveOneGroupOut cross-validation on the session group which corresponds to a
# # leave a session out scheme, then pass the cross-validator object to the cv
# # parameter of decoder.leave-one-session-out For more details please take a
# # look at:
# # <https://nilearn.github.io/auto_examples/plot_decoding_tutorial.html#measuring-prediction-scores-using-cross-validation>
# from sklearn.model_selection import LeaveOneGroupOut
# cv = LeaveOneGroupOut()
# decoder = Decoder(estimator='svc', mask=mask_img, standardize=True,
#                   screening_percentile=5, scoring='accuracy', cv=cv)
# # Compute the prediction accuracy for the different folds (i.e. session)
# # decoder.fit(fmri_niimgs, conditions)
# # Print the CV scores
# print(decoder.cv_scores_['Negatif'])
# decoder.fit(fmri_niimgs, conditions)
# print(decoder.cv_scores_)
# print(decoder.cv_params_['Negatif'])
# print(np.mean(decoder.cv_scores_['Negatif']))
#############################################################################