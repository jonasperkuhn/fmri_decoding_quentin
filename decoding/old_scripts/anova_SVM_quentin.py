#################################################################
#################################################################
##########################    PILOT 1   #########################
##########################  SVM AVOVA   #########################
#################################################################


##################### IMPORT DES FONCTIONS #########################

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
from nilearn.plotting import view_img

##################### IMPORT BEHAVIORAL DATA ########################

behavioral = pd.read_csv('/Users/quentingallet/Downloads/onsets_decoding_pilot01_ses12.csv', delimiter=';')
conditions = behavioral['Condition']
condition_mask = behavioral['Condition'].isin(['Neutre', 'Negatif'])
conditions = conditions[condition_mask]
print(conditions.unique())

#Nombre de session
session_label = behavioral['Session'][condition_mask]

#################### ANALYSE SR, MASQUES WHOLE BRAIN ####################################

fmri_ses1 = '//Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/baseline_session1/srSession_1_epi3mm_MB2_TE30_TR2000_IRMf_20210727114430_4.nii'
fmri_ses2 = '/Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/baseline_session2/srSession_2_epi3mm_MB2_TE30_TR2000_IRMf_20210727114430_6.nii'
anat = '/Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/T1/wDicom_baseline_mprage_sag_T1_160sl_iPAT2_20210727114430_2.nii'



# Concaténation des images fonctionnels

fmri_ses12 = concat_imgs([fmri_ses1,fmri_ses2])
func_img = index_img(fmri_ses12, condition_mask)


# IMPORT DU MASQUE VOIR BINARISATION

mask = '/Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/mask_neuroquery/mask.nii'

plotting.plot_roi(mask, bg_img=anat, cmap='Paired')

##############################ANOVA#################################

mask_img = mask
decoder = Decoder(estimator='svc', mask=mask_img, smoothing_fwhm=4,
                  standardize=True, screening_percentile=5, scoring='accuracy', cv=5)

############# FIT AVEC LE DECODER ET PREDICTION###########

decoder.fit(func_img, conditions)
y_pred = decoder.predict(func_img)
prediction = decoder.predict(func_img)


print(prediction)
print((prediction == conditions).sum() / float(len(conditions)))

print(decoder.cv_scores_)
print(decoder.cv_params_['Negatif'])
print(np.mean(decoder.cv_scores_['Negatif']))


########### CV AVEC LEAVE ONE GROUP OUT################################

cv = LeaveOneGroupOut()

decoder = Decoder(estimator='svc', mask=mask_img, standardize=True,
                  screening_percentile=5, scoring='accuracy', cv=cv,)

# Compute the prediction accuracy for the different folds (i.e. session)
decoder.fit(func_img, conditions, groups=session_label)

# Print the CV scores
print(decoder.cv_scores_['Negatif'])
print(decoder.cv_scores_)
print(np.mean(decoder.cv_scores_['Negatif']))
################ VISUALISATION DE LA CARTE DE POIDS#######################

weight_img = decoder.coef_img_['Negatif']
plot_stat_map(weight_img, bg_img=anat, title='SVM weights')


#VUE SUR INTERNET
view_img(weight_img, bg_img=anat,
         title="SVM weights", dim=-1)

#SAVE DE LA CARTE DE POIDS 

#weight_img.to_filename('WM_ANOVA_WB')

#################### ANALYSE SR, MASQUES EMOTION REGULATION ####################################

fmri_ses1 = '//Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/baseline_session1/srSession_1_epi3mm_MB2_TE30_TR2000_IRMf_20210727114430_4.nii'
fmri_ses2 = '/Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/baseline_session2/srSession_2_epi3mm_MB2_TE30_TR2000_IRMf_20210727114430_6.nii'
anat = '/Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/T1/wDicom_baseline_mprage_sag_T1_160sl_iPAT2_20210727114430_2.nii'



# Concaténation des images fonctionnels

fmri_ses12 = concat_imgs([fmri_ses1,fmri_ses2])
func_img = index_img(fmri_ses12, condition_mask)


# IMPORT DU MASQUE VOIR BINARISATION

#mask = '/Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/mask_neuroquery/mask.nii'
mask = '/Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/mask_neuroquery/wmask_emotion_reg_bin.nii'

plotting.plot_roi(mask, bg_img=anat, cmap='Paired')

##############################ANOVA#################################

mask_img = mask
decoder = Decoder(estimator='svc', mask=mask_img, smoothing_fwhm=4,
                  standardize=True, screening_percentile=5, scoring='accuracy', cv=5)

############# FIT AVEC LE DECODER ET PREDICTION###########

decoder.fit(func_img, conditions)
y_pred = decoder.predict(func_img)
prediction = decoder.predict(func_img)


print(prediction)
print((prediction == conditions).sum() / float(len(conditions)))
print(decoder.cv_scores_)
print(decoder.cv_params_['Negatif'])
print(np.mean(decoder.cv_scores_['Negatif']))

########### CV AVEC LEAVE ONE GROUP OUT################################

cv = LeaveOneGroupOut()

decoder = Decoder(estimator='svc', mask=mask_img, standardize=True,
                  screening_percentile=5, scoring='accuracy', cv=cv,)

# Compute the prediction accuracy for the different folds (i.e. session)
decoder.fit(func_img, conditions, groups=session_label)

# Print the CV scores
print(decoder.cv_scores_['Negatif'])
print(decoder.cv_scores_)
print(np.mean(decoder.cv_scores_['Negatif']))
################ VISUALISATION DE LA CARTE DE POIDS#######################

weight_img = decoder.coef_img_['Negatif']
plot_stat_map(weight_img, bg_img=anat, title='SVM weights')


#VUE SUR INTERNET
view_img(weight_img, bg_img=anat,
         title="SVM weights", dim=-1)

#SAVE DE LA CARTE DE POIDS 

#weight_img.to_filename('WM_ANOVA_WB')

#################### ANALYSE SR, MASQUES EMOTION ####################################

fmri_ses1 = '//Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/baseline_session1/srSession_1_epi3mm_MB2_TE30_TR2000_IRMf_20210727114430_4.nii'
fmri_ses2 = '/Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/baseline_session2/srSession_2_epi3mm_MB2_TE30_TR2000_IRMf_20210727114430_6.nii'
anat = '/Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/T1/wDicom_baseline_mprage_sag_T1_160sl_iPAT2_20210727114430_2.nii'



# Concaténation des images fonctionnels

fmri_ses12 = concat_imgs([fmri_ses1,fmri_ses2])
func_img = index_img(fmri_ses12, condition_mask)


# IMPORT DU MASQUE VOIR BINARISATION

mask = '/Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/mask_neuroquery/wmask_emotion_bin2.nii'

plotting.plot_roi(mask, bg_img=anat, cmap='Paired')

##############################ANOVA#################################

mask_img = mask
decoder = Decoder(estimator='svc', mask=mask_img, smoothing_fwhm=4,
                  standardize=True, screening_percentile=5, scoring='accuracy', cv=5)

############# FIT AVEC LE DECODER ET PREDICTION###########

decoder.fit(func_img, conditions)
y_pred = decoder.predict(func_img)
prediction = decoder.predict(func_img)


print(prediction)
print((prediction == conditions).sum() / float(len(conditions)))


print(decoder.cv_scores_)
print(decoder.cv_params_['Negatif'])
print(np.mean(decoder.cv_scores_['Negatif']))


########### CV AVEC LEAVE ONE GROUP OUT################################

cv = LeaveOneGroupOut()

decoder = Decoder(estimator='svc', mask=mask_img, standardize=True,
                  screening_percentile=5, scoring='accuracy', cv=cv,)

# Compute the prediction accuracy for the different folds (i.e. session)
decoder.fit(func_img, conditions, groups=session_label)

# Print the CV scores
print(decoder.cv_scores_['Negatif'])
print(decoder.cv_scores_)
print(np.mean(decoder.cv_scores_['Negatif']))
################ VISUALISATION DE LA CARTE DE POIDS#######################

weight_img = decoder.coef_img_['Negatif']
plot_stat_map(weight_img, bg_img=anat, title='SVM weights')


#VUE SUR INTERNET
view_img(weight_img, bg_img=anat,
         title="SVM weights", dim=-1)

#SAVE DE LA CARTE DE POIDS 

#weight_img.to_filename('WM_ANOVA_WB')

#################### ANALYSE SR, MASQUES Amydgale ####################################

fmri_ses1 = '//Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/baseline_session1/srSession_1_epi3mm_MB2_TE30_TR2000_IRMf_20210727114430_4.nii'
fmri_ses2 = '/Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/baseline_session2/srSession_2_epi3mm_MB2_TE30_TR2000_IRMf_20210727114430_6.nii'
anat = '/Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/T1/wDicom_baseline_mprage_sag_T1_160sl_iPAT2_20210727114430_2.nii'



# Concaténation des images fonctionnels

fmri_ses12 = concat_imgs([fmri_ses1,fmri_ses2])
func_img = index_img(fmri_ses12, condition_mask)


# IMPORT DU MASQUE VOIR BINARISATION

mask = '/Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/mask_neuroquery/wmask_amygdala_bin.nii'

plotting.plot_roi(mask, bg_img=anat, cmap='Paired')

##############################ANOVA#################################

mask_img = mask
decoder = Decoder(estimator='svc', mask=mask_img, smoothing_fwhm=4,
                  standardize=True, screening_percentile=5, scoring='accuracy', cv=5)

############# FIT AVEC LE DECODER ET PREDICTION###########

decoder.fit(func_img, conditions)
y_pred = decoder.predict(func_img)
prediction = decoder.predict(func_img)


print(prediction)
print((prediction == conditions).sum() / float(len(conditions)))


print(decoder.cv_scores_)
print(decoder.cv_params_['Negatif'])
print(np.mean(decoder.cv_scores_['Negatif']))


########### CV AVEC LEAVE ONE GROUP OUT################################

cv = LeaveOneGroupOut()

decoder = Decoder(estimator='svc', mask=mask_img, standardize=True,
                  screening_percentile=5, scoring='accuracy', cv=cv,)

# Compute the prediction accuracy for the different folds (i.e. session)
decoder.fit(func_img, conditions, groups=session_label)

# Print the CV scores
print(decoder.cv_scores_['Negatif'])
print(decoder.cv_scores_)
print(np.mean(decoder.cv_scores_['Negatif']))
################ VISUALISATION DE LA CARTE DE POIDS#######################

weight_img = decoder.coef_img_['Negatif']
plot_stat_map(weight_img, bg_img=anat, title='SVM weights')


#VUE SUR INTERNET
view_img(weight_img, bg_img=anat,
         title="SVM weights", dim=-1)

#SAVE DE LA CARTE DE POIDS 

#weight_img.to_filename('WM_ANOVA_WB')


#################### ANALYSE SWR, MASQUES WHOLE BRAIN ####################################

fmri_ses1 = '//Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/baseline_session1/swrSession_1_epi3mm_MB2_TE30_TR2000_IRMf_20210727114430_4.nii'
fmri_ses2 = '/Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/baseline_session2/swrSession_2_epi3mm_MB2_TE30_TR2000_IRMf_20210727114430_6.nii'
anat = '/Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/T1/wDicom_baseline_mprage_sag_T1_160sl_iPAT2_20210727114430_2.nii'



# Concaténation des images fonctionnels

fmri_ses12 = concat_imgs([fmri_ses1,fmri_ses2])
func_img = index_img(fmri_ses12, condition_mask)


# IMPORT DU MASQUE VOIR BINARISATION

mask = '/Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/mask_neuroquery/mask.nii'

plotting.plot_roi(mask, bg_img=anat, cmap='Paired')

##############################ANOVA#################################

mask_img = mask
decoder = Decoder(estimator='svc', mask=mask_img, smoothing_fwhm=4,
                  standardize=True, screening_percentile=5, scoring='accuracy', cv=5)

############# FIT AVEC LE DECODER ET PREDICTION###########

decoder.fit(func_img, conditions)
y_pred = decoder.predict(func_img)
prediction = decoder.predict(func_img)


print(prediction)
print((prediction == conditions).sum() / float(len(conditions)))


print(decoder.cv_scores_)
print(decoder.cv_params_['Negatif'])
print(np.mean(decoder.cv_scores_['Negatif']))


########### CV AVEC LEAVE ONE GROUP OUT################################

cv = LeaveOneGroupOut()

decoder = Decoder(estimator='svc', mask=mask_img, standardize=True,
                  screening_percentile=5, scoring='accuracy', cv=cv,)

# Compute the prediction accuracy for the different folds (i.e. session)
decoder.fit(func_img, conditions, groups=session_label)

# Print the CV scores
print(decoder.cv_scores_['Negatif'])
print(decoder.cv_scores_)
print(np.mean(decoder.cv_scores_['Negatif']))
################ VISUALISATION DE LA CARTE DE POIDS#######################

weight_img = decoder.coef_img_['Negatif']
plot_stat_map(weight_img, bg_img=anat, title='SVM weights')


#VUE SUR INTERNET
view_img(weight_img, bg_img=anat,
         title="SVM weights", dim=-1)

#SAVE DE LA CARTE DE POIDS 

#weight_img.to_filename('WM_ANOVA_WB')

#################### ANALYSE SWR, MASQUES EMOTION REGULATION ####################################

fmri_ses1 = '//Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/baseline_session1/swrSession_1_epi3mm_MB2_TE30_TR2000_IRMf_20210727114430_4.nii'
fmri_ses2 = '/Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/baseline_session2/swrSession_2_epi3mm_MB2_TE30_TR2000_IRMf_20210727114430_6.nii'
anat = '/Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/T1/wDicom_baseline_mprage_sag_T1_160sl_iPAT2_20210727114430_2.nii'



# Concaténation des images fonctionnels

fmri_ses12 = concat_imgs([fmri_ses1,fmri_ses2])
func_img = index_img(fmri_ses12, condition_mask)


# IMPORT DU MASQUE VOIR BINARISATION

mask = '/Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/mask_neuroquery/mask_emotion_reg_bin.nii'

plotting.plot_roi(mask, bg_img=anat, cmap='Paired')

##############################ANOVA#################################

mask_img = mask
decoder = Decoder(estimator='svc', mask=mask_img, smoothing_fwhm=4,
                  standardize=True, screening_percentile=5, scoring='accuracy', cv=5)

############# FIT AVEC LE DECODER ET PREDICTION###########

decoder.fit(func_img, conditions)
y_pred = decoder.predict(func_img)
prediction = decoder.predict(func_img)


print(prediction)
print((prediction == conditions).sum() / float(len(conditions)))


print(decoder.cv_scores_)
print(decoder.cv_params_['Negatif'])
print(np.mean(decoder.cv_scores_['Negatif']))


########### CV AVEC LEAVE ONE GROUP OUT################################

cv = LeaveOneGroupOut()

decoder = Decoder(estimator='svc', mask=mask_img, standardize=True,
                  screening_percentile=5, scoring='accuracy', cv=cv,)

# Compute the prediction accuracy for the different folds (i.e. session)
decoder.fit(func_img, conditions, groups=session_label)

# Print the CV scores
print(decoder.cv_scores_['Negatif'])
print(decoder.cv_scores_)
print(np.mean(decoder.cv_scores_['Negatif']))
################ VISUALISATION DE LA CARTE DE POIDS#######################

weight_img = decoder.coef_img_['Negatif']
plot_stat_map(weight_img, bg_img=anat, title='SVM weights')


#VUE SUR INTERNET
view_img(weight_img, bg_img=anat,
         title="SVM weights", dim=-1)

#SAVE DE LA CARTE DE POIDS 

#weight_img.to_filename('WM_ANOVA_WB')

#################### ANALYSE SWR, MASQUES EMOTION ####################################

fmri_ses1 = '//Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/baseline_session1/swrSession_1_epi3mm_MB2_TE30_TR2000_IRMf_20210727114430_4.nii'
fmri_ses2 = '/Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/baseline_session2/swrSession_2_epi3mm_MB2_TE30_TR2000_IRMf_20210727114430_6.nii'
anat = '/Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/T1/wDicom_baseline_mprage_sag_T1_160sl_iPAT2_20210727114430_2.nii'



# Concaténation des images fonctionnels

fmri_ses12 = concat_imgs([fmri_ses1,fmri_ses2])
func_img = index_img(fmri_ses12, condition_mask)


# IMPORT DU MASQUE VOIR BINARISATION

mask = '/Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/mask_neuroquery/mask_emotion_bin2.nii'

plotting.plot_roi(mask, bg_img=anat, cmap='Paired')

##############################ANOVA#################################

mask_img = mask
decoder = Decoder(estimator='svc', mask=mask_img, smoothing_fwhm=4,
                  standardize=True, screening_percentile=5, scoring='accuracy', cv=5)

############# FIT AVEC LE DECODER ET PREDICTION###########

decoder.fit(func_img, conditions)
y_pred = decoder.predict(func_img)
prediction = decoder.predict(func_img)


print(prediction)
print((prediction == conditions).sum() / float(len(conditions)))

print(decoder.cv_scores_)
print(decoder.cv_params_['Negatif'])
print(np.mean(decoder.cv_scores_['Negatif']))


########### CV AVEC LEAVE ONE GROUP OUT################################

cv = LeaveOneGroupOut()

decoder = Decoder(estimator='svc', mask=mask_img, standardize=True,
                  screening_percentile=5, scoring='accuracy', cv=cv,)

# Compute the prediction accuracy for the different folds (i.e. session)
decoder.fit(func_img, conditions, groups=session_label)

# Print the CV scores
print(decoder.cv_scores_['Negatif'])
print(decoder.cv_scores_)
print(np.mean(decoder.cv_scores_['Negatif']))
################ VISUALISATION DE LA CARTE DE POIDS#######################

weight_img = decoder.coef_img_['Negatif']
plot_stat_map(weight_img, bg_img=anat, title='SVM weights')


#VUE SUR INTERNET
view_img(weight_img, bg_img=anat,
         title="SVM weights", dim=-1)

#SAVE DE LA CARTE DE POIDS 

#weight_img.to_filename('WM_ANOVA_WB')

#################### ANALYSE SWR, MASQUES Amydgale ####################################

fmri_ses1 = '//Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/baseline_session1/swrSession_1_epi3mm_MB2_TE30_TR2000_IRMf_20210727114430_4.nii'
fmri_ses2 = '/Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/baseline_session2/swrSession_2_epi3mm_MB2_TE30_TR2000_IRMf_20210727114430_6.nii'
anat = '/Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/T1/wDicom_baseline_mprage_sag_T1_160sl_iPAT2_20210727114430_2.nii'



# Concaténation des images fonctionnels

fmri_ses12 = concat_imgs([fmri_ses1,fmri_ses2])
func_img = index_img(fmri_ses12, condition_mask)


# IMPORT DU MASQUE VOIR BINARISATION

mask = '/Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/mask_neuroquery/mask_amygdala_bin.nii'

plotting.plot_roi(mask, bg_img=anat, cmap='Paired')

##############################ANOVA#################################

mask_img = mask
decoder = Decoder(estimator='svc', mask=mask_img, smoothing_fwhm=4,
                  standardize=True, screening_percentile=5, scoring='accuracy', cv=5)

############# FIT AVEC LE DECODER ET PREDICTION###########

decoder.fit(func_img, conditions)
y_pred = decoder.predict(func_img)
prediction = decoder.predict(func_img)


print(prediction)
print((prediction == conditions).sum() / float(len(conditions)))

print(decoder.cv_scores_)
print(decoder.cv_params_['Negatif'])
print(np.mean(decoder.cv_scores_['Negatif']))


########### CV AVEC LEAVE ONE GROUP OUT################################

cv = LeaveOneGroupOut()

decoder = Decoder(estimator='svc', mask=mask_img, standardize=True,
                  screening_percentile=5, scoring='accuracy', cv=cv,)

# Compute the prediction accuracy for the different folds (i.e. session)
decoder.fit(func_img, conditions, groups=session_label)

# Print the CV scores
print(decoder.cv_scores_['Negatif'])
print(decoder.cv_scores_)
print(np.mean(decoder.cv_scores_['Negatif']))
################ VISUALISATION DE LA CARTE DE POIDS#######################

weight_img = decoder.coef_img_['Negatif']
plot_stat_map(weight_img, bg_img=anat, title='SVM weights')


#VUE SUR INTERNET
view_img(weight_img, bg_img=anat,
         title="SVM weights", dim=-1)

#SAVE DE LA CARTE DE POIDS 

#weight_img.to_filename('WM_ANOVA_WB')

#################### ANALYSE R, MASQUES WHOLE BRAIN ####################################

fmri_ses1 = '//Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/baseline_session1/rSession_1_epi3mm_MB2_TE30_TR2000_IRMf_20210727114430_4.nii'
fmri_ses2 = '/Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/baseline_session2/rSession_2_epi3mm_MB2_TE30_TR2000_IRMf_20210727114430_6.nii'
anat = '/Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/T1/wDicom_baseline_mprage_sag_T1_160sl_iPAT2_20210727114430_2.nii'



# Concaténation des images fonctionnels

fmri_ses12 = concat_imgs([fmri_ses1,fmri_ses2])
func_img = index_img(fmri_ses12, condition_mask)


# IMPORT DU MASQUE VOIR BINARISATION

mask = '/Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/mask_neuroquery/mask.nii'

plotting.plot_roi(mask, bg_img=anat, cmap='Paired')

##############################ANOVA#################################

mask_img = mask
decoder = Decoder(estimator='svc', mask=mask_img, smoothing_fwhm=4,
                  standardize=True, screening_percentile=5, scoring='accuracy', cv=5)

############# FIT AVEC LE DECODER ET PREDICTION###########

decoder.fit(func_img, conditions)
y_pred = decoder.predict(func_img)
prediction = decoder.predict(func_img)


print(prediction)
print((prediction == conditions).sum() / float(len(conditions)))


print(decoder.cv_scores_)
print(decoder.cv_params_['Negatif'])
print(np.mean(decoder.cv_scores_['Negatif']))


########### CV AVEC LEAVE ONE GROUP OUT################################

cv = LeaveOneGroupOut()

decoder = Decoder(estimator='svc', mask=mask_img, standardize=True,
                  screening_percentile=5, scoring='accuracy', cv=cv,)

# Compute the prediction accuracy for the different folds (i.e. session)
decoder.fit(func_img, conditions, groups=session_label)

# Print the CV scores
print(decoder.cv_scores_['Negatif'])
print(decoder.cv_scores_)
print(np.mean(decoder.cv_scores_['Negatif']))
################ VISUALISATION DE LA CARTE DE POIDS#######################

weight_img = decoder.coef_img_['Negatif']
plot_stat_map(weight_img, bg_img=anat, title='SVM weights')


#VUE SUR INTERNET
view_img(weight_img, bg_img=anat,
         title="SVM weights", dim=-1)

#SAVE DE LA CARTE DE POIDS 

#weight_img.to_filename('WM_ANOVA_WB')

#################### ANALYSE R, MASQUES EMOTION REGULATION ####################################

fmri_ses1 = '//Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/baseline_session1/rSession_1_epi3mm_MB2_TE30_TR2000_IRMf_20210727114430_4.nii'
fmri_ses2 = '/Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/baseline_session2/rSession_2_epi3mm_MB2_TE30_TR2000_IRMf_20210727114430_6.nii'
anat = '/Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/T1/wDicom_baseline_mprage_sag_T1_160sl_iPAT2_20210727114430_2.nii'



# Concaténation des images fonctionnels

fmri_ses12 = concat_imgs([fmri_ses1,fmri_ses2])
func_img = index_img(fmri_ses12, condition_mask)


# IMPORT DU MASQUE VOIR BINARISATION

#mask = '/Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/mask_neuroquery/mask.nii'
mask = '/Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/mask_neuroquery/wmask_emotion_reg_bin.nii'

plotting.plot_roi(mask, bg_img=anat, cmap='Paired')

##############################ANOVA#################################

mask_img = mask
decoder = Decoder(estimator='svc', mask=mask_img, smoothing_fwhm=4,
                  standardize=True, screening_percentile=5, scoring='accuracy', cv=5)

############# FIT AVEC LE DECODER ET PREDICTION###########

decoder.fit(func_img, conditions)
y_pred = decoder.predict(func_img)
prediction = decoder.predict(func_img)


print(prediction)
print((prediction == conditions).sum() / float(len(conditions)))


print(decoder.cv_scores_)
print(decoder.cv_params_['Negatif'])
print(np.mean(decoder.cv_scores_['Negatif']))


########### CV AVEC LEAVE ONE GROUP OUT################################

cv = LeaveOneGroupOut()

decoder = Decoder(estimator='svc', mask=mask_img, standardize=True,
                  screening_percentile=5, scoring='accuracy', cv=cv,)

# Compute the prediction accuracy for the different folds (i.e. session)
decoder.fit(func_img, conditions, groups=session_label)

# Print the CV scores
print(decoder.cv_scores_['Negatif'])
print(decoder.cv_scores_)
print(np.mean(decoder.cv_scores_['Negatif']))
################ VISUALISATION DE LA CARTE DE POIDS#######################

weight_img = decoder.coef_img_['Negatif']
plot_stat_map(weight_img, bg_img=anat, title='SVM weights')


#VUE SUR INTERNET
view_img(weight_img, bg_img=anat,
         title="SVM weights", dim=-1)

#SAVE DE LA CARTE DE POIDS 

#weight_img.to_filename('WM_ANOVA_WB')

#################### ANALYSE R, MASQUES EMOTION ####################################

fmri_ses1 = '//Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/baseline_session1/rSession_1_epi3mm_MB2_TE30_TR2000_IRMf_20210727114430_4.nii'
fmri_ses2 = '/Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/baseline_session2/rSession_2_epi3mm_MB2_TE30_TR2000_IRMf_20210727114430_6.nii'
anat = '/Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/T1/wDicom_baseline_mprage_sag_T1_160sl_iPAT2_20210727114430_2.nii'



# Concaténation des images fonctionnels

fmri_ses12 = concat_imgs([fmri_ses1,fmri_ses2])
func_img = index_img(fmri_ses12, condition_mask)


# IMPORT DU MASQUE VOIR BINARISATION

mask = '/Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/mask_neuroquery/wmask_emotion_bin2.nii'

plotting.plot_roi(mask, bg_img=anat, cmap='Paired')

##############################ANOVA#################################

mask_img = mask
decoder = Decoder(estimator='svc', mask=mask_img, smoothing_fwhm=4,
                  standardize=True, screening_percentile=5, scoring='accuracy', cv=5)

############# FIT AVEC LE DECODER ET PREDICTION###########

decoder.fit(func_img, conditions)
y_pred = decoder.predict(func_img)
prediction = decoder.predict(func_img)


print(prediction)
print((prediction == conditions).sum() / float(len(conditions)))


print(decoder.cv_scores_)
print(decoder.cv_params_['Negatif'])
print(np.mean(decoder.cv_scores_['Negatif']))


########### CV AVEC LEAVE ONE GROUP OUT################################

cv = LeaveOneGroupOut()

decoder = Decoder(estimator='svc', mask=mask_img, standardize=True,
                  screening_percentile=5, scoring='accuracy', cv=cv,)

# Compute the prediction accuracy for the different folds (i.e. session)
decoder.fit(func_img, conditions, groups=session_label)

# Print the CV scores
print(decoder.cv_scores_['Negatif'])
print(decoder.cv_scores_)
print(np.mean(decoder.cv_scores_['Negatif']))
################ VISUALISATION DE LA CARTE DE POIDS#######################

weight_img = decoder.coef_img_['Negatif']
plot_stat_map(weight_img, bg_img=anat, title='SVM weights')


#VUE SUR INTERNET
view_img(weight_img, bg_img=anat,
         title="SVM weights", dim=-1)

#SAVE DE LA CARTE DE POIDS 

#weight_img.to_filename('WM_ANOVA_WB')

#################### ANALYSE R, MASQUES Amydgale ####################################

fmri_ses1 = '//Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/baseline_session1/rSession_1_epi3mm_MB2_TE30_TR2000_IRMf_20210727114430_4.nii'
fmri_ses2 = '/Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/baseline_session2/rSession_2_epi3mm_MB2_TE30_TR2000_IRMf_20210727114430_6.nii'
anat = '/Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/T1/wDicom_baseline_mprage_sag_T1_160sl_iPAT2_20210727114430_2.nii'



# Concaténation des images fonctionnels

fmri_ses12 = concat_imgs([fmri_ses1,fmri_ses2])
func_img = index_img(fmri_ses12, condition_mask)


# IMPORT DU MASQUE VOIR BINARISATION

mask = '/Users/quentingallet/Desktop/Dossier bureau/ML_FIRST_PILOT/mask_neuroquery/wmask_amygdala_bin.nii'

plotting.plot_roi(mask, bg_img=anat, cmap='Paired')

##############################ANOVA#################################

mask_img = mask
decoder = Decoder(estimator='svc', mask=mask_img, smoothing_fwhm=4,
                  standardize=True, screening_percentile=5, scoring='accuracy', cv=5)

############# FIT AVEC LE DECODER ET PREDICTION###########

decoder.fit(func_img, conditions)
y_pred = decoder.predict(func_img)
prediction = decoder.predict(func_img)


print(prediction)
print((prediction == conditions).sum() / float(len(conditions)))


print(decoder.cv_scores_)
print(decoder.cv_params_['Negatif'])
print(np.mean(decoder.cv_scores_['Negatif']))


########### CV AVEC LEAVE ONE GROUP OUT################################

cv = LeaveOneGroupOut()

decoder = Decoder(estimator='svc', mask=mask_img, standardize=True,
                  screening_percentile=5, scoring='accuracy', cv=cv,)

# Compute the prediction accuracy for the different folds (i.e. session)
decoder.fit(func_img, conditions, groups=session_label)

# Print the CV scores
print(decoder.cv_scores_['Negatif'])
print(decoder.cv_scores_)
print(np.mean(decoder.cv_scores_['Negatif']))
################ VISUALISATION DE LA CARTE DE POIDS#######################

weight_img = decoder.coef_img_['Negatif']
plot_stat_map(weight_img, bg_img=anat, title='SVM weights')


#VUE SUR INTERNET
view_img(weight_img, bg_img=anat,
         title="SVM weights", dim=-1)

#SAVE DE LA CARTE DE POIDS 

#weight_img.to_filename('WM_ANOVA_WB')

