#!/usr/bin/env python3

""" Script to decode image type (negative vs. neutral) from fmri brain activity """

import numpy as np
import functions_decoding as my_fcts


# todo: check whether it runs for pilot 2; check whether all runs for pilots 3 and 4
# todo: for pilot 6, set "strategie" name of pause in onsets file to that of previous strategy (to include it into trials)

# define main params
data_path = "C:/Users/Jonas/PycharmProjects/fmri_decoding_quentin/decoding/data/pilot_03/"  # set path to data folder of current set
preprocessing = "r"  # specify as 'r' (realigned), 'sr' (realigned + smoothed), or 'swr' (sr + normalization); if swr, set perform_decoding_cv(anova=True)
cv_type = 'k_fold'  # cross-validation type: either 'k_fold' or 'block_out'
n_folds = 5  # number of folds to perform in k-fold cross-validation; only used if cv_type == 'k_fold'
anova = False  # if True, anova is performed as feature reduction method prior to decoding
strategy = "Affects positifs"  # specify strategy to decode, corresponding to the brain data in the folder (from "Affects positifs", "Pleine conscience", "Reevaluation cognitive", "Pas d'instructions")
random_state = 8

# load data
fmri_niimgs, fname_anat, mask, conds_fmri, condition_mask, conditions, cond_names = \
    my_fcts.load_data(strategy, preprocessing, data_path, plot=False)

# build and fit decoder in cv
decoder = my_fcts.perform_decoding_cv(conditions, fmri_niimgs, mask, conds_fmri,
                              condition_mask, random_state, cv_type=cv_type, n_folds=n_folds, anova=anova)

# plot decoder weights
#plot_weights(decoder, fname_anat, condition=cond_names[1])

# save decoder weights
#weigth_img = decoder.coef_img_[cond_names[1]]
#weigth_img.to_filename(data_path + 'W1/weights.nii')

# evaluate decoder
scores = decoder.cv_scores_[cond_names[1]]  # classification accuracy for each fold
mean_score = np.mean(scores)  # average classification accuracy across folds
# save evaluation results in txt file
#save_accs_to_txt(mean_score, scores, data_path)
print(mean_score)
#print(np.std(scores))
