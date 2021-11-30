#!/usr/bin/env python3

""" Script to compare different decoding settings """

import numpy as np
import functions_decoding as my_fcts

# define main params
strategy = "Regulation_3"  # specify strategy corresponding to the brain data in the folder (from "Affects positifs", "Pleine conscience", "Reevaluation cognitive", "Pas d'instructions"; for pilot 4, "Regulation_3")
data_path = "C:/Users/Jonas/PycharmProjects/fmri_decoding_quentin/decoding/data/pilot_04/"  # set path to data folder of current set
random_state = 8

# scores_3PC, std_3PC = compare_cvs(strategy, data_path, random_state)
# scores_3AP, std_3AP = compare_cvs(strategy, data_path, random_state)
# scores_4PI, std_4PI = compare_cvs(strategy, data_path, random_state)
scores_4RE, std_4RE = my_fcts.compare_cvs(strategy, data_path, random_state)
print('comparison finished')

mean_scores_sessions = np.mean(np.array([scores_3PC, scores_3AP, scores_4PI]), axis=0)
mean_std_sessions = np.mean(np.array([std_3PC, std_3AP, std_4PI]), axis=0)
avg_std = pd.DataFrame(mean_std_sessions, columns=preprocessing_types)
