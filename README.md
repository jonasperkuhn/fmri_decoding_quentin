# fmri_decoding_quentin
## Instructions for using script
- two main scripts: generate_weight_map.py (for single session data with onsets) and decoding_pilot_1_2.py for decoding data from the first two pilots
- before using the script, move preprocessed brain data (fmri & T1), onsets, and binarized mask in subject space to data folder (cf. below)
- within the script, set the following variables:
  - data_path: indicate the path to your data folder
  - preprocessing: preprocessing method for which you want to decode
  - strategy: emotion regulation strategy session (the one that corresponds to the brain data in ./EPIs_baseline/)
- below, where the functions are called, you can further adjust the following function inputs:
  - cv_type: either 'k_fold' or 'block_out'
  - n_folds: (int) only relevant for 'k_fold', else param is ignored
  - anova: (boolean) whether you want to perform an anova before decoding
  - 

## Data folder structure
- Based on data folder structure generated automatically during preprocessing via Pamela's matlab script
- binarized roi mask in subject space:
``` .../Mask_ROI_emo/ ```
- brain data for specified strategy:
``` .../EPIs_baseline/ ```
  - as preprocessed 4D .nii files
- mean epi:
``` .../MCTemplate/ ```
- anatomical data in:
``` .../T1/ ```
- onsets file (event onsets, opennft output .txt file)
``` .../Onsets/ ```
- Empty folder to save decoded weights:
``` .../W1/ ```
### Data folder for pilot 1/2:
- raw or binarized roi masks:
``` .../pilot02/masks/ ```
- brain data for session 1 in:
``` .../pilot02/ses1/ ```
- brain data for session 2 in:
``` .../pilot02/ses2/ ```
- anatomical data in:
``` .../pilot02/T1/ ```
- behavioral file (specifying session, condition, and tr; each row is one tr)
``` .../pilot02/data_behav.csv ```
