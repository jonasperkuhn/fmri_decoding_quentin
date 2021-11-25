# fmri_decoding_quentin
## Instructions for using script
- Main script: `generate_weight_map.py` (for single session data with onsets)
- Other scripts (harder to handle, but may contain useful code):
  - `compare_preproc_and_cvtype.py` (to compare cv params across multiple sessions and emot. reg. strategies)
  - `decoding_pilot_1_2.py` (for decoding data from the first two pilots)
- Before using the `generate_weight_map.py` script, move preprocessed brain data (fmri & T1), onsets, and binarized mask in subject space to data folder (cf. "Data folder structure" below)
- Within the script, set the following variables:
  - data_path: indicate the path to your data folder
  - preprocessing: preprocessing method for which you want to decode
  - cv_type: either 'k_fold' or 'block_out'
  - n_folds: (int) only relevant for 'k_fold', else param is ignored
  - anova: (boolean) whether you want to perform an anova before decoding
  - strategy: emotion regulation strategy session (the one that corresponds to the brain data in ./EPIs_baseline/)
  - random_state: random seed used in cross-validation

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
### Alternative data folder structure for pilot 1/2 (multiple sessions):
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


## Setting up the environment
- With conda (Anaconda Environment): use env_decoding_nfb.yml
  - in anaconda prompt, execute: `conda env create --file=/PATH/TO/env_decoding_nfb.yml --prefix=/PATH/TO/MYENV`
- With pip (Virtual Environment):
  - python=3.7
  - pandas=1.3.*
  - numpy=1.21.*
  - nilearn=0.8.*
  - scikit-learn=0.23.*
  - matplotlib=3.4.*
