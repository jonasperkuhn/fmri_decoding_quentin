# fmri_decoding_quentin
## Instructions for using script
to do
- to binarize a mask, move it to the data folder and set the input param 'bin_mask' to True
- 

## Data folder structure
### Save pilot data in:
``` bash 
project_folder/data/pilot02/
```
### Within pilot data folder:
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
