# Dashboard for Machine Learning-Based DSM2 Salinity Emulator
## Usage
`conda env create -n ann -f environment.yml`

`activate ann`

`python dashboard.py`

## File Descriptions
#### Scripts
dashboard.py - main dashboard application  
evaluateann.py - run the ANN (imported by dashboard.py)  
hyperparams.py - hyperparameters for ANN training  
trainann.py - script to train ANN  
annutils.py - supporting utilities for ANN  
train_mtl_nn_dsm2_random_split.py - archive of original code from UCD (not used)  

#### Data
ann_inp.csv - one year of inputs for ANN evaluation (2014)  
dsm2_ann_inputs_20220204.xlsx - full set of inputs for ANN training  
input_scale.csv - matrix of initial scaling factors  
.\models - directory for the trained ANNs  

#### YAML
environment.yml - conda environment specification
