# Data Analysis

## Download the Data

1. Download the data from the server to your local `data` folder.

2. Download the json file from the server.

# Run the Analysis
To run the analysis, your `data` folder should contain a folder `baselines` with the json-files containing the task information and a folder `couterfactual_experiment` with a folder for each reference image condition containing the pkl-files.  
To generate the figures of the appendix, execute [analysis_ICLR_style](analysis_ICLR_style.ipynb). These notebooks will use the following helper scripts:
[utils_figures](utils_figures.py), [utils_figures_helper](utils_figures_helper.py), 
[utils_MTurk_figures](utils_MTurk_figures.py), and [utils_data](utils_data.py).