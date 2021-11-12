# Generate Stimuli for Causal Feature Visualization Experiment

## Main and Catch Trials


### At first, generate the raw data locally

1. Execute `create_local_raw_data.sh` to save the activations of an ImageNet subset for all feature maps.

### Next, generate the stimuli locally

1. At first, create a directory under `$DATAPATH/`, which we will refer to as 
   `PathToCausalVisualizationsFolder`. Its name should either include the keyword `pure` or `mixed`, which corresponds 
   to a pure or mixed reference images respectively, so that the generation pipeline can determine the correct number of
   stimuli to generate (9 or 5 and 4). 
   
2. In the folder `PathToCausalVisualizationsFolder`, add the csv file `layer_folder_mapping_{trial_type}.csv`. Here, 
   `trial_type` is `instruction_practice_catch` or `sampled_trials`. The csv file's structure is as follows:
    
    ``encoding in folder structure - correspondence in InceptionV1
layer_number,kernel_size_number,channel_number,layer_name,pre_post_relu,kernel_size,feature_map_number
0,1,0,mixed3a,pre_relu,3x3,189``
   
   Note that the file `layer_folder_mapping_instruction_practice_catch.csv` has letters as the `layer_number`, 
   `kernel_size_number`, and `channel_number` in a non-consecutive order because the first two authors selected the easiest 
   units from a larger selection that they labeled a-z.   

3. Next, make sure `create_local_data.sh` contains the correct `stimuli_folder` argument (`-s`) from step 1.
   
4. Then execute `./create_local_data.sh PathToCausalVisualizationsFolder trial_type`, where `trial_type` corresponds to 
   the `trial_type` described above, to create and store data on the cluster in the `PathToCausalVisualizationsFolder` 
   path. Optionally, you can add a third argument `install` to install all the required packages. 
   The `create_local_data.sh` script calls `1_save_natural_reference_and_default_images_new.ipynb`,  
   `2_occlusion_activations_in_Inception_V1.ipynb`, `3_occlusion_save_query_images.ipynb`,
   `4_save_synthetic_reference_images.ipynb`, and also `8_blur_activations_and_save_maximally_activating_blur_img.ipynb`
   in case of the `pure` condition.

5. In total, you have to run the pipeline for the `mixed` and `pure` conditions.


### Then, copy the stimuli over to the server

To copy over the main and the catch trials:
1. Make sure `create_mturk_data.sh` contains the `parent_folder`s of the stimuli. Then execute 
   `./create_mturk_data.sh PathToCausalVisualizationsFolder` to create data on the cluster and store them in the 
   `PathToCausalVisualizationsFolder` path. The `create_mturk_data.sh` script calls `create_task_structure_json.py` and
   `create_task_structure_from_json.py` and creates the baselines.
    
2. Then use `rclone copy` to copy the generated stimuli to the web server.

## Instruction Stimuli

1. [5_instruction_handpicked.ipynb](5_instruction_handpicked.ipynb)
2. [6_instruction_trials_arranged_for_server.ipynb](6_instruction_trials_arranged_for_server.ipynb)
3. Then use `rclone copy` to copy the instruction stimuli to the web server.

## Practice Stimuli

1. To generate the practice stimuli, execute [7_practice_trials_arranged_for_server](7_practice_trials_arranged_for_server.ipynb).
2. Finally, copy these trials over to the server.


## Generate Query Image for Figure 1B

To generate the query images in Figure 1B, execute [9_fig1_handpicked](9_fig1_handpicked.ipynb).

### Folder structure

```
parent_folder (e.g. $DATAPATH/stimuli/stimuli_pure_conditions)
│   layer_folder_mapping_sampled_trials.csv
│   layer_folder_mapping_instruction_practice_catch.csv
│
└───channel 
    │
    └───sampled_trials
    │   │
    │   └───layer_0
    │   │   │
    │   │   └───kernel_size_[1,3]
    │   │       │
    │   │       └───channel_0
    │   │           │
    │   │           └───natural_images
    │   │           │   │
    │   │           │   └───batch_0
    │   │           │   │   │   activation_max.csv
    │   │           │   │   │   reference_max_[0-9].png
    │   │           │   │   │
    │   │           │   │   └───val
    │   │           │   │   │   │
    │   │           │   │   │   └───n04141327
    │   │           │   │   │           ILSVRC2012_val_00032048.JPEG # this images cropped is max_9.png and query_default.png
    │   │           │   │   │
    │   │           │   │   └───30_percent_side_length
    │   │           │   │   │       activations_for_occlusions_of_30_percent.npy
    │   │           │   │   │       query_default.png
    │   │           │   │   │       query_max_activation.png
    │   │           │   │   │       query_min_activation.png
    │   │           │   │   │
    │   │           │   │   └───[4,5]0_percent_side_length
    │   │           │   │           (as above)
    │   │           │   │
    │   │           │   └───batch_[1-19]
    │   │           │           (as above)
    │   │           │
    │   │           └───natural_blur_images
    │   │           │   │
    │   │           │   └───batch_0
    │   │           │   │   │
    │   │           │   │   └───40_percent_side_length
    │   │           │   │           activations_for_occlusions_of_40_percent_reference_max_[0,9].npy
    │   │           │   │           blurred_reference_max_[0,9].png
    │   │           │   │
    │   │           │   └───batch_[1-19]
    │   │           │           (as above)
    │   │           │
    │   │           └───optimized_images
    │   │                   max_additional_global_diversity_loss.npy
    │   │                   max_objective_values.npy
    │   │                   reference_max_[0-9].png
    │   │
    │   └───layer_[1-8]
    │           (as above)
    │
    └───instruction_practice_catch
        │
        └───layer_a
        │   │
        │   └───kernel_size_a
        │       │
        │       └───channel_a
        │           │
        │           └───natural_images
        │           │   │
        │           │   └───batch_0
        │           │   │   │   activation_max.csv
        │           │   │   │   reference_max_[0-9].png
        │           │   │   │
        │           │   │   └───val
        │           │   │   │   │
        │           │   │   │   └───n03220513
        │           │   │   │           ILSVRC2012_val_00030349.JPEG # this images cropped is max_9.png and query_default.png
        │           │   │   │
        │           │   │   └───30_percent_side_length
        │           │   │   │       activations_for_occlusions_of_30_percent.npy
        │           │   │   │       query_default.png
        │           │   │   │       query_max_activation.png
        │           │   │   │       query_min_activation.png
        │           │   │   │
        │           │   │   └───[4,5]0_percent_side_length
        │           │   │           (as above)
        │           │   │
        │           │   └───batch_[1-19]
        │           │           (as above)
        │           │
        │           └───natural_blur_images
        │           │   │
        │           │   └───batch_0
        │           │   │   │
        │           │   │   └───40_percent_side_length
        │           │   │           activations_for_occlusions_of_40_percent_reference_max_[0,9].npy
        │           │   │           blurred_reference_max_[0,9].png
        │           │   │
        │           │   └───batch_[1-19]
        │           │           (as above)
        │           │
        │           └───optimized_images
        │                   max_additional_global_diversity_loss.npy
        │                   max_objective_values.npy
        │                   reference_max_[0-9].png
        │
        └───layer_[other_letters]
                (as above)
                # catch trials: see create_task_structure_json.py
```