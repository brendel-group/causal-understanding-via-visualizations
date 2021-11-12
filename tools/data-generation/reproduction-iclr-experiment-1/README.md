1. Execute `./create_data.sh PathToCausalVisualizationsFolder` to create data on the CIN cluster and store them in the `PathToCausalVisualizationsFolder` path.
2. On the web server execute:
    ```bash
    rm -r mturk-data/experiments/reproduction_iclr_experiment_1/natural_* mturk-data/experiments/reproduction_iclr_experiment_1/optimized_*
    
    rclone copy --transfers=100 -P --stats-one-line cin:PathToCausalVisualizationsFolder/tools/data-generation/reproduction-iclr-experiment-1/data mturk-data/experiments/reproduction_iclr_experiment_1/
    ```
    where `PathToCausalVisualizationsFolder` must be replaced with the path to the root folder of the project.