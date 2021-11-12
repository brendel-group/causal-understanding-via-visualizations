python3 spawn_experiment.py \
  --experiment-name=reproduction_iclr_experiment_1_ks_1 \
  --task-namespace=natural_9_references \
  --n-tasks=130 \
  --n-repetitions=1 \
  --environment=real \
  --reward=1.25 \
  --row-variability-threshold=1 \
  --min-instruction-time=15 \
  --max-instruction-time=-1 \
  --min-total-response-time=90 \
  --max-total-response-time=600 \
  --catch-trial-ratio-threshold=0.6 \
  --hit-lifetime=1.0 \
  --single-participation-identifier=20210421-natural \
  --previous-single-participation-qualifications \
    reproduction_iclr_experiment_1_ks_1-20210415-natural reproduction_iclr_experiment_1_ks_1-20210415-optimized \
    reproduction_iclr_experiment_1_ks_1-20210416-natural reproduction_iclr_experiment_1_ks_1-20210416-optimized \
    reproduction_iclr_experiment_1_ks_1-20210420-natural reproduction_iclr_experiment_1_ks_1-20210420-optimized \
    --output-folder=output/full_experiment_20210421/natural

python3 spawn_experiment.py \
  --experiment-name=reproduction_iclr_experiment_1_ks_1 \
  --task-namespace=optimized_9_references \
  --n-tasks=130 \
  --n-repetitions=1 \
  --environment=real \
  --reward=1.25 \
  --row-variability-threshold=1 \
  --min-instruction-time=15 \
  --max-instruction-time=-1 \
  --min-total-response-time=90 \
  --max-total-response-time=600 \
  --catch-trial-ratio-threshold=0.6 \
  --hit-lifetime=1.0 \
  --single-participation-identifier=20210421-optimized \
  --previous-single-participation-qualifications \
    reproduction_iclr_experiment_1_ks_1-20210415-natural reproduction_iclr_experiment_1_ks_1-20210415-optimized \
    reproduction_iclr_experiment_1_ks_1-20210416-natural reproduction_iclr_experiment_1_ks_1-20210416-optimized \
    reproduction_iclr_experiment_1_ks_1-20210420-natural reproduction_iclr_experiment_1_ks_1-20210420-optimized \
    reproduction_iclr_experiment_1_ks_1-20210421-natural \
  --output-folder=output/full_experiment_20210421/optimized
