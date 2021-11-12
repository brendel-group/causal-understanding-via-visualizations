"""
 This script creates a json-file with the structure of the tasks for the MTurk replication experiment based on the first
 ICLR lab experiment. The json file contains a dictionary (called data) with one key ('tasks') and value (a list).
 This list contains n_tasks dictionaries with four entries:
 'index'                 indicating the index of the task
 'raw_trials'            contains a list with the data of --n-trials trials
 'raw_catch_trials'      contains a list with the data of --n-catch-trials catch trials
 'trials'                contains a list with the shuffled and merged data of raw_trials and raw_catch_trials
 Each list item contains a dictionary with the entries 'mode' ('catch_trial' or 'normal'), 'queries' (absolute paths)
 and 'references' (absolute paths), e.g.:
 {'mode': 'catch_trial',
 'queries': '...stimuli/channel/
    sampled_trials/layer_p/kernel_size_p/channel_p/natural_images/batch_0',
 'references': '...stimuli/channel/
    sampled_trials/layer_p/kernel_size_p/channel_p/optimized_images'}
"""

import argparse
import glob
import os
import random
import numpy as np
import json

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--source-folder",
    required=True,
    help="Path to source stimuli used in original experiment.",
)
parser.add_argument(
    "-o", "--output", required=True, help="Where to save final json structure."
)
parser.add_argument(
    "-k",
    "--kernel-sizes",
    nargs="+",
    help="Channels to use",
    type=str,
    required=True,
    choices=("0", "1", "2", "3"),
)
parser.add_argument(
    "-nc",
    "--n-catch-trials",
    type=int,
    required=True,
    help="Number of catch trials per task.",
)
parser.add_argument(
    "-nt", "--n-trials", type=int, required=True, help="Number of trials per task."
)
parser.add_argument(
    "-nh", "--n-tasks", type=int, required=True, help="Number of tasks."
)
parser.add_argument("-m", "--mode", choices=("natural", "optimized"), required=True)
parser.add_argument("--seed", type=int, default=1)

args = parser.parse_args()
print(args)

n_lab_participants = 10  # number of participants whose data is being replicated
assert args.n_tasks % n_lab_participants == 0, (
    f"n-tasks should be a multiple of {n_lab_participants} because that's how"
    f" many participants we are replicating out of the 10 participants in ICLR Exp I"
)

random.seed(args.seed)
np.random.seed(args.seed)

layers_available_for_trials = glob.glob(os.path.join(args.source_folder, "layer_[0-9]"))
layers_available_for_catch_trials = glob.glob(
    os.path.join(args.source_folder, "layer_[a-z]")
)

# We are only considering randomly selected feature maps, hence channel_0 and never channel_1
layers_available_for_trials = [
    os.path.join(f, f"kernel_size_{c}", "channel_0", "natural_images",)
    for f in layers_available_for_trials
    for c in args.kernel_sizes
]
layers_available_for_catch_trials = [
    os.path.join(f, f"kernel_size_{f[-1]}", f"channel_{f[-1]}", "natural_images")
    for f in layers_available_for_catch_trials
]


def create_trials(layers_available, lab_participant_idx, n_trials, trial_mode):
    """ create dictionary with information for each trial of one participant and the given trial_mode

    Args:
        layers_available:    list of layers available for trials
        lab_participant_idx: int indicating id of lab-participant whose data is copied
        n_trials:            int indicating number of trials that one worker will do
        trial_mode:          str indicating catch trial or main trial

    Returns:
        lkb_for_task:      dictionary with mode, queries and references information for each trial (see top of file)

    """
    # chose layer + kernel_sizes to use for that task
    if n_trials == len(layers_available):
        lk_for_task = layers_available
    else:
        replace_boolean = False if n_trials <= len(layers_available) else True
        lk_for_task = np.random.choice(
            layers_available, size=n_trials, replace=replace_boolean
        )

    # all batches are the same id
    if args.mode == "natural":
        batch_id = lab_participant_idx + 1
    elif args.mode == "optimized":
        batch_id = lab_participant_idx + 2

    if trial_mode == "catch_trial":
        batch_id = 0

    if args.mode == "natural":
        # use same batch for query and reference images
        lkb_for_task = [
            dict(
                queries=os.path.join(lk, f"batch_{batch_id}"),
                references=os.path.join(lk, f"batch_{batch_id}"),
                mode=trial_mode,
            )
            for lk in lk_for_task
        ]
    else:
        # there is no notion of batches for optimized references
        # catch trials show optimized query images, too
        if trial_mode == "catch_trial":
            lkb_for_task = [
                dict(
                    queries=os.path.join(os.path.dirname(lk), "optimized_images"),
                    references=os.path.join(os.path.dirname(lk), "optimized_images"),
                    mode=trial_mode,
                )
                for lk in lk_for_task
            ]
        else:
            lkb_for_task = [
                dict(
                    queries=os.path.join(lk, f"batch_{batch_id}"),
                    references=os.path.join(os.path.dirname(lk), "optimized_images"),
                    mode=trial_mode,
                )
                for lk in lk_for_task
            ]

    return lkb_for_task


data = dict(tasks=[])

for task_idx in range(args.n_tasks):
    lab_participant_idx = task_idx // (args.n_tasks // n_lab_participants)
    worker_idx = task_idx % (args.n_tasks // n_lab_participants)

    # create task data
    task = dict(index=task_idx + 1)

    task["raw_trials"] = create_trials(
        layers_available_for_trials, lab_participant_idx, args.n_trials, "normal"
    )
    task["raw_catch_trials"] = create_trials(
        layers_available_for_catch_trials,
        lab_participant_idx,
        args.n_catch_trials,
        "catch_trial",
    )

    task["trials"] = task["raw_trials"] + task["raw_catch_trials"]
    random.shuffle(task["trials"])

    data["tasks"].append(task)


with open(args.output, "w") as f:
    json.dump(data, f)
