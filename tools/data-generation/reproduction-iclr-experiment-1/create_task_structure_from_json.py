import os
import argparse
import json
import glob
import sys
import shutil
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument(
    "-t", "--target-folder", required=True, help="Where to save the folder structure."
)
parser.add_argument("-i", "--input", required=True, help="Input json structure.")
parser.add_argument(
    "-nr",
    "--n-reference-images",
    required=True,
    help="Number of reference images to show.",
)

args = parser.parse_args()

if (
    os.path.exists(args.target_folder)
    and len(list(glob.glob(os.path.join(args.target_folder, "*")))) > 0
):
    print("Error: Target folder exists and is not empty.")
    sys.exit(-1)

with open(args.input, "r") as f:
    data = json.load(f)

experiment_name = os.path.basename(args.target_folder)

tasks = data["tasks"]

for task in tqdm(tasks, position=0):
    task_folder = os.path.join(args.target_folder, f"task_{task['index']}")
    os.makedirs(task_folder, exist_ok=True)

    # copy stimuli
    trials_folder = os.path.join(task_folder, "trials")
    os.makedirs(trials_folder, exist_ok=True)

    for i, trial in enumerate(tqdm(task["trials"], position=1, leave=False)):
        source_references_path = trial["references"]
        source_queries_path = trial["queries"]

        # create folders
        trial_folder = os.path.join(trials_folder, f"trial_{i + 1}")
        target_references_path = os.path.join(trial_folder, "references")
        target_queries_path = os.path.join(trial_folder, "queries")
        os.makedirs(trial_folder, exist_ok=True)
        os.makedirs(target_references_path, exist_ok=True)
        os.makedirs(target_queries_path, exist_ok=True)

        if trial["mode"] == "catch_trial":
            # copy queries
            shutil.copy(
                os.path.join(source_queries_path, "min_1.png"),
                os.path.join(target_queries_path, "min.png"),
            )
            shutil.copy(
                os.path.join(source_queries_path, "max_8.png"),
                os.path.join(target_queries_path, "max.png"),
            )
        else:
            # copy queries
            shutil.copy(
                os.path.join(source_queries_path, "min_0.png"),
                os.path.join(target_queries_path, "min.png"),
            )
            shutil.copy(
                os.path.join(source_queries_path, "max_9.png"),
                os.path.join(target_queries_path, "max.png"),
            )

        # copy references
        for source_reference in glob.glob(
            os.path.join(source_references_path, "*.png")
        ):
            target_reference = os.path.join(
                target_references_path, os.path.basename(source_reference)
            )
            shutil.copy(source_reference, target_reference)

    # create index
    index = dict(
        task_name=f"{experiment_name}/task_{task['index']}",
        n_reference_images=args.n_reference_images,
        n_trials=len(task["trials"]),
        catch_trial_idxs=[
            idx + 1
            for idx in range(len(task["trials"]))
            if task["trials"][idx]["mode"] == "catch_trial"
        ],
    )
    with open(os.path.join(task_folder, "index.json"), "w") as f:
        json.dump(index, f)
