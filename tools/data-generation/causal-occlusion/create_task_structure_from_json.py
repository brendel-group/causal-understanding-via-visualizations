import os
import argparse
import json
import glob
import sys
import shutil
from tqdm import tqdm
import random


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
        # create folders
        trial_folder = os.path.join(trials_folder, f"trial_{i + 1}")
        target_references_path = os.path.join(trial_folder, "references")
        target_queries_path = os.path.join(trial_folder, "queries")
        os.makedirs(trial_folder, exist_ok=True)
        os.makedirs(target_references_path, exist_ok=True)
        os.makedirs(target_queries_path, exist_ok=True)

        # copy queries
        source_queries_path = trial["queries"]
        shutil.copy(
            os.path.join(source_queries_path, "query_max_activation.png"),
            os.path.join(target_queries_path, "max.png"),
        )
        shutil.copy(
            os.path.join(source_queries_path, "query_default.png"),
            os.path.join(target_queries_path, "base.png"),
        )

        # In the none condition, we use a randomly chosen minimal query image, whereas for all other
        # conditions we use the regular minimal query image.
        if "no_references" in args.target_folder and trial["mode"] == "catch_trial":
            possible_source_min_query_path = [
                "$DATAPATH/stimuli/stimuli_pure_conditions/"
                f"channel/instruction_practice_catch/layer_{it}/kernel_size_{it}/channel_{it}/natural_images/batch_0/"
                "40_percent_side_length/query_min_activation.png"
                for it in "vwxyp"  # feature maps not selected to be used in the experiment as catch/practice etc.
            ]
            source_min_query_path = possible_source_min_query_path[
                random.randint(0, len(possible_source_min_query_path) - 1)
            ]
            shutil.copy(
                source_min_query_path, os.path.join(target_queries_path, "min.png")
            )
        else:
            shutil.copy(
                os.path.join(source_queries_path, "query_min_activation.png"),
                os.path.join(target_queries_path, "min.png"),
            )

        # copy references
        random.seed(task["index"] + i)  # This is not ideally set
        # in case of mixed condition
        if "optimized" in args.target_folder and "natural" in args.target_folder:
            source_references_optimized_path = trial["references_optimized"]
            source_references_natural_path = trial["references_natural"]

            optimized_imgs_paths_list = glob.glob(
                os.path.join(source_references_optimized_path, "*.png")
            )
            natural_imgs_paths_list = glob.glob(
                os.path.join(source_references_natural_path, "*.png")
            )

            # first copy over the natural base image as the very last image in the mixed folder
            last_reference = natural_imgs_paths_list[
                -1
            ]  # depending on file system, not necessarily the max_5 img!
            base_img_dirname = os.path.dirname(last_reference)
            last_img_basename = os.path.basename(last_reference)
            base_img_basename = f"{last_img_basename.split('.png')[0][:-1]}{len(natural_imgs_paths_list)-1}.png"
            source_reference = os.path.join(base_img_dirname, base_img_basename)
            target_reference = os.path.join(
                target_references_path,
                f"reference_max_{len(natural_imgs_paths_list) + len(optimized_imgs_paths_list) -1}.png",
            )
            shutil.copy(source_reference, target_reference)
            # remove path to base img from list
            natural_imgs_paths_list.remove(source_reference)

            # shuffle the image numbering
            random.shuffle(optimized_imgs_paths_list)
            random.shuffle(natural_imgs_paths_list)
            # first copy optimized (indices 0-3) and then optimized images (indices 4-8)
            for reference_idx, source_reference in enumerate(optimized_imgs_paths_list):
                # continue counting of reference images
                target_reference = os.path.join(
                    target_references_path, f"reference_max_{reference_idx}.png"
                )
                shutil.copy(source_reference, target_reference)
            for reference_idx, source_reference in enumerate(natural_imgs_paths_list):
                target_reference = os.path.join(
                    target_references_path,
                    f"reference_max_{reference_idx + len(optimized_imgs_paths_list)}.png",
                )
                shutil.copy(source_reference, target_reference)
        # in case of natural or optimized condition
        else:
            source_references_path = trial["references"]

            imgs_paths_list = glob.glob(os.path.join(source_references_path, "*.png"))

            # first copy over the base image as the very last image in the folder
            last_reference = imgs_paths_list[
                -1
            ]  # depending on file system, not necessarily the max_5 img!
            base_img_dirname = os.path.dirname(last_reference)
            last_img_basename = os.path.basename(last_reference)
            base_img_basename = (
                f"{last_img_basename.split('.png')[0][:-1]}{len(imgs_paths_list)-1}.png"
            )
            source_reference = os.path.join(base_img_dirname, base_img_basename)
            target_reference = os.path.join(
                target_references_path, f"reference_max_{len(imgs_paths_list) -1}.png"
            )
            shutil.copy(source_reference, target_reference)
            # remove path to base img from list
            imgs_paths_list.remove(source_reference)

            # shuffle the image numbering
            random.shuffle(imgs_paths_list)

            for reference_idx, source_reference in enumerate(imgs_paths_list):
                target_reference = os.path.join(
                    target_references_path, f"reference_max_{reference_idx}.png"
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
