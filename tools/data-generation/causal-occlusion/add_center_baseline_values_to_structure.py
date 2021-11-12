"""Adds the distances (from the patch to the center of the image for both min and may query image) to
a copy of the experiment structure"""

import numpy as np
import json
import os
import argparse
import occlusion_utils as ut


def calculate_patch_center_distance(x_start, y_start, occlusion_size_i):
    """Calculate the distance from the occlusion patch's center to
    the center of the images"""

    image_center = 111.5

    x_mean = x_start + occlusion_size_i / 2
    y_mean = y_start + occlusion_size_i / 2

    # pythagoras theorem
    distance = np.sqrt((image_center - y_mean) ** 2 + (image_center - x_mean) ** 2)

    return distance


def process_trial(trial, occlusion_type_index):
    patch_relative_size = ut.percentage_side_length_list[occlusion_type_index]
    patch_absolute_size = ut.occlusion_sizes_list[occlusion_type_index]
    heatmap_size = ut.heatmap_sizes_list[occlusion_type_index]

    list_of_positions = ut.get_list_of_occlusion_positions(
        heatmap_size, patch_absolute_size
    )

    fn = os.path.join(
        trial["queries"],
        f"activations_for_occlusions_of_{patch_relative_size}_percent.npy",
    )

    if args.path_search != "":
        fn = fn.replace(args.path_search, args.path_replace)

    activations = np.load(fn)

    min_activation_idx = activations[:-1].argmin()
    max_activation_idx = activations[:-1].argmax()

    min_position = list_of_positions[min_activation_idx]
    max_position = list_of_positions[max_activation_idx]

    min_distance_to_center = calculate_patch_center_distance(
        min_position[0], min_position[1], patch_absolute_size
    )
    max_distance_to_center = calculate_patch_center_distance(
        max_position[0], max_position[1], patch_absolute_size
    )

    trial["min_query_center_distance"] = "%.12f" % min_distance_to_center
    trial["max_query_center_distance"] = "%.12f" % max_distance_to_center


parser = argparse.ArgumentParser()
parser.add_argument(
    "--input-structure",
    required=True,
    help="path to the input experiment structure json",
)
parser.add_argument(
    "--output-structure",
    required=True,
    help="where to save the final experiment structure " "with the appended info",
)
parser.add_argument(
    "--occlusion-size",
    required=True,
    type=int,
    choices=ut.percentage_side_length_list,
    help="occlusion size used for the stimuli",
)
parser.add_argument(
    "--path-search",
    required=False,
    default="",
    help="in case the path to the raw image data has changed, this argument can be used to "
    "replace parts of the path with something else.",
)
parser.add_argument(
    "--path-replace",
    required=False,
    default="",
    help="value to replace parts of the image paths with",
)

args = parser.parse_args()

assert (args.path_search == "" and args.path_replace == "") or (
    args.path_search != "" and args.path_replace != ""
)

occlusion_size_index = ut.percentage_side_length_list.index(args.occlusion_size)

with open(args.input_structure, "r") as f:
    structure = json.load(f)

for task in structure["tasks"]:
    for trial in task["raw_trials"]:
        process_trial(trial, occlusion_size_index)
    for trial in task["raw_catch_trials"]:
        process_trial(trial, occlusion_size_index)
    for trial in task["trials"]:
        process_trial(trial, occlusion_size_index)

with open(args.output_structure, "w") as f:
    json.dump(structure, f)
