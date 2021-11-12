"""Adds the stddev of the patches to the experiment structure"""

import os
import numpy as np
import argparse
import occlusion_utils as ut
import json
from PIL import Image


def calculate_patch_variance(image, position):
    x_start, x_end, y_start, y_end = position

    return np.std(image[x_start:x_end, y_start:y_end, :])


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

    # load image
    query_image_path = os.path.join(trial["queries"], "query_default.png")
    image = np.array(Image.open(query_image_path)).astype(np.float) / 255.0

    min_patch_std = calculate_patch_variance(image, min_position)
    max_patch_std = calculate_patch_variance(image, max_position)

    trial["min_query_patch_std"] = "%.12f" % min_patch_std
    trial["max_query_patch_std"] = "%.12f" % max_patch_std


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
