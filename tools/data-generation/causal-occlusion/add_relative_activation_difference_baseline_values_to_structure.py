"""Adds the activation of min, max and base query image to a copy of the experiment structure"""

import numpy as np
import json
import os
import argparse
import occlusion_utils as ut


def process_trial(trial, occlusion_type_index):
    patch_relative_size = ut.percentage_side_length_list[occlusion_type_index]
    fn = os.path.join(
        trial["queries"],
        f"activations_for_occlusions_of_{patch_relative_size}_percent.npy",
    )

    if args.path_search != "":
        fn = fn.replace(args.path_search, args.path_replace)

    activations = np.load(fn)

    base_query_activation = activations[-1]
    min_query_activation = activations.min()
    max_query_activation = activations.max()

    trial["base_query_activation"] = "%.12f" % base_query_activation
    trial["min_query_activation"] = "%.12f" % min_query_activation
    trial["max_query_activation"] = "%.12f" % max_query_activation


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
    choices=(30, 40, 50),
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

occlusion_size_index = [30, 40, 50].index(args.occlusion_size)

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
