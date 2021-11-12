"""Adds the saliency probability of the patches to the experiment structure"""

import os
import numpy as np
import argparse
import occlusion_utils as ut
import json
from PIL import Image
import torch
from deepgaze_pytorch.deepgaze2_dsrex3 import deepgaze2_dsrex3
from tqdm import tqdm


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

    # find min/max over all tested images except the default image which is the last entry in the list
    min_activation_idx = activations[:-1].argmin()
    max_activation_idx = activations[:-1].argmax()

    min_position = list_of_positions[min_activation_idx]
    max_position = list_of_positions[max_activation_idx]

    # load image
    default_query_image_path = os.path.join(trial["queries"], "query_default.png")
    default_query_image = np.array(Image.open(default_query_image_path)).astype(
        np.float
    )
    saliency_probability = deepgaze_predict_density(default_query_image)

    # integrate the saliency probability over the min and max patch position, respectively
    min_patch_saliency = np.mean(
        saliency_probability[
            min_position[0] : min_position[1], min_position[2] : min_position[3]
        ]
    )
    max_patch_saliency = np.mean(
        saliency_probability[
            max_position[0] : max_position[1], max_position[2] : max_position[3]
        ]
    )

    trial["min_query_patch_saliency"] = "%.12f" % min_patch_saliency
    trial["max_query_patch_saliency"] = "%.12f" % max_patch_saliency


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

# load deepgaze model
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
model = deepgaze2_dsrex3(pretrained=False)
model.to(device)
weights = torch.load(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "deepgaze_pytorch/DeepGazeII_DSREx3.pth",
    ),
    map_location=device,
)
model.load_state_dict(weights)
# use constant, i.e. uniform, center bias as suggested by Matthias
# (this means, the model prior assumes all pixel positions have the same saliency)
centerbias_log_probability = np.log(np.ones((224, 224)) / 224 / 224)


def deepgaze_predict_density(image):
    image_data = torch.tensor([image.transpose(2, 0, 1)], dtype=torch.float32).to(
        device
    )

    centerbias_data = torch.tensor([centerbias_log_probability] * len(image_data)).to(
        device
    )

    model_output = model.forward(x=image_data, centerbias=centerbias_data)

    log_density = (
        model_output.detach()  # make sure we're not carrying any gradients around
        .cpu()  # make sure we're not on the GPU
        .numpy()  # convert to numpy data
    )
    log_density = log_density[0, 0]

    return np.exp(log_density)


for task in tqdm(structure["tasks"]):
    for trial in task["raw_trials"]:
        process_trial(trial, occlusion_size_index)
    for trial in task["raw_catch_trials"]:
        process_trial(trial, occlusion_size_index)
    for trial in task["trials"]:
        process_trial(trial, occlusion_size_index)

with open(args.output_structure, "w") as f:
    json.dump(structure, f)
