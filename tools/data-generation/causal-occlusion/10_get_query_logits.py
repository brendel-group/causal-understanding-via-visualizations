# use this script to extract the softmax predictions for the base, max and min query images
# call this with:
#
# python 10_get_query_logits.py \
#   --data-dir=$DATAPATH/stimuli/stimuli_pure_conditions \
#   -o cfv_3_query_logits.pkl --n-batches=10
#
# to generate a pickle file containing the predictions

# %%
import glob
from typing import Callable, Optional, Any, Tuple, List

import torchvision.datasets
from torchvision.datasets import VisionDataset

print("starting 10_get_query_logits")

# %% md

# Investigate Occlusion Stimuli

# %% md

## Imports

# %%

# general imports
import numpy as np
from tqdm import tqdm
import pickle
import random
import tensorflow as tf
import torch
import os
import argparse

# %%

# lucid imports
import lucid.modelzoo.vision_models as models
from render import import_model

# %%

# custom imports
import occlusion_utils as ut

# %%
# define loader
class ListbasedDatasetFolder(VisionDataset):
    def __init__(
        self,
        root: str,
        file_names: List[str],
        loader: Callable[[str], Any] = torchvision.datasets.folder.pil_loader,
        extensions: Optional[Tuple[str, ...]] = None,
        transform: Optional[Callable] = None,
        path_transform: Optional[Callable] = None,
    ) -> None:
        super(ListbasedDatasetFolder, self).__init__(
            root, transform=transform, target_transform=path_transform
        )
        self.loader = loader
        self.extensions = extensions

        self.file_names = file_names

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path = self.file_names[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, path

    def __len__(self) -> int:
        return len(self.file_names)


# %% md

## Load model

# %%

# import InceptionV1 from the Lucid modelzoo
model = models.InceptionV1()
model.load_graphdef()

# %% md

## Parameters


# %%

# setting seeds
tf.set_random_seed(1234)
random.seed(0)
np.random.seed(0)

# %%

parser = argparse.ArgumentParser()
# # $DATAPATH/stimuli/stimuli_pure_conditions
parser.add_argument(
    "-d", "--data-dir", required=True, help="Path to load query images from"
)
parser.add_argument("-o", "--output", required=True, help="Path to save data to")
parser.add_argument("-nt", "--n-batches", default=10, type=int)
args = parser.parse_args()
print(args)

# %%

data_dir = args.data_dir
# %% md

## Load experiment specification

# %%

query_filenames = glob.glob(
    os.path.join(
        data_dir,
        "channel",
        "sampled_trials",
        "layer_**",
        "kernel_size_**",
        "channel_0",
        "natural_images",
        "batch_**",
        "40_percent_side_length",
        "*.png",
    )
)
task_filter = lambda fn: int(fn.split("batch_")[1].split("/")[0]) < args.n_batches
query_filenames = [fn for fn in query_filenames if task_filter(fn)]

# sort to make sure we always have the order: base.png, max.png, min.png for each trial and task
query_filenames = list(sorted(query_filenames))
# %%

dataset = ListbasedDatasetFolder(
    data_dir, query_filenames, transform=torchvision.transforms.ToTensor()
)
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
)


# %%
activations = []
paths = []
with tf.Graph().as_default() as graph, tf.Session() as sess:
    image = tf.placeholder(tf.float32, shape=(1, 224, 224, 3))
    print("image.shape", image.shape)
    model_instance = import_model(model, image)

    for images, paths_batch in tqdm(data_loader):
        images = images.numpy().transpose((0, 2, 3, 1))
        activations_batch = sess.run(model_instance("softmax2"), {image: images})
        activations.append(activations_batch)
        paths.append(paths_batch)

activations = np.concatenate(activations).tolist()
paths = np.concatenate(paths).tolist()
data = list(zip(activations, paths))

with open(args.output, "wb") as f:
    pickle.dump(data, f)
# %%
