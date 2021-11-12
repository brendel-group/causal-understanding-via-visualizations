# first execute 10_get_query_logits.py to create a pkl file
# then use this script to calculate the top5/top1 accuracies

import pickle
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--input",
    required=True,
    help="Input pkl file created with 10_get_query_logits.py",
)
args = parser.parse_args()

with open(args.input, "rb") as f:
    data = pickle.load(f)

# only use the 1000 logits for the imagenet classes
logits = [it[0][:1000] for it in data]

top_5_predictions = [np.array(it).argsort()[-5:][::-1] for it in logits]
grouped_top_5_predictions = list(zip(*(iter(top_5_predictions),) * 3))

consistency_max = [(grp[0] == grp[1]).sum() for grp in grouped_top_5_predictions]
consistency_min = [(grp[0] == grp[2]).sum() for grp in grouped_top_5_predictions]

print("mean consistency (0-5) for max queries:", np.mean(consistency_max))
print("mean consistency (0-5) for min queries:", np.mean(consistency_min))

top_5_accuracy_max = np.mean([grp[0][0] in grp[1] for grp in grouped_top_5_predictions])
top_5_accuracy_min = np.mean([grp[0][0] in grp[2] for grp in grouped_top_5_predictions])

print(
    "top5 accuracy (wrt. top1 prediction of base image) for max queries:",
    np.mean(top_5_accuracy_max),
)
print(
    "top5 accuracy (wrt. top1 prediction of base image)  for min queries:",
    np.mean(top_5_accuracy_min),
)

top_5_top5_accuracy_max = np.mean(
    [
        len(set(grp[0]).intersection(set(grp[1]))) > 0
        for grp in grouped_top_5_predictions
    ]
)
top_5_top5_accuracy_min = np.mean(
    [
        len(set(grp[0]).intersection(set(grp[2]))) > 0
        for grp in grouped_top_5_predictions
    ]
)

print(
    "top5 accuracy (wrt. top5 prediction of base image) for max queries:",
    np.mean(top_5_top5_accuracy_max),
)
print(
    "top5 accuracy (wrt. top5 prediction of base image)  for min queries:",
    np.mean(top_5_top5_accuracy_min),
)
