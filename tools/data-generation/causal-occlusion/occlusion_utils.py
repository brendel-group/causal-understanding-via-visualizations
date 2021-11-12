import numpy as np

# for dataloader
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import csv


# Parameters
objective = "channel"
n_batches = 20  # same as in ICLR experiments
# for occlusions:
heatmap_sizes_list = [80, 68, 57]
occlusion_sizes_list = [66, 90, 112]
percentage_side_length_list = [30, 40, 50]


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = original_tuple + (path,)
        return tuple_with_path


def get_data_loader(data_dir, do_resize_and_center_crop=True):
    # make deterministic
    torch.manual_seed(1234)

    if do_resize_and_center_crop:
        transform_function = transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
        )
    else:
        transform_function = transforms.Compose([transforms.ToTensor()])

    # preprocessing (corresponds to ResNet)
    this_dataset = ImageFolderWithPaths(data_dir, transform_function)

    data_loader = torch.utils.data.DataLoader(
        this_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
    )

    torch.set_grad_enabled(
        True
    )  # save memory and computation cost by not calculating the grad

    return data_loader


def get_number_of_stimuli(stimuli_dir):
    """Note that for the natural images, it makes sense to think of the batch
    size as the number of bins"""
    # for pure conditions (natural or optimized):
    if "pure" in stimuli_dir:
        n_reference_images = 9
        batch_size_natural = (
            n_reference_images + 1
        )  # number of reference images (9) + 1 query image
        batch_size_optimized = n_reference_images
    # for joint condition:
    elif "mixed" in stimuli_dir:
        n_reference_images = 5
        batch_size_natural = (
            n_reference_images + 1
        )  # number of reference images (9) + 1 query image
        batch_size_optimized = n_reference_images - 1
    else:
        raise ValueError("You are not choosing the correct stimuli_dir!")

    return n_reference_images, batch_size_natural, batch_size_optimized


def get_list_of_occlusion_positions(heatmap_size_i, occlusion_size_i):
    """generate a list of all occlusion positions with stride 2.
    The stride of 2 is something we decided to guarantee reasonable compute time.

    Args:
        heatmap_size_i:     size of heatmap
        occlusion_size_i:   size of occlusion
    Returns:
        list_of_positions:  list of positions
                            size: heatmap_size_i * heatmap_size_i.
                            one list entry is a tuple of the x and y
                            start and stop positions, e.g.
                            [(0, 66, 0, 66), (0, 66, 2, 68), ...,
                                (158, 224, 156, 222), (158, 224, 158, 224)]
                            for heatmap_size_i = 80, occlusion_size_i = 66
    """

    list_of_positions = []

    # for all positions on the horizontal axis ...
    for occlusion_position_x in range(heatmap_size_i):
        # and all positions on the vertical axis ...
        for occlusion_position_y in range(heatmap_size_i):
            # determine the start and end position of the occlusion patch
            x_start = 2 * occlusion_position_x
            x_end = x_start + occlusion_size_i
            y_start = 2 * occlusion_position_y
            y_end = y_start + occlusion_size_i

            list_of_positions.append((x_start, x_end, y_start, y_end))

    return list_of_positions


def get_tf_activations_list_whole_net(model_instance, unit_specs_df):
    """create model_instance for each unique combinations of layer_name-pre_post_relu and return this list"""

    # create list of combinations of layer_name-pre_post_relu
    layer_str_list = []
    for idx, row in unit_specs_df.iterrows():
        layer_str = f"{row['layer_name']}_{row['pre_post_relu']}"
        layer_str_list.append(layer_str)

    # create model_instance for each unique combinations of layer_name-pre_post_relu
    tf_activations_list = []
    unique_layer_str_list = list(set(layer_str_list))
    unique_layer_str_list.sort()
    for cur_layer_str in unique_layer_str_list:
        tf_activations_list.append(model_instance(cur_layer_str))

    return tf_activations_list, unique_layer_str_list


def get_tf_activations_list(model_instance, layer_name, pre_post_relu):
    """create model_instance for each unique combinations of layer_name-pre_post_relu and return this list"""

    # create list of combinations of layer_name-pre_post_relu
    layer_str_list = [f"{layer_name}_{pre_post_relu}"]

    # create model_instance for each unique combinations of layer_name-pre_post_relu
    tf_activations_list = []
    unique_layer_str_list = list(set(layer_str_list))
    unique_layer_str_list.sort()
    for cur_layer_str in unique_layer_str_list:
        tf_activations_list.append(model_instance(cur_layer_str))

    return tf_activations_list, unique_layer_str_list


def get_activation_according_to_objective(
    objective, activations_np, feature_map_number
):
    # get activations according to objective
    if objective == "neuron":
        # neuron number
        filter_location = activations_np.shape[2] // 2
        unit_activations = activations_np[
            :, filter_location, filter_location, feature_map_number
        ]  # batch_size, x, y, number_feature_maps
    elif objective == "channel":
        unit_activations = np.mean(np.mean(activations_np, axis=1), axis=1)[
            :, feature_map_number
        ]  # batch_size

    return unit_activations


def write_unit_activations_to_csv(
    unit_activations, path_activations_whole_dataset_csv, paths, targets
):
    # convert torch tensor to list
    targets_list = targets.tolist()
    # if file does not exist yet, initialize csv and write
    if not os.path.isfile(path_activations_whole_dataset_csv):
        with open(path_activations_whole_dataset_csv, "w") as csvFile:
            csv_writer = csv.writer(csvFile, delimiter=",", lineterminator="\n")
            csv_writer.writerow(["activations from whole batch"])
            csv_writer.writerow(["path to image", "activation", "target class"])
            for csv_row, path_i in enumerate(paths):
                csv_writer.writerow(
                    [path_i, unit_activations[csv_row].tolist(), targets_list[csv_row]]
                )
    # append to csv
    else:
        with open(path_activations_whole_dataset_csv, "a") as csvFile:
            csv_writer = csv.writer(csvFile, delimiter=",", lineterminator="\n")
            for csv_row, path_i in enumerate(paths):
                csv_writer.writerow(
                    [path_i, unit_activations[csv_row].tolist(), targets_list[csv_row]]
                )


def create_unit_activations_dataset_rows(unit_activations, paths, targets):
    # convert torch tensor to list
    targets_list = targets.tolist()

    results = []
    for csv_row, path_i in enumerate(paths):
        results.append(
            {
                "path to image": path_i,
                "activation": unit_activations[csv_row].tolist(),
                "target class": targets_list[csv_row],
            }
        )
    return results


def remove_white_margins(ax):
    import matplotlib.pyplot as plt

    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
