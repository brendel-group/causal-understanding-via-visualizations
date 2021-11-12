# flake8: noqa E501
# pylint: disable=not-callable
# E501: line too long

from collections import defaultdict
from datetime import datetime
import glob
import os

from boltons.cacheutils import cached, LRU
from boltons.fileutils import mkdir_p
from boltons.iterutils import windowed
from IPython import get_ipython
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysaliency
from pysaliency.filter_datasets import iterate_crossvalidation
from pysaliency.plotting import visualize_distribution
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

from .data import (
    ImageDataset,
    FixationDataset,
    ImageDatasetSampler,
    FixationMaskTransform,
)
from .loading import (
    import_class,
    build_model,
    DeepGazeCheckpointModel,
    SharedPyTorchModel,
    _get_from_config,
)
from .metrics import log_likelihood, nss, auc


def display_if_in_IPython(*args, **kwargs):
    if get_ipython():
        display(*args, **kwargs)


baseline_performance = cached(LRU(max_size=3))(
    lambda model, *args, **kwargs: model.information_gain(*args, **kwargs)
)


def eval_epoch(
    model, dataset, device, baseline_model, metrics=None, averaging_element="fixation"
):
    print("Averaging element", averaging_element)
    model.eval()

    if metrics is None:
        metrics = ["LL", "IG", "NSS", "AUC"]

    metric_scores = {}
    metric_functions = {
        "LL": log_likelihood,
        "NSS": nss,
        "AUC": auc,
    }
    batch_weights = []

    with torch.no_grad():
        pbar = tqdm(dataset)
        for batch in pbar:
            image = batch["image"].to(device)
            # print(image.shape)
            centerbias = batch["centerbias"].to(device)
            fixation_mask = batch["fixation_mask"].to(device)
            x_hist = batch.get("x_hist", torch.tensor([])).to(device)
            y_hist = batch.get("y_hist", torch.tensor([])).to(device)
            weights = batch["weight"].to(device)
            durations = batch.get("durations", torch.tensor([])).to(device)

            log_density = model(
                image, centerbias, x_hist=x_hist, y_hist=y_hist, durations=durations
            )

            for metric_name, metric_fn in metric_functions.items():
                if metric_name not in metrics:
                    continue
                metric_scores.setdefault(metric_name, []).append(
                    metric_fn(log_density, fixation_mask, weights=weights)
                    .detach()
                    .cpu()
                    .numpy()
                )
            batch_weights.append(weights.detach().cpu().numpy().sum())

            for display_metric in ["LL", "NSS", "AUC"]:
                if display_metric in metrics:
                    pbar.set_description(
                        "{} {:.05f}".format(
                            display_metric,
                            np.average(
                                metric_scores[display_metric], weights=batch_weights
                            ),
                        )
                    )
                    break

    data = {
        metric_name: np.average(scores, weights=batch_weights)
        for metric_name, scores in metric_scores.items()
    }
    if "IG" in metrics:
        baseline_ll = baseline_performance(
            baseline_model,
            dataset.dataset.stimuli,
            dataset.dataset.fixations,
            verbose=True,
            average=averaging_element,
        )
        data["IG"] = data["LL"] - baseline_ll

    return data


def train_epoch(model, dataset, optimizer, device):
    model.train()
    losses = []
    batch_weights = []

    pbar = tqdm(dataset)
    for batch in pbar:
        optimizer.zero_grad()

        image = batch["image"].to(device)
        centerbias = batch["centerbias"].to(device)
        fixation_mask = batch["fixation_mask"].to(device)
        x_hist = batch.get("x_hist", torch.tensor([])).to(device)
        y_hist = batch.get("y_hist", torch.tensor([])).to(device)
        weights = batch["weight"].to(device)
        durations = batch.get("durations", torch.tensor([])).to(device)

        log_density = model(
            image, centerbias, x_hist=x_hist, y_hist=y_hist, durations=durations
        )

        loss = -log_likelihood(log_density, fixation_mask, weights=weights)
        losses.append(loss.detach().cpu().numpy())

        batch_weights.append(weights.detach().cpu().numpy().sum())

        pbar.set_description(
            "{:.05f}".format(np.average(losses, weights=batch_weights))
        )

        loss.backward()

        optimizer.step()

    return np.average(losses, weights=batch_weights)


def restore_from_checkpoint(model, optimizer, scheduler, path):
    print("Restoring from", path)
    data = torch.load(path)
    if "optimizer" in data:
        # checkpoint contains training progress
        model.load_state_dict(data["model"])
        optimizer.load_state_dict(data["optimizer"])
        scheduler.load_state_dict(data["scheduler"])
        torch.set_rng_state(data["rng_state"])
        return data["step"], data["loss"]
    else:
        # checkpoint contains just a model
        missing_keys, unexpected_keys = model.load_state_dict(data, strict=False)
        if missing_keys:
            print("WARNING! missing keys", missing_keys)
        if unexpected_keys:
            print("WARNING! Unexpected keys", unexpected_keys)


def save_training_state(model, optimizer, scheduler, step, loss, path):
    data = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng_state": torch.get_rng_state(),
        "step": step,
        "loss": loss,
    }

    torch.save(data, path)


def plot_scanpath(x_hist, y_hist, x, y, ax):
    for (x1, x2), (y1, y2) in zip(windowed(x_hist, 2), windowed(y_hist, 2)):
        if x1 == x2 and y1 == y2:
            continue
        ax.arrow(
            x1,
            y1,
            x2 - x1,
            y2 - y1,
            length_includes_head=True,
            head_length=20,
            head_width=20,
            color="red",
            zorder=10,
            linewidth=2,
        )

    x1 = x_hist[-1]
    y1 = y_hist[-1]
    x2 = x
    y2 = y
    ax.arrow(
        x1,
        y1,
        x2 - x1,
        y2 - y1,
        length_includes_head=True,
        head_length=20,
        head_width=20,
        color="blue",
        linestyle=":",
        linewidth=2,
        zorder=10,
    )


def visualize(model, vis_data_loader):
    model.eval()

    device = next(model.parameters()).device

    batch = next(iter(vis_data_loader))

    image = batch["image"].to(device)
    centerbias = batch["centerbias"].to(device)
    fixation_mask = batch["fixation_mask"].to(device)
    x_hist = batch.get("x_hist", torch.tensor([])).to(device)
    y_hist = batch.get("y_hist", torch.tensor([])).to(device)
    durations = batch.get("durations", torch.tensor([])).to(device)

    log_density = model(
        image, centerbias, x_hist=x_hist, y_hist=y_hist, durations=durations
    )

    log_density = log_density.detach().cpu().numpy()
    fixation_indices = fixation_mask.coalesce().indices().detach().cpu().numpy()
    rgb_image = image.detach().cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8)
    x_hist = x_hist.detach().cpu().numpy()
    y_hist = y_hist.detach().cpu().numpy()

    width = 4.0
    height = width / rgb_image.shape[2] * rgb_image.shape[1]
    f, axs = plt.subplots(
        len(rgb_image), 2, figsize=(2 * width, height * len(rgb_image))
    )

    for row in range(len(rgb_image)):
        axs[row, 0].imshow(rgb_image[row])

        bs, ys, xs = fixation_indices

        ys = ys[bs == row]
        xs = xs[bs == row]

        if len(x_hist):
            _x_hist = x_hist[row]
        else:
            _x_hist = []

        if len(y_hist):
            _y_hist = y_hist[row]
        else:
            _y_hist = []

        visualize_distribution(log_density[row], ax=axs[row, 1])

        if len(_x_hist):
            plot_scanpath(_x_hist, _y_hist, xs[0], ys[0], axs[row, 0])
            plot_scanpath(_x_hist, _y_hist, xs[0], ys[0], axs[row, 1])
        else:
            axs[row, 0].scatter(xs, ys)
            axs[row, 1].scatter(xs, ys)

        axs[row, 0].set_axis_off()
        axs[row, 1].set_axis_off()

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.01, hspace=0.01)

    return f


def train(
    this_directory,
    model,
    train_stimuli,
    train_fixations,
    train_baseline,
    val_stimuli,
    val_fixations,
    val_baseline,
    optimizer_config,
    lr_scheduler_config,
    minimum_learning_rate,
    # initial_learning_rate, learning_rate_scheduler, learning_rate_decay, learning_rate_decay_epochs, learning_rate_backlook, learning_rate_reset_strategy, minimum_learning_rate,
    batch_size=2,
    ratio_used=0.25,
    validation_metric="IG",
    validation_metrics=["IG", "LL", "AUC", "NSS"],
    iteration_element="image",
    averaging_element="fixation",
    validation_epochs=1,
    startwith=None,
):
    mkdir_p(this_directory)

    print("TRAINING DATASET", len(train_fixations.x))
    print("VALIDATION DATASET", len(val_fixations.x))

    if os.path.isfile(os.path.join(this_directory, "final.pth")):
        print("Training Already finished")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device", device)

    model.to(device)

    print("optimizer", optimizer_config)
    print("lr_scheduler", lr_scheduler_config)

    optimizer_class = import_class(optimizer_config["type"])
    optimizer = optimizer_class(model.parameters(), **optimizer_config["params"])

    scheduler_class = import_class(lr_scheduler_config["type"])
    scheduler = scheduler_class(optimizer, **lr_scheduler_config["params"])

    if iteration_element == "image":
        dataset_class = ImageDataset
    elif iteration_element == "fixation":
        dataset_class = lambda *args, **kwargs: FixationDataset(
            *args, **kwargs, included_fixations=model.included_fixations
        )

    train_dataset = dataset_class(
        train_stimuli,
        train_fixations,
        train_baseline,
        transform=FixationMaskTransform(),
        average=averaging_element,
    )
    val_dataset = dataset_class(
        val_stimuli,
        val_fixations,
        val_baseline,
        transform=FixationMaskTransform(),
        average=averaging_element,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=ImageDatasetSampler(
            train_dataset, batch_size=batch_size, ratio_used=ratio_used
        ),
        pin_memory=False,
        num_workers=0,  # doesn't work for sparse tensors yet. Might work soon.
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_sampler=ImageDatasetSampler(val_dataset, batch_size=batch_size),
        pin_memory=False,
        num_workers=0,
    )

    if iteration_element == "image":
        vis_stimuli, vis_fixations = pysaliency.create_subset(
            val_stimuli, val_fixations, list(range(batch_size))
        )
    if iteration_element == "fixation":
        vis_stimuli, vis_fixations = pysaliency.create_subset(
            val_stimuli, val_fixations, [0]
        )
        vis_fixations = vis_fixations[:batch_size]
    vis_dataset = dataset_class(
        vis_stimuli,
        vis_fixations,
        val_baseline,
        transform=FixationMaskTransform(),
        average=averaging_element,
    )
    vis_data_loader = torch.utils.data.DataLoader(
        vis_dataset,
        batch_sampler=ImageDatasetSampler(
            vis_dataset, batch_size=batch_size, shuffle=False
        ),
        pin_memory=False,
    )

    val_metrics = defaultdict(lambda: [])

    if startwith is not None:
        restore_from_checkpoint(model, optimizer, scheduler, startwith)

    writer = SummaryWriter(os.path.join(this_directory, "log"), flush_secs=30)

    columns = ["epoch", "timestamp", "learning_rate", "loss"]
    for metric in validation_metrics:
        columns.append(f"validation_{metric}")

    progress = pd.DataFrame(columns=columns)

    step = 0
    last_loss = np.nan

    def save_step():

        save_training_state(
            model,
            optimizer,
            scheduler,
            step,
            last_loss,
            "{}/step-{:04d}.pth".format(this_directory, step),
        )

        # f = visualize(model, vis_data_loader)
        # display_if_in_IPython(f)

        # writer.add_figure('prediction', f, step)
        writer.add_scalar("training/loss", last_loss, step)
        writer.add_scalar(
            "training/learning_rate",
            optimizer.state_dict()["param_groups"][0]["lr"],
            step,
        )
        writer.add_scalar(
            "parameters/sigma", model.finalizer.gauss.sigma.detach().cpu().numpy(), step
        )
        writer.add_scalar(
            "parameters/center_bias_weight",
            model.finalizer.center_bias_weight.detach().cpu().numpy()[0],
            step,
        )

        if step % validation_epochs == 0:
            _val_metrics = eval_epoch(
                model,
                val_loader,
                device,
                val_baseline,
                metrics=validation_metrics,
                averaging_element=averaging_element,
            )
        else:
            print("Skipping validation")
            _val_metrics = {}

        for key, value in _val_metrics.items():
            val_metrics[key].append(value)

        for key, value in _val_metrics.items():
            writer.add_scalar(f"validation/{key}", value, step)

        new_row = {
            "epoch": step,
            "timestamp": datetime.utcnow(),
            "learning_rate": optimizer.state_dict()["param_groups"][0]["lr"],
            "loss": last_loss,
            #'validation_ig': val_igs[-1]
        }
        for key, value in _val_metrics.items():
            new_row["validation_{}".format(key)] = value

        progress.loc[step] = new_row

        print(progress.tail(n=2))
        print(
            progress[["validation_{}".format(key) for key in val_metrics]].idxmax(
                axis=0
            )
        )

        progress.to_csv("{}/log.csv".format(this_directory))

        for old_step in range(1, step):
            # only check if we are computing validation metrics...
            if val_metrics[validation_metric] and old_step == np.argmax(
                val_metrics[validation_metric]
            ):
                continue
            for filename in glob.glob(
                "{}/step-{:04d}.pth".format(this_directory, old_step)
            ):
                print("removing", filename)
                os.remove(filename)

    old_checkpoints = sorted(glob.glob(os.path.join(this_directory, "step-*.pth")))
    if old_checkpoints:
        last_checkpoint = old_checkpoints[-1]
        print("Found old checkpoint", last_checkpoint)
        step, last_loss = restore_from_checkpoint(
            model, optimizer, scheduler, last_checkpoint
        )
        print("Setting step to", step)

    if step == 0:
        print("Beginning training")
        save_step()

    else:
        print("Continuing from step", step)
        progress = pd.read_csv(os.path.join(this_directory, "log.csv"), index_col=0)
        val_metrics = {}
        for column_name in progress.columns:
            if column_name.startswith("validation_"):
                val_metrics[column_name.split("validation_", 1)[1]] = list(
                    progress[column_name]
                )

        if step not in progress.epoch.values:
            print("Epoch not yet evaluated, evaluating...")
            save_step()

        print(progress)

    while optimizer.state_dict()["param_groups"][0]["lr"] >= minimum_learning_rate:
        step += 1
        last_loss = train_epoch(model, train_loader, optimizer, device)
        save_step()
        scheduler.step()

    # if learning_rate_reset_strategy == 'validation':
    #   best_step = np.argmax(val_metrics[validation_metric])
    #   print("Best previous validation in step {}, saving as final result".format(best_step))
    #   restore_from_checkpoint(model, optimizer, scheduler, os.path.join(this_directory, 'step-{:04d}.pth'.format(best_step)))
    # else:
    #    print("Not resetting to best validation epoch")

    torch.save(model.state_dict(), "{}/final.pth".format(this_directory))

    for filename in glob.glob(os.path.join(this_directory, "step-*")):
        print("removing", filename)
        os.remove(filename)


def _get_stimulus_filename(stimuli_stimulus):
    stimuli = stimuli_stimulus.stimuli
    index = stimuli_stimulus.index
    if isinstance(stimuli, pysaliency.FileStimuli):
        return stimuli.filenames[index]
    elif isinstance(stimuli, pysaliency.datasets.ObjectStimuli):
        return _get_stimulus_filename(stimuli.stimulus_objects[index])
    else:
        raise TypeError(
            "Stimuli of type {} don't have filenames!".format(type(stimuli))
        )


def get_filenames_for_stimuli(stimuli):
    if isinstance(stimuli, pysaliency.FileStimuli):
        return list(stimuli.filenames)
    if isinstance(stimuli, pysaliency.datasets.ObjectStimuli):
        return [_get_stimulus_filename(s) for s in stimuli.stimulus_objects]


def make_file_stimuli(stimuli):
    return pysaliency.FileStimuli(get_filenames_for_stimuli(stimuli))


def _get_dataset(dataset_config, training_config=None):
    """return stimuli, fixations, centerbias"""
    centerbias = None
    if isinstance(dataset_config, str):
        dataset_config = {"name": dataset_config, "filters": []}

    if "name" in dataset_config:
        raise ValueError("please specify filenames")
        # stimuli, fixations = get_dataset(dataset_config['name'], location='pysaliency_datasets')
    else:
        stimuli = pysaliency.read_hdf5(dataset_config["stimuli"])
        fixations = pysaliency.read_hdf5(dataset_config["fixations"])

    print("Loading stimulus ids")
    for _ in tqdm(stimuli.stimulus_ids):
        pass

    pysaliency_config = dict(dataset_config)
    centerbias_file = pysaliency_config.pop("centerbias", None)
    stimuli, fixations = pysaliency.load_dataset_from_config(pysaliency_config)

    centerbias_path = _get_from_config("centerbias", dataset_config, training_config)
    centerbias = pysaliency.HDF5Model(stimuli, centerbias_path)

    return stimuli, fixations, centerbias


def iterate_crossvalidation_config(stimuli, fixations, crossval_config):
    for (
        fold_no,
        (
            train_stimuli,
            train_fixations,
            val_stimuli,
            val_fixations,
            test_stimuli,
            test_fixations,
        ),
    ) in enumerate(
        iterate_crossvalidation(
            stimuli,
            fixations,
            crossval_folds=crossval_config["folds"],
            val_folds=crossval_config["val_folds"],
            test_folds=crossval_config["test_folds"],
            stratified_attributes=crossval_config["stratified_attributes"],
        )
    ):

        yield crossval_config[
            "folds"
        ], fold_no, train_stimuli, train_fixations, val_stimuli, val_fixations, test_stimuli, test_fixations


def run_training_part(
    root_directory, training_config, full_config, final_cleanup=False, args=None
):
    print("Running training part", training_config["name"])
    print("Configuration of this training part:")
    print(yaml.safe_dump(training_config))

    if "val_dataset" in training_config and "crossvalidation" in training_config:
        raise ValueError("Cannot specify both validation dataset and crossvalidation")

    directory = os.path.join(root_directory, training_config["name"])

    train_stimuli, train_fixations, train_centerbias = _get_dataset(
        training_config["train_dataset"], training_config
    )
    if "val_dataset" in training_config:
        assert "crossvalidation" not in training_config
        val_stimuli, val_fixations, val_centerbias = _get_dataset(
            training_config["val_dataset"], training_config
        )
        if "test_dataset" in training_config:
            test_stimuli, test_fixations, test_centerbias = _get_dataset(
                training_config["test_dataset"], training_config
            )
        else:
            test_stimuli = test_fixations = test_centerbias = None

        def iter_fn():
            return [
                {
                    "config": training_config,
                    "directory": directory,
                    "fold_no": None,
                    "crossval_folds": None,
                    "train_stimuli": train_stimuli,
                    "train_fixations": train_fixations,
                    "train_centerbias": train_centerbias,
                    "val_stimuli": val_stimuli,
                    "val_fixations": val_fixations,
                    "val_centerbias": val_centerbias,
                    "test_stimuli": test_stimuli,
                    "test_fixations": test_fixations,
                    "test_centerbias": test_centerbias,
                }
            ]

    else:
        assert "crossvalidation" in training_config

        def iter_fn():
            for (
                crossval_folds,
                fold_no,
                _train_stimuli,
                _train_fixations,
                _val_stimuli,
                _val_fixations,
                _test_stimuli,
                _test_fixations,
            ) in iterate_crossvalidation_config(
                train_stimuli, train_fixations, training_config["crossvalidation"]
            ):
                yield {
                    "config": training_config,
                    "directory": os.path.join(
                        directory, f"crossval-{crossval_folds}-{fold_no}"
                    ),
                    "fold_no": fold_no,
                    "crossval_folds": crossval_folds,
                    "train_stimuli": _train_stimuli,
                    "train_fixations": _train_fixations,
                    "train_centerbias": train_centerbias,
                    "val_stimuli": _val_stimuli,
                    "val_fixations": _val_fixations,
                    "val_centerbias": train_centerbias,
                    "test_stimuli": _test_stimuli,
                    "test_fixations": _test_fixations,
                    "test_centerbias": train_centerbias,
                }

    for part in iter_fn():

        if (
            args.crossval_fold_number is not None
            and part["fold_no"] != args.crossval_fold_number
        ):
            print("Skipping crossval fold number", part["fold_no"])
            continue

        model = build_model(training_config["model"])

        startwith = part["config"]["startwith"]
        if startwith is not None:
            startwith = startwith.format(
                root_directory=root_directory,
                crossval_folds=part["crossval_folds"],
                fold_no=part["fold_no"],
            )

        if not args.no_training:
            train(
                this_directory=part["directory"],
                model=model,
                train_stimuli=part["train_stimuli"],
                train_fixations=part["train_fixations"],
                train_baseline=part["train_centerbias"],
                val_stimuli=part["val_stimuli"],
                val_fixations=part["val_fixations"],
                val_baseline=part["val_centerbias"],
                optimizer_config=part["config"]["optimizer"],
                lr_scheduler_config=part["config"]["lr_scheduler"],
                minimum_learning_rate=part["config"]["minimal_learning_rate"],
                startwith=startwith,
                iteration_element=part["config"]["iteration_element"],
                averaging_element=part["config"]["averaging_element"],
                batch_size=part["config"]["batch_size"],
                ratio_used=part["config"]["training_dataset_ratio_per_epoch"],
                validation_epochs=part["config"]["validation_epochs"],
                validation_metrics=part["config"]["validation_metrics"],
                validation_metric=part["config"]["validation_metric"],
            )

        else:
            print("Skipping training")

    if final_cleanup:
        run_cleanup(iter_fn, directory, training_config["final_cleanup"])
        return

    if not args.no_evaluation:
        if training_config["evaluation"]:
            run_evaluation(iter_fn, directory, training_config)

        if training_config["cleanup"]:
            run_cleanup(iter_fn, directory, training_config["cleanup"])
    else:
        print("Skipping evaluation")


def run_evaluation(iter_fn, directory, training_config):
    evaluation_config = training_config["evaluation"]
    if evaluation_config["compute_metrics"]:
        compute_metrics(
            iter_fn, directory, evaluation_config["compute_metrics"], training_config
        )
    if evaluation_config["compute_predictions"]:
        compute_predictions(
            iter_fn,
            directory,
            evaluation_config["compute_predictions"],
            training_config,
        )


def _get_dataset_for_part(part, dataset):
    if dataset == "training":
        return part["train_stimuli"], part["train_fixations"], part["train_centerbias"]
    if dataset == "validation":
        return part["val_stimuli"], part["val_fixations"], part["val_centerbias"]
    if dataset == "test":
        return part["test_stimuli"], part["test_fixations"], part["test_centerbias"]
    raise ValueError(dataset)


def compute_metrics(iter_fn, directory, evaluation_config, training_config):
    metrics = []
    # metrics = {metric: {dataset: [] for dataset in evaluation_config['datasets']} for metric in evaluation_config['metrics']}
    weights = {dataset: [] for dataset in evaluation_config["datasets"]}

    results_file = os.path.join(directory, "results.csv")

    if not os.path.isfile(results_file):
        for part in iter_fn():
            this_directory = part["directory"]

            if os.path.isfile(os.path.join(this_directory, "results.csv")):
                results = pd.read_csv(
                    os.path.join(this_directory, "results.csv"), index_col=0
                )

                metrics.append(results)

                for dataset in evaluation_config["datasets"]:
                    _stimuli, _fixations, _ = _get_dataset_for_part(part, dataset)
                    weights[dataset].append(len(_fixations.x))

                continue

            model = build_model(training_config["model"])
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print("Using device", device)
            model.to(device)

            restore_from_checkpoint(
                model, None, None, os.path.join(this_directory, "final.pth")
            )

            this_results = {metric: {} for metric in evaluation_config["metrics"]}

            for dataset in evaluation_config["datasets"]:
                _stimuli, _fixations, _centerbias = _get_dataset_for_part(part, dataset)

                if part["config"]["iteration_element"] == "image":
                    dataset_class = ImageDataset
                elif part["config"]["iteration_element"] == "fixation":
                    dataset_class = lambda *args, **kwargs: FixationDataset(
                        *args, **kwargs, included_fixations=model.included_fixations
                    )
                else:
                    raise ValueError(part["config"]["iteration_element"])

                _dataset = dataset_class(
                    _stimuli,
                    _fixations,
                    _centerbias,
                    transform=FixationMaskTransform(),
                    average=part["config"]["averaging_element"],
                )

                loader = torch.utils.data.DataLoader(
                    _dataset,
                    batch_sampler=ImageDatasetSampler(
                        _dataset, batch_size=part["config"]["batch_size"], shuffle=False
                    ),
                    pin_memory=False,
                    num_workers=0,  # doesn't work for sparse tensors yet. Might work soon.
                )

                _results = eval_epoch(
                    model,
                    loader,
                    device,
                    _centerbias,
                    metrics=evaluation_config["metrics"],
                    averaging_element=part["config"]["averaging_element"],
                )

                for metric in evaluation_config["metrics"]:
                    this_results[metric][dataset] = _results[metric]

                if part["config"]["averaging_element"] == "fixation":
                    weights[dataset].append(len(_fixations.x))
                elif part["config"]["averaging_element"] == "image":
                    weights[dataset].append(len(_stimuli))

            result_df = pd.DataFrame(
                this_results, columns=evaluation_config["metrics"]
            ).loc[evaluation_config["datasets"]]
            result_df.to_csv(os.path.join(this_directory, "results.csv"))

            metrics.append(result_df)

        rows = []
        for dataset in evaluation_config["datasets"]:
            _weights = weights[dataset]
            relative_weights = np.array(_weights) / np.array(_weights).sum()
            _results = [df.loc[dataset] for df in metrics]
            _result = sum(weight * df for weight, df in zip(relative_weights, _results))
            rows.append(_result)

        result_df = pd.DataFrame(rows)

        result_df.to_csv(results_file)

        print(result_df)
    else:
        print(pd.read_csv(results_file, index_col=0))


def compute_predictions(iter_fn, directory, prediction_config, training_config):

    model = build_model(training_config["model"])
    shared_model = SharedPyTorchModel(model)

    for dataset in prediction_config["datasets"]:
        print(f"Computing predictions for dataset {dataset}")
        models = {}
        dataset_stimuli = []
        dataset_fixations = []
        for part in iter_fn():
            this_directory = part["directory"]
            checkpoint = os.path.join(this_directory, "final.pth")

            _stimuli, _fixations, _centerbias = _get_dataset_for_part(part, dataset)
            models[_stimuli] = DeepGazeCheckpointModel(
                shared_model, checkpoint, _centerbias
            )

            dataset_stimuli.append(_stimuli)
            dataset_fixations.append(_fixations)

        model = pysaliency.StimulusDependentModel(models, check_stimuli=False)
        stimuli, fixations = pysaliency.concatenate_datasets(
            dataset_stimuli, dataset_fixations
        )

        file_stimuli = make_file_stimuli(stimuli)
        pysaliency.export_model_to_hdf5(
            model, file_stimuli, os.path.join(directory, f"{dataset}_predictions.hdf5")
        )


def run_cleanup(iter_fn, directory, cleanup_config):
    if cleanup_config["cleanup_checkpoints"]:
        for part in iter_fn():
            this_directory = part["directory"]
            for filename in glob.glob(os.path.join(this_directory, "*.ckpt.*")):
                print("removing", filename)
                os.remove(filename)
