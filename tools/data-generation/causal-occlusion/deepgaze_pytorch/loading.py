from collections.abc import Mapping
from collections import OrderedDict
from copy import deepcopy
import importlib
import os

from .deepgaze import DeepGazeIII as TorchDeepGazeIII, FeatureExtractor
from .layers import LayerNorm, Conv2dMultiInput, Bias, LayerNormMultiInput
from glom import glom
import numpy as np
import pysaliency
from pysaliency.datasets import as_stimulus
from pysaliency.filter_datasets import iterate_crossvalidation
from pysaliency.utils import remove_trailing_nans
import torch
import torch.nn as nn

import yaml

from .config import config_schema


def dict_merge(dct, merge_dct):
    """ Recursive dict merge. The ``merge_dct`` is merged into
    ``dct``.
    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :return: None
    """
    for k, v in merge_dct.items():
        if k in dct and isinstance(dct[k], dict) and isinstance(merge_dct[k], Mapping):
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]

    return dct


def reverse_dict_merge(dct, fallback_dct):
    """ Recursive dict merge. The ``merge_dct`` is merged into
    ``dct``.
    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :return: None
    """
    for k, v in fallback_dct.items():
        if (
            k in dct
            and isinstance(dct[k], dict)
            and isinstance(fallback_dct[k], Mapping)
        ):
            reverse_dict_merge(dct[k], fallback_dct[k])
        elif k in dct:
            # don't fallback
            pass
        else:
            dct[k] = fallback_dct[k]

    return dct


def expand_config(config):
    bare_training_config = dict(config["training"])
    del bare_training_config["parts"]
    bare_training_config["model"] = config["model"]

    for part in config["training"]["parts"]:
        reverse_dict_merge(part, deepcopy(bare_training_config))

    config_schema.validate(config)
    return config


def get_experiment_config(experiment_directory):
    config = yaml.safe_load(open(os.path.join(experiment_directory, "config.yaml")))
    config = config_schema.validate(config)

    return expand_config(config)


def build_readout_network_from_config(readout_config):
    layers = OrderedDict()
    input_channels = readout_config["input_channels"]

    for k, layer_spec in enumerate(readout_config["layers"]):
        if layer_spec["layer_norm"]:
            if isinstance(input_channels, int):
                layers[f"layernorm{k}"] = LayerNorm(input_channels)
            else:
                layers[f"layernorm{k}"] = LayerNormMultiInput(input_channels)

        if isinstance(input_channels, int):
            layers[f"conv{k}"] = nn.Conv2d(
                input_channels, layer_spec["channels"], (1, 1), bias=False
            )
        else:
            layers[f"conv{k}"] = Conv2dMultiInput(
                input_channels, layer_spec["channels"], (1, 1), bias=False
            )
        input_channels = layer_spec["channels"]

        assert not layer_spec["batch_norm"]

        if layer_spec["bias"]:
            layers[f"bias{k}"] = Bias(input_channels)

        if layer_spec["activation_fn"] == "relu":
            layers[f"relu{k}"] = nn.ReLU()
        elif layer_spec["activation_fn"] == "softplus":
            layers[f"softplus{k}"] = nn.Softplus()
        elif layer_spec["activation_fn"] is None:
            pass
        else:
            raise ValueError(layer_spec["activation_fn"])

    return nn.Sequential(layers)


def import_class(name):
    module_name, class_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def build_model(model_config):
    assert len(model_config["features"]) == 1
    (features_key,) = list(model_config["features"].keys())
    features_config = model_config["features"][features_key]

    feature_class = import_class(features_config["type"])
    features = feature_class(**features_config["params"])

    feature_extractor = FeatureExtractor(features, features_config["used_features"])

    saliency_network = build_readout_network_from_config(
        model_config["saliency_network"]
    )
    if model_config["scanpath_network"] is not None:
        scanpath_network = build_readout_network_from_config(
            model_config["scanpath_network"]
        )
    else:
        scanpath_network = None
    fixation_selection_network = build_readout_network_from_config(
        model_config["fixation_selection_network"]
    )
    model = TorchDeepGazeIII(
        features=feature_extractor,
        saliency_network=saliency_network,
        scanpath_network=scanpath_network,
        fixation_selection_network=fixation_selection_network,
        downsample=model_config["downscale_factor"],
        readout_factor=model_config["readout_factor"],
        saliency_map_factor=model_config["saliency_map_factor"],
        included_fixations=model_config["included_previous_fixations"],
    )

    for scope in model_config["fixated_scopes"]:
        for parameter_name, parameter in model.named_parameters():
            if parameter_name.startswith(scope):
                print("Fixating parameter", parameter_name)
                parameter.requires_grad = False

    print("Remaining training parameters")
    for parameter_name, parameter in model.named_parameters():
        if parameter.requires_grad:
            print(parameter_name)

    return model


def _get_from_config(key, *configs, **kwargs):
    """get config keys with fallbacks"""
    for config in configs:
        try:
            return glom(config, key, **kwargs)
        except KeyError:
            pass
    raise KeyError(key, configs)


# assert _get_from_config('a.b', {'a': {'c': 1}}, {'a': {'b': 2}}) == 2


def _get_dataset(dataset_config, training_config=None):
    """return stimuli, fixations, centerbias"""
    centerbias = None
    if isinstance(dataset_config, str):
        dataset_config = {"name": dataset_config, "filters": []}

    stimuli = pysaliency.read_hdf5(dataset_config["stimuli"])
    fixations = pysaliency.read_hdf5(dataset_config["fixations"])

    pysaliency_config = dict(dataset_config)
    pysaliency_config.pop("centerbias", None)
    stimuli, fixations = pysaliency.load_dataset_from_config(pysaliency_config)

    centerbias_path = _get_from_config("centerbias", dataset_config, training_config)
    centerbias = pysaliency.HDF5Model(stimuli, centerbias_path, check_shape=True)

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
        )
    ):

        yield crossval_config[
            "folds"
        ], fold_no, train_stimuli, train_fixations, val_stimuli, val_fixations, test_stimuli, test_fixations


def _get_dataset_for_part(part, dataset):
    if dataset == "training":
        return part["train_stimuli"], part["train_fixations"], part["train_centerbias"]
    if dataset == "validation":
        return part["val_stimuli"], part["val_fixations"], part["val_centerbias"]
    if dataset == "test":
        return part["test_stimuli"], part["test_fixations"], part["test_centerbias"]
    raise ValueError(dataset)


def get_training_part_iter(root_directory, training_config, full_config):
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

    return iter_fn


class SharedPyTorchModel(object):
    def __init__(self, model, device=None):
        self.model = model
        self.device = device
        self.active_checkpoint = None

    def load_checkpoint(self, checkpoint_path):
        if self.active_checkpoint != checkpoint_path:
            print("Loading checkpoint", checkpoint_path)
            data = torch.load(checkpoint_path, map_location=self.device)
            print("Activating checkpoint")
            self.model.load_state_dict(data)
            print("done")
            self.active_checkpoint = checkpoint_path


class DeepGazeCheckpointScanpathModel(pysaliency.ScanpathModel):
    def __init__(self, shared_model, checkpoint, centerbias_model):
        super().__init__()

        self.checkpoint = checkpoint
        self.centerbias_model = centerbias_model
        self.shared_model = shared_model

    def conditional_log_density(
        self, stimulus, x_hist, y_hist, t_hist, attributes=None, out=None
    ):
        x_hist = np.asarray(remove_trailing_nans(x_hist))
        y_hist = np.asarray(remove_trailing_nans(y_hist))
        t_hist = np.asarray(remove_trailing_nans(t_hist))

        assert len(x_hist) == len(y_hist) == len(t_hist)

        included_fixations = self.shared_model.model.included_fixations

        self.shared_model.load_checkpoint(self.checkpoint)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        images = torch.tensor(
            [as_stimulus(stimulus).stimulus_data.transpose(2, 0, 1)],
            dtype=torch.float32,
        ).to(device)
        centerbiases = torch.tensor(
            [self.centerbias_model.log_density(stimulus)], dtype=torch.float32
        ).to(device)
        x_hist = torch.tensor([x_hist[included_fixations]], dtype=torch.float32).to(
            device
        )
        y_hist = torch.tensor([y_hist[included_fixations]], dtype=torch.float32).to(
            device
        )
        durations = torch.tensor([]).to(device)

        log_density = (
            self.shared_model.model.forward(
                images, centerbiases, x_hist=x_hist, y_hist=y_hist, durations=durations
            )[0, :, :]
            .detach()
            .cpu()
            .numpy()
        )

        return log_density


class DeepGazeCheckpointModel(pysaliency.Model):
    def __init__(self, shared_model, checkpoint, centerbias_model):
        super().__init__(caching=False)

        self.checkpoint = checkpoint
        self.centerbias_model = centerbias_model
        self.shared_model = shared_model

    def _log_density(self, stimulus):
        self.shared_model.load_checkpoint(self.checkpoint)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        images = torch.tensor([stimulus.transpose(2, 0, 1)], dtype=torch.float32).to(
            device
        )
        centerbiases = torch.tensor(
            [self.centerbias_model.log_density(stimulus)], dtype=torch.float32
        ).to(device)

        log_density = (
            self.shared_model.model.forward(images, centerbiases)[0, :, :]
            .detach()
            .cpu()
            .numpy()
        )

        return log_density


def build_pysaliency_model(
    iter_fn,
    directory,
    training_config,
    dataset_name="test",
    centerbias=None,
    fallback_to_mixture=False,
    check_stimuli=True,
    static=None,
):
    if static is None:
        static = not len(training_config["model"]["included_previous_fixations"])

    if static:
        model_class = DeepGazeCheckpointModel
        mixture_model = pysaliency.models.MixtureModel
        stimulus_dependent_model = pysaliency.models.StimulusDependentScanpathModel
    else:
        model_class = DeepGazeCheckpointScanpathModel
        mixture_model = pysaliency.models.MixtureScanpathModel
        stimulus_dependent_model = pysaliency.models.StimulusDependentModel

    model = build_model(training_config["model"])

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device", device)
    model.to(device)

    shared_model = SharedPyTorchModel(model, device=device)

    models = OrderedDict()
    dataset_stimuli = []
    dataset_fixations = []
    for part in iter_fn():
        this_directory = part["directory"]
        checkpoint = os.path.join(this_directory, "final.pth")

        _stimuli, _fixations, _centerbias = _get_dataset_for_part(part, dataset_name)
        if centerbias is not None:
            _centerbias = centerbias
        models[_stimuli] = model_class(shared_model, checkpoint, _centerbias)

        dataset_stimuli.append(_stimuli)
        dataset_fixations.append(_fixations)

    if fallback_to_mixture:
        fallback_model = mixture_model(list(models.values()), check_norm=False)
    else:
        fallback_model = None

    model = stimulus_dependent_model(
        models, fallback_model=fallback_model, check_stimuli=check_stimuli
    )

    return model


def build_deepgaze_from_experiment(
    experiment_directory,
    training_part,
    dataset_name="test",
    centerbias=None,
    fallback_to_mixture=False,
    check_stimuli=True,
    static=None,
):
    config = get_experiment_config(experiment_directory)

    (part_config,) = [
        part for part in config["training"]["parts"] if part["name"] == training_part
    ]
    iter_fn = get_training_part_iter(experiment_directory, part_config, config)

    model = build_pysaliency_model(
        iter_fn,
        experiment_directory,
        part_config,
        dataset_name=dataset_name,
        centerbias=centerbias,
        fallback_to_mixture=fallback_to_mixture,
        check_stimuli=check_stimuli,
        static=static,
    )

    return model


def load_pytorch_model_from_experiment(
    experiment_directory,
    training_part,
    crossval_fold=None,
    device=None,
    load_checkpoint=True,
):
    config = get_experiment_config(experiment_directory)
    (part_config,) = [
        part for part in config["training"]["parts"] if part["name"] == training_part
    ]

    model = build_model(part_config["model"])

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device", device)
    model.to(device)

    if load_checkpoint:
        checkpoint_path = [experiment_directory, training_part]
        if "crossvalidation" in part_config:
            checkpoint_path.append(
                f"crossval-{part_config['crossvalidation']['folds']}-{crossval_fold}"
            )
        checkpoint_path.append("final.pth")

        data = torch.load(os.path.join(*checkpoint_path), map_location=device)
        model.load_state_dict(data, strict=False)

    return model
