import os

from torch.utils import model_zoo
import yaml

from .deepgaze import FeatureExtractor, Finalizer, DeepGazeIIIMixture, MixtureModel
from .loading import import_class, build_readout_network_from_config


def build_deepgaze_mixture(model_config, components=10):
    assert len(model_config["features"]) == 1
    (features_key,) = list(model_config["features"].keys())
    features_config = model_config["features"][features_key]

    feature_class = import_class(features_config["type"])
    features = feature_class(**features_config["params"])

    feature_extractor = FeatureExtractor(features, features_config["used_features"])

    saliency_networks = []
    scanpath_networks = []
    fixation_selection_networks = []
    finalizers = []
    for component in range(components):
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

        saliency_networks.append(saliency_network)
        scanpath_networks.append(scanpath_network)
        fixation_selection_networks.append(fixation_selection_network)
        finalizers.append(
            Finalizer(
                sigma=8.0,
                learn_sigma=True,
                saliency_map_factor=model_config["saliency_map_factor"],
            )
        )

    return DeepGazeIIIMixture(
        features=feature_extractor,
        saliency_networks=saliency_networks,
        scanpath_networks=scanpath_networks,
        fixation_selection_networks=fixation_selection_networks,
        finalizers=finalizers,
        downsample=model_config["downscale_factor"],
        readout_factor=model_config["readout_factor"],
        saliency_map_factor=model_config["saliency_map_factor"],
        included_fixations=model_config["included_previous_fixations"],
    )


def deepgaze2_dsrex3(pretrained=True):
    config_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "deepgaze2_dsrex3.yaml"
    )
    with open(config_file) as f:
        configs = yaml.safe_load(f)

    backbone_models = [
        build_deepgaze_mixture(config, components=3 * 10) for config in configs
    ]
    model = MixtureModel(backbone_models)

    if pretrained:
        raise NotImplementedError()
        model.load_state_dict(model_zoo.load_url(None))

    return model
