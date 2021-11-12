import schema


def convert_to_update_schema(full_schema, create_schema=True):
    """Remove all defaults from schema to allow it to be used as update"""
    if isinstance(full_schema, schema.Schema):
        full_schema = full_schema.schema

    if isinstance(full_schema, dict):
        new_schema = {}
        for key, item in full_schema.items():
            if not isinstance(key, schema.Optional):
                # make optional
                key = schema.Optional(key)
            else:
                # make sure to create new optional without default instead of removing default from existing
                key = schema.Optional(key.key)

            if isinstance(item, (schema.Schema, dict)):
                item = convert_to_update_schema(item, create_schema=False)

            new_schema[key] = item

        if create_schema:
            new_schema = schema.Schema(new_schema)

        return new_schema

    return full_schema


number = schema.Schema(schema.Or(int, float))

readout_network_spec = schema.Schema(
    {
        "input_channels": schema.Or(int, [int]),
        "layers": [
            {
                "channels": int,
                "name": str,
                schema.Optional("activation_fn", default="relu"): schema.Or(None, str),
                schema.Optional("activation_scope", default=None): schema.Or(None, str),
                schema.Optional("bias", default=True): bool,
                schema.Optional("layer_norm", default=False): bool,
                schema.Optional("batch_norm", default=False): bool,
            }
        ],
    }
)

model_spec = schema.Schema(
    {
        "downscale_factor": number,
        "readout_factor": int,
        "saliency_map_factor": int,
        "included_previous_fixations": [int],
        "include_previous_x": bool,
        "include_previous_y": bool,
        "included_durations": [int],
        schema.Optional("fixated_scopes", default=[]): [str],
        "features": {
            str: {
                "type": str,
                schema.Optional("params", default={}): dict,
                "used_features": [str],
            }
        },
        "scanpath_network": schema.Or(readout_network_spec, None),
        "saliency_network": readout_network_spec,
        "fixation_selection_network": readout_network_spec,
    }
)

dataset_spec = schema.Schema(
    schema.Or(
        str,
        {
            schema.Optional("name"): str,
            schema.Optional("stimuli"): str,
            schema.Optional("fixations"): str,
            schema.Optional("centerbias"): str,
            schema.Optional("filters", default=[]): [dict],
        },
    )
)

crossvalidation_spec = schema.Schema(
    {
        "folds": int,
        "val_folds": int,
        "test_folds": int,
        schema.Optional("stratified_attributes", default=None): schema.Or(None, [str]),
    }
)

optimizer_spec = schema.Schema(
    {"type": str, schema.Optional("params", default={}): dict,}
)

lr_scheduler_spec = schema.Schema(
    {"type": str, schema.Optional("params", default={}): dict,}
)

evaluation_spec = schema.Schema(
    {
        schema.Optional("compute_metrics", default={}): {
            schema.Optional("metrics", default=["IG", "LL", "AUC", "NSS"]): [
                schema.Or("IG", "LL", "AUC", "NSS")
            ],
            schema.Optional("datasets", default=["training", "validation", "test"]): [
                schema.Or("training", "validation", "test")
            ],
        },
        schema.Optional("compute_predictions", default={}): schema.Or(
            {}, {"datasets": [schema.Or("training", "validation", "test")]}
        ),
    }
)


cleanup_spec = schema.Schema(
    {schema.Optional("cleanup_checkpoints", default=False): bool}
)


default_optimizer = optimizer_spec.validate(
    {"type": "torch.optim.Adam", "params": {"lr": 0.01}}
)

default_scheduler = lr_scheduler_spec.validate(
    {
        "type": "torch.optim.lr_scheduler.MultiStepScheduler",
        "params": {"milestones": [10, 20, 30, 40, 50, 60, 70, 80]},
    }
)


training_part_spec = schema.Schema(
    {
        "name": str,
        "train_dataset": dataset_spec,
        schema.Optional("optimizer"): convert_to_update_schema(optimizer_spec),
        schema.Optional("lr_scheduler"): convert_to_update_schema(lr_scheduler_spec),
        schema.Optional("minimal_learning_rate"): number,
        schema.Optional("iteration_element"): schema.Or("fixation", "image"),
        schema.Optional("averaging_element"): schema.Or("fixation", "image"),
        schema.Optional("model"): convert_to_update_schema(model_spec),
        schema.Optional("training_dataset_ratio_per_epoch"): float,
        schema.Optional("validation_epochs"): int,
        schema.Optional("centerbias"): str,
        schema.Optional("val_dataset"): dataset_spec,
        schema.Optional("test_dataset"): dataset_spec,
        schema.Optional("crossvalidation"): crossvalidation_spec,
        schema.Optional("validation_metric", default="IG"): schema.Or(
            "IG", "LL", "AUC", "NSS"
        ),
        schema.Optional("validation_metrics", default=["LL", "IG", "AUC", "NSS"]): [
            schema.Or("IG", "LL", "AUC", "NSS")
        ],
        schema.Optional("startwith", default=None): schema.Or(str, None),
        schema.Optional(
            "evaluation", default=evaluation_spec.validate({})
        ): evaluation_spec,
        schema.Optional("batch_size"): int,
        schema.Optional("cleanup", default=cleanup_spec.validate({})): cleanup_spec,
        schema.Optional(
            "final_cleanup", default=cleanup_spec.validate({})
        ): cleanup_spec,
    }
)

config_schema = schema.Schema(
    {
        "model": model_spec,
        "training": {
            schema.Optional("optimizer", default=default_optimizer): optimizer_spec,
            schema.Optional(
                "lr_scheduler", default=default_scheduler
            ): lr_scheduler_spec,
            schema.Optional("minimal_learning_rate", default=1.0e-7): number,
            schema.Optional("batch_size", default=2): int,
            schema.Optional("iteration_element", default="fixation"): schema.Or(
                "fixation", "image"
            ),
            schema.Optional("averaging_element", default="fixation"): schema.Or(
                "fixation", "image"
            ),
            schema.Optional("training_dataset_ratio_per_epoch", default=0.25): float,
            schema.Optional("validation_epochs", default=1): int,
            "parts": [training_part_spec],
        },
    }
)
