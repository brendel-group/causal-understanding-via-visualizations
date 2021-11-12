import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import utils_figures_helper as utf_helper
import utils_figures as utf


file_type = ".pdf"
fontsize_axes_labels = 10
fontsize_tick_labels = 8
x_tick_label_rotation = 30
error_bar_linewidth = 1
uniform_sizing_labels_list = ["Hand-\nPicked", "Min+Max 9"]
uniform_sizing_labels_list = ["Min+Max 9"]  # "Hand-\nPicked",
sharey = True
dh_01 = 0.03
dh_12 = 0.17
dh_02 = 0.33


def plot_row_variability_details_upper_extracted(
    df, criteria_pass, results_folder, save_fig
):
    plt.figure(figsize=(4, 3))
    plt.tick_params(axis="both", which="major", labelsize=12)
    if criteria_pass:
        criteria_str = "Included"
        plt.title(f"Included Data, 0 < Number of Trials < 21")
    else:
        criteria_str = "Excluded"
        plt.title(f"Excluded Data, $\\neg$(0 < Number of Trials < 21)")
    plt.hist(
        df[df["row_variability_result"] == criteria_pass][
            "row_variability_details_details_upper_exctracted"
        ]
    )
    # no upper and right axis
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().set_xlim(0, 12)  # this is over 9 main + 3 catch trials
    plt.tight_layout()
    # plt.title(f"Histogram over Row Variability\n{criteria_str} Data\nExclusion Criterion >= 1 trial from less frequently selected row")

    plt.ylabel("Number of Tasks", fontsize=12)
    plt.xlabel("Number of Trials Left Image was Chosen", fontsize=12)

    plot_name = "exclusion_criterion_row_variability_details_ratio_extracted"
    for version in range(100):
        file_name = os.path.join(
            results_folder,
            f"{plot_name}_{criteria_str.lower()}_data_{version}{file_type}",
        )
        # if file_name does not yet exist, use it
        if not os.path.exists(file_name):
            break
    if save_fig:
        print(f"figure saved under {file_name}")
        plt.savefig(file_name, bbox_inches="tight")

    plt.show()


def plot_instruction_time_details_extracted(
    df, criteria_pass, results_folder, save_fig
):
    plt.figure(figsize=(4, 3))
    plt.tick_params(axis="both", which="major", labelsize=12)
    if criteria_pass:
        criteria_str = "Included"
        plt.title(f"{criteria_str} Data, 15s $<$ Time")
    else:
        criteria_str = "Excluded"
        plt.title(f"{criteria_str} Data, 15s $\\nless$ Time")
    plt.hist(
        df[df["instruction_time_result"] == criteria_pass][
            "instruction_time_details_extracted"
        ]
    )
    plt.tight_layout()
    # plt.title(f"Histogram over Instruction Time\n{criteria_str} Data\nExclusion Criterion > 15s")

    plt.ylabel("Number of Tasks", fontsize=12)
    plt.xlabel("Time [s]", fontsize=12)

    # no axis on top and right
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plot_name = "exclusion_criterion_instruction_time_details_extracted"
    for version in range(100):
        file_name = os.path.join(
            results_folder,
            f"{plot_name}_{criteria_str.lower()}_data_{version}{file_type}",
        )
        # if file_name does not yet exist, use it
        if not os.path.exists(file_name):
            break
    if save_fig:
        print(f"figure saved under {file_name}")
        plt.savefig(file_name, bbox_inches="tight")

    plt.show()


def plot_total_response_time_details_extracted(
    df, criteria_pass, results_folder, save_fig
):
    plt.figure(figsize=(4, 3))
    plt.tick_params(axis="both", which="major", labelsize=12)
    if criteria_pass:
        criteria_str = "Included"
        plt.title(f"{criteria_str} Data, 90s < Time < 900s")
    else:
        criteria_str = "Excluded"
        plt.title(f"{criteria_str} Data, $\\neg$(90s < Time < 900s)")
    plt.hist(
        df[df["total_response_time_result"] == criteria_pass][
            "total_response_time_details_extracted"
        ],
        bins=12,
    )
    plt.tight_layout()

    plt.ylabel("Number of Tasks", fontsize=12)
    plt.xlabel("Time [s]", fontsize=12)

    # no axis on top and right
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plot_name = "exclusion_criterion_total_response_time_details_extracted"
    for version in range(100):
        file_name = os.path.join(
            results_folder,
            f"{plot_name}_{criteria_str.lower()}_data_{version}{file_type}",
        )
        # if file_name does not yet exist, use it
        if not os.path.exists(file_name):
            break
    if save_fig:
        print(f"figure saved under {file_name}")
        plt.savefig(file_name, bbox_inches="tight")

    plt.show()


def plot_catch_trials_details_ratio_exctracted(
    df, criteria_pass, results_folder, save_fig
):
    plt.figure(figsize=(4, 3))
    plt.tick_params(axis="both", which="major", labelsize=12)
    if criteria_pass:
        criteria_str = "Included"
        plt.title(f"{criteria_str} Data, Ratio > 0.66")
    else:
        criteria_str = "Excluded"
        plt.title(f"{criteria_str} Data, Ratio $\\ngtr$ 0.66")

    plt.hist(
        df[df["catch_trials_result"] == criteria_pass][
            "catch_trials_details_ratio_exctracted"
        ]
    )
    plt.tight_layout()

    plt.ylabel("Number of Tasks", fontsize=12)
    plt.xlabel("Ratio of Correct Trials", fontsize=12)

    # no axis on top and right
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plot_name = "exclusion_criterion_catch_trials_details_ratio_extracted"
    for version in range(100):
        file_name = os.path.join(
            results_folder,
            f"{plot_name}_{criteria_str.lower()}_data_{version}{file_type}",
        )
        # if file_name does not yet exist, use it
        if not os.path.exists(file_name):
            break
    if save_fig:
        print(f"figure saved under {file_name}")
        plt.savefig(file_name, bbox_inches="tight")

    plt.show()


def plot_failed_trials_histogram(
    df,
    is_catch_trials,
    results_folder,
    save_fig,
    instr_type_list=["optimized", "natural", "mixed"],
):
    instr_type_str = "mode"
    instr_type_list

    available_layers = sorted(df["layer"].unique().tolist())
    bar_width = 1
    offsets = [bar_width * idx for idx in range(len(instr_type_list))]
    for i, reference_type_i in enumerate(instr_type_list):
        df_factor_i = df[df[instr_type_str] == reference_type_i].copy()
        data = df_factor_i[~df_factor_i["correct"]]["layer"].value_counts()

        data = {k: data[k] if k in data else 0 for k in available_layers}

        y_values = [data[al] for al in available_layers]
        x_values = [
            x + offsets[i]
            for x in np.arange(0, len(available_layers))
            * ((len(instr_type_list) + 1) * bar_width)
        ]

        plt.bar(x_values, y_values, color=utf.colors[reference_type_i], width=bar_width)

    plt.tight_layout()

    plt.xticks(
        np.arange(0, len(available_layers)) * ((len(instr_type_list) + 1) * bar_width)
        + 1,
        available_layers,
    )

    plt.ylabel("Number of Failed Trials", fontsize=12)
    plt.xlabel("Feature map used for Trial", fontsize=12)

    # no axis on top and right
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    if is_catch_trials:
        plt.title("Histogram over failed Catch Trials")
        plot_name = "catch_trials_histogram_failed"
    else:
        plt.title("Histogram over failed Trials")
        plot_name = "trials_histogram_failed"
    for version in range(100):
        file_name = os.path.join(
            results_folder, f"{plot_name}_data_{version}{file_type}"
        )
        # if file_name does not yet exist, use it
        if not os.path.exists(file_name):
            break
    if save_fig:
        print(f"figure saved under {file_name}")
        plt.savefig(file_name, bbox_inches="tight")

    plt.show()


def plot_accuracy_per_layer(
    df,
    results_folder,
    save_fig,
    instr_type_list=["optimized", "natural", "mixed"],
    title_prefix="",
    legend=True,
    show_sem=False,
):
    fig, ax = plt.subplots()

    instr_type_str = "mode"

    available_layers = sorted(df["layer"].unique().tolist())

    catch_trial_layer_map = {"j": 6, "t": 4, "z": 7}
    available_layers_labels = [
        int(catch_trial_layer_map.get(k, k)) for k in available_layers
    ]

    # sort both labels and raw layer info by layer label
    available_layers, available_layers_labels = zip(
        *sorted(
            list(zip(available_layers, available_layers_labels)), key=lambda x: x[0]
        )
    )

    fig.set_size_inches((0.5 + 3.6 / 9 * len(available_layers), 3))

    bar_width = 1
    offsets = [bar_width * idx for idx in range(len(instr_type_list))]
    for i, reference_type_i in enumerate(instr_type_list):
        df_factor_i = df[df[instr_type_str] == reference_type_i].copy()
        data = df_factor_i[df_factor_i["correct"]]["layer"].value_counts()

        data = {k: data[k] if k in data else 0 for k in available_layers}

        normalizer = df_factor_i["layer"].value_counts()

        y_values = [data[al] / normalizer[al] for al in available_layers]
        x_values = [
            x + offsets[i]
            for x in np.arange(0, len(available_layers))
            * ((len(instr_type_list) + 1) * bar_width)
        ]

        if show_sem:
            # calculate errors
            df_factor_i["imageset"] = df_factor_i["batch"] % 10
            y_error_data = {
                k: df_factor_i[df_factor_i["layer"] == k].groupby("imageset")["correct"]
                for k in available_layers
            }
            # get mean per imageset
            y_error_mean_data = {k: y_error_data[k].mean() for k in available_layers}
            # then calcualte std over these means to get std of accuracy; then normalize to get 2SEM
            y_error_values = (
                2
                * np.array([np.std(y_error_mean_data[k]) for k in available_layers])
                / np.sqrt(len(y_error_data))
            )
        else:
            y_error_values = None
        plt.bar(
            x_values,
            y_values,
            yerr=y_error_values,
            color=utf.colors[reference_type_i],
            width=bar_width,
            label=reference_type_i,
        )

    if legend:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    plt.xticks(
        np.arange(0, len(available_layers)) * ((len(instr_type_list) + 1) * bar_width)
        + 1,
        [int(k) + 1 for k in available_layers_labels],
    )

    plt.ylabel("Accuracy")
    plt.xlabel("Layer")

    # no axis on top and right
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # plt.title(title_prefix + "Accuracy per Layer")
    plot_name = "layerwise_accuracy"
    for version in range(100):
        file_name = os.path.join(
            results_folder, f"{plot_name}_data_{version}{file_type}"
        )
        # if file_name does not yet exist, use it
        if not os.path.exists(file_name):
            break
    if save_fig:
        print(f"figure saved under {file_name}")
        plt.savefig(file_name, bbox_inches="tight")

    plt.show()


def plot_exclusion_criteria(df_checks, proportion, results_folder, save_fig):
    plt.figure(figsize=(4, 3))
    rects = plt.bar(
        np.arange(2),
        [
            (len(df_checks) - df_checks["passed_checks"].sum()),
            df_checks["passed_checks"].sum(),
        ],
    )
    utf_helper.autolabel_counts(rects, plt.gca())
    plt.xticks(np.arange(2), ["Failed", "Passed"], fontsize=12)
    plt.ylim(0, 345)
    plt.tight_layout()
    plt.ylabel("Number of Responses", fontsize=12)

    # no axis on top and right
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plot_name = f"exclusion_criteria_all_{'proportional' if proportion else 'absolute'}"
    for version in range(100):
        file_name = os.path.join(results_folder, f"{plot_name}_{version}{file_type}")
        # if file_name does not yet exist, use it
        if not os.path.exists(file_name):
            break
    if save_fig:
        print(f"figure saved under {file_name}")
        plt.savefig(file_name, bbox_inches="tight")

    plt.show()

    for criterion in [c for c in df_checks.columns if c.endswith("_result")]:
        plt.figure(figsize=(4, 3))
        if proportion:
            rects = plt.bar(
                np.arange(2),
                [
                    (len(df_checks) - df_checks[criterion].sum()) / len(df_checks),
                    df_checks[criterion].sum() / len(df_checks),
                ],
            )
        else:
            rects = plt.bar(
                np.arange(2),
                [
                    (len(df_checks) - df_checks[criterion].sum()),
                    df_checks[criterion].sum(),
                ],
            )
        utf_helper.autolabel_counts(rects, plt.gca())
        plt.xticks(np.arange(2), ["Failed", "Passed"], fontsize=12)
        plt.ylim(0, 345)
        plt.tight_layout()
        title = f"{criterion.replace('_result', '').replace('_', ' ').title()}"
        plt.ylabel("Number of Responses", fontsize=12)

        # no axis on top and right
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)

        plot_name = f"exclusion_criterion_{criterion.replace('_result', '')}_{'proportional' if proportion else 'absolute'}"

        for version in range(100):
            file_name = os.path.join(
                results_folder, f"{plot_name}_{version}{file_type}"
            )
            # if file_name does not yet exist, use it
            if not os.path.exists(file_name):
                break
        if save_fig:
            print(f"figure saved under {file_name}")
            plt.savefig(file_name, bbox_inches="tight")

        plt.show()


def plot_task_postings(df, proportion, results_folder, save_fig):
    plt.figure(figsize=(4, 3))
    n_bins = df["task_id"].value_counts()[0]
    rects = plt.bar(np.arange(n_bins), df["task_id"].value_counts().value_counts())
    utf_helper.autolabel_counts(rects, plt.gca())
    plt.xticks(np.arange(n_bins), np.arange(1, n_bins + 1), fontsize=12)
    plt.tight_layout()
    plt.ylabel("Number of Tasks", fontsize=12)
    plt.xlabel("Number of Postings", fontsize=12)

    # no axis on top and right
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plot_name = (
        f"histogram_over_postings_{'proportional' if proportion else 'absolute'}"
    )
    for version in range(100):
        file_name = os.path.join(results_folder, f"{plot_name}_{version}.pdf")
        # if file_name does not yet exist, use it
        if not os.path.exists(file_name):
            break
    if save_fig:
        print(f"figure saved under {file_name}")
        plt.savefig(file_name, bbox_inches="tight")

    plt.show()


def plot_practice_trials_attempts(df, proportion, results_folder, save_fig):
    hist = df["demo_trials_repetitions"].value_counts()
    normalizer = np.sum(hist) if proportion else 1
    plt.bar(np.array(hist.index), np.array(hist) / normalizer)
    plt.xticks(
        np.arange(1, max(hist.index) + 1, dtype=int),
        np.arange(1, max(hist.index) + 1, dtype=int),
    )
    plt.tight_layout()
    plt.title("Histogram over Practice Block Attempts")
    plt.ylabel(
        "Number of Participants w/ x Attempts to \n Pass the Practice trials",
        fontsize=12,
    )
    plt.xlabel("Number of Practice Block Attempts", fontsize=12)

    # no axis on top and right
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    y_max = np.max(np.array(hist) / normalizer)
    plt.ylim(0, np.ceil(y_max / 10) * 10)
    plt.vlines(5.5, 0, 75, linestyles="dashed", color="k")

    plt.tick_params(axis="both", which="major", labelsize=12)

    plot_name = (
        f"histogram_over_demo_trials_{'proportional' if proportion else 'absolute'}"
    )
    for version in range(100):
        file_name = os.path.join(results_folder, f"{plot_name}_{version}.pdf")
        # if file_name does not yet exist, use it
        if not os.path.exists(file_name):
            break
    if save_fig:
        print(f"figure saved under {file_name}")
        plt.savefig(file_name, bbox_inches="tight")

    plt.show()


def plot_accuracy_per_batch(
    df,
    results_folder,
    save_fig,
    instr_type_list=["optimized", "natural", "mixed"],
    labels=["Synthetic", "Natural", "Mixed"],
    fig_name_suffix="",
):
    def generate_dfs_grouped_by_batch(df, batch_ids):
        df = df.copy()

        for bid in batch_ids:
            selected_df = df[df["batch"] == bid]
            yield selected_df

    fig, axes = plt.subplots(3, 4)
    fig.set_size_inches(8, 8)
    batch_ids = sorted(df["batch"].unique().tolist())
    for batch_idx, (batch_df, ax) in enumerate(
        zip(generate_dfs_grouped_by_batch(df, batch_ids), axes.flatten())
    ):
        utf.make_plot_synthetic_imgs_are_helpful(
            batch_df,
            instr_type_list=instr_type_list,
            labels=labels,
            save_fig=False,
            show_fig=False,
            ax=ax,
            ylim_min=-0.3,
            ylim_max=1.3,
        )
        ax.set_title(f"Batch {batch_ids[batch_idx]}")
    plt.tight_layout()

    # no axis on top and right
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plot_name = f"accuracy_per_batch"
    for version in range(100):
        file_name = os.path.join(
            results_folder, f"{plot_name}{fig_name_suffix}_{version}.pdf"
        )
        # if file_name does not yet exist, use it
        if not os.path.exists(file_name):
            break
    if save_fig:
        print(f"figure saved under {file_name}")
        plt.savefig(file_name, bbox_inches="tight")

    plt.show()


def rand_jitter(arr, rng):
    stdev = 0.01 * (max(arr) - min(arr))
    return arr + rng.standard_normal(len(arr)) * stdev


def plot_accuracy_vs_relative_activation_difference(
    df, results_folder, save_fig, fig_name_suffix="", seed=1,
):
    rng = np.random.default_rng(seed=seed)
    columns = ["batch", "layer", "kernel_size", "mode"]
    averaged_df = df.groupby(columns, as_index=False, axis=0).mean()[
        columns
        + [
            "base_query_activation",
            "min_query_activation",
            "max_query_activation",
            "correct",
        ]
    ]
    averaged_df["relative_activation_difference"] = (
        averaged_df["max_query_activation"] - averaged_df["min_query_activation"]
    ) / averaged_df["base_query_activation"]
    averaged_df = averaged_df.rename({"correct": "accuracy"}, axis=1)
    colors = averaged_df["mode"].map(lambda x: utf.colors[x])
    plt.figure(figsize=(2.86535, 2.86535 / 1.2))
    plt.scatter(
        averaged_df["relative_activation_difference"],
        rand_jitter(averaged_df["accuracy"], rng),
        marker=".",
        s=2,
        color=colors,
    )
    plt.xlabel("Relative activation difference")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.xlim([0.1, 1.4])
    plt.xticks((0.2, 0.6, 1.0, 1.4))

    # no axis on top and right
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    from scipy.stats import spearmanr

    print(
        spearmanr(
            averaged_df["relative_activation_difference"], averaged_df["accuracy"]
        )
    )

    plot_name = f"accuracy_vs_relative_activation_difference"
    for version in range(100):
        file_name = os.path.join(
            results_folder, f"{plot_name}{fig_name_suffix}_{version}" + file_type
        )
        # if file_name does not yet exist, use it
        if not os.path.exists(file_name):
            break
    if save_fig:
        print(f"figure saved under {file_name}")
        plt.savefig(file_name, bbox_inches="tight")

    plt.show()


def plot_binned_accuracy_vs_relative_activation_difference(
    df, results_folder, save_fig, fig_name_suffix="", n_bins=5
):
    columns = ["batch", "layer", "kernel_size", "mode"]
    averaged_df = df.groupby(columns, as_index=False, axis=0).mean()[
        columns
        + [
            "base_query_activation",
            "min_query_activation",
            "max_query_activation",
            "correct",
        ]
    ]
    averaged_df["relative_activation_difference"] = (
        averaged_df["max_query_activation"] - averaged_df["min_query_activation"]
    ) / averaged_df["base_query_activation"]
    averaged_df = averaged_df.rename({"correct": "accuracy"}, axis=1)

    plt.figure(figsize=(2.86535, 2.86535 / 1.2))
    for mode in averaged_df["mode"].unique():
        # create 5 bins such that each contain the same number of elements
        bins = scipy.stats.mstats.mquantiles(
            averaged_df[averaged_df["mode"] == mode]["relative_activation_difference"],
            [0, 1.0 / 5, 2.0 / 5, 3.0 / 5, 4.0 / 5, 1],
        )

        means = scipy.stats.binned_statistic(
            averaged_df[averaged_df["mode"] == mode]["relative_activation_difference"],
            averaged_df[averaged_df["mode"] == mode]["accuracy"],
            statistic="mean",
            bins=bins,
        )[0]
        stds = scipy.stats.binned_statistic(
            averaged_df[averaged_df["mode"] == mode]["relative_activation_difference"],
            averaged_df[averaged_df["mode"] == mode]["accuracy"],
            statistic="std",
            bins=bins,
        )[0]

        color = utf.colors[mode]

        plt.errorbar(
            (bins[1:] + bins[:-1]) / 2.0,
            means,
            marker=".",
            yerr=stds,
            ls="--",
            linewidth=1,
            capsize=3,
            elinewidth=2,
            color=color,
        )
        plt.scatter((bins[1:] + bins[:-1]) / 2.0, means, marker=".", s=2, color=color)
    plt.xlabel("Relative activation difference")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.xlim([0.1, 1.4])
    plt.xticks((0.2, 0.6, 1.0, 1.4))

    # no axis on top and right
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    from scipy.stats import spearmanr

    print(
        spearmanr(
            averaged_df["relative_activation_difference"], averaged_df["accuracy"]
        )
    )

    plot_name = f"accuracy_vs_relative_activation_difference_binned"
    for version in range(100):
        file_name = os.path.join(
            results_folder, f"{plot_name}{fig_name_suffix}_{version}" + file_type
        )
        # if file_name does not yet exist, use it
        if not os.path.exists(file_name):
            break
    if save_fig:
        print(f"figure saved under {file_name}")
        plt.savefig(file_name, bbox_inches="tight")

    plt.show()


def plot_accuracy_vs_relative_activation_difference_with_different_markers(
    df, results_folder, focus_mode, save_fig, seed=1
):
    rng = np.random.default_rng(seed=seed)
    columns = ["batch", "layer", "mode", focus_mode]
    averaged_df = df.groupby(columns, as_index=False, axis=0).mean()[
        columns
        + [
            "base_query_activation",
            "min_query_activation",
            "max_query_activation",
            "correct",
        ]
    ]
    averaged_df["relative_activation_difference"] = (
        averaged_df["max_query_activation"] - averaged_df["min_query_activation"]
    ) / averaged_df["base_query_activation"]
    averaged_df = averaged_df.rename({"correct": "accuracy"}, axis=1)
    colors = averaged_df["mode"].map(lambda x: utf.colors[x])
    marker_list = [".", "x", "o", "*", "+", "<", ">", "2", "h"]
    for i, focus_mode_i in enumerate(averaged_df[focus_mode].unique()):
        print(focus_mode, focus_mode_i)
        averaged_df_i = averaged_df[averaged_df[focus_mode] == focus_mode_i]
        colors = averaged_df_i["mode"].map(lambda x: utf.colors[x])
        plt.scatter(
            rand_jitter(averaged_df_i["relative_activation_difference"], rng),
            rand_jitter(averaged_df_i["accuracy"], rng),
            marker=marker_list[i],
            color=colors,
            label=focus_mode_i,
        )

        from scipy.stats import spearmanr

        print(
            spearmanr(
                averaged_df_i["relative_activation_difference"],
                averaged_df_i["accuracy"],
            )
        )

    plt.legend()
    plt.xlabel("Relative activation difference")
    plt.ylabel("Accuracy")
    plt.tight_layout()

    # no axis on top and right
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plot_name = f"accuracy_vs_relative_activation_difference_different_markers"
    for version in range(100):
        file_name = os.path.join(results_folder, f"{plot_name}_{version}.pdf")
        # if file_name does not yet exist, use it
        if not os.path.exists(file_name):
            break
    if save_fig:
        print(f"figure saved under {file_name}")
        plt.savefig(file_name, bbox_inches="tight")

    plt.show()


def plot_performance_by_worker(df, results_folder, save_fig, seed=1):
    averaged_df = df.groupby(["task_number", "mode"], as_index=False, axis=0).mean()
    averaged_df = averaged_df[["task_number", "mode", "correct"]]
    averaged_df = averaged_df.rename({"correct": "accuracy"}, axis=1)
    colors = averaged_df["mode"].map(lambda x: utf.colors[x])

    rng = np.random.default_rng(seed=seed)

    fig, ax = plt.subplots()
    plt.scatter(
        rand_jitter(averaged_df["task_number"], rng),
        rand_jitter(averaged_df["accuracy"], rng),
        marker=".",
        color=colors,
    )
    plt.xlabel("Worker ID")
    plt.ylabel(
        f"Accuracy (over {len(df['layer'].unique()) * len(df['kernel_size'].unique())} trials)"
    )

    # no axis on top and right
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plot_name = f"worker_performance"
    for version in range(100):
        file_name = os.path.join(results_folder, f"{plot_name}_{version}.pdf")
        # if file_name does not yet exist, use it
        if not os.path.exists(file_name):
            break
    if save_fig:
        print(f"figure saved under {file_name}")
        plt.savefig(file_name, bbox_inches="tight")

    plt.show()


def plot_worker_baseline_consistency_matrix(
    cohens_kappa_matrix,
    cohens_kappa_std_matrix,
    extended_mode_list,
    extended_mode_label_list,
    results_folder,
    save_fig,
    fig_name_suffix="",
    show_text=True,
    cmap="viridis",
    vmin=None,
    vmax=None,
):

    assert len(extended_mode_label_list) == len(extended_mode_list)

    fig, ax = plt.subplots(
        figsize=(7 * len(extended_mode_list) / 9, 7 * len(extended_mode_list) / 9)
    )
    im = plt.imshow(cohens_kappa_matrix, cmap=cmap, vmin=vmin, vmax=vmax)

    if show_text:
        cmap_ = plt.get_cmap(cmap)

        for i in range(len(extended_mode_list)):
            for j in range(len(extended_mode_list)):
                bg_color_rgba = cmap_(cohens_kappa_matrix[i, j])
                bg_color_gray = (
                    1.0 / 3 * bg_color_rgba[0]
                    + 1.0 / 3 * bg_color_rgba[1]
                    + 1.0 / 3 * bg_color_rgba[2]
                )

                if bg_color_gray > 0.35:
                    text_color = "k"
                else:
                    text_color = "w"
                if np.issubdtype(cohens_kappa_matrix.dtype, np.integer):
                    # with values between 0 and 100
                    text = ax.text(
                        j,
                        i,
                        f"{cohens_kappa_matrix[i, j]}\u00B1{cohens_kappa_std_matrix[i, j]}",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=10,
                        rotation=0,
                    )
                else:
                    # with values between 0 and 1
                    text = ax.text(
                        j,
                        i,
                        f"{round(cohens_kappa_matrix[i, j],2)}\u00B1{round(cohens_kappa_std_matrix[i, j],2)}",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=10,
                        rotation=45,
                    )

    plt.xticks(
        np.arange(len(extended_mode_label_list)), extended_mode_label_list, rotation=45
    )
    plt.yticks(np.arange(len(extended_mode_label_list)), extended_mode_label_list)

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax)

    plt.tight_layout()

    if save_fig:
        plot_name = "consistency_matrix"

        for version in range(100):
            file_name = os.path.join(
                results_folder, f"{plot_name}{fig_name_suffix}_{version}{file_type}"
            )
            # if file_name does not yet exist, use it
            if not os.path.exists(file_name):
                break
        print(f"figure saved under {file_name}")
        plt.savefig(file_name, bbox_inches="tight")


def sub_plot_cohens_kappa_by_batch(
    kappa_cond_i_cond_ii_dict,
    conditions_list,
    extended_mode_label_dict,
    figures_folder,
    exp_str,
    save_fig=False,
):
    for fig_name_suffix in ["_mean_and_std", "_mean_and_sem", "_scatter"]:

        scatter = False
        mean_and_std = False
        mean_and_sem = False
        if fig_name_suffix == "_mean_and_std":
            mean_and_std = True
        elif fig_name_suffix == "_mean_and_sem":
            mean_and_sem = True
        elif fig_name_suffix == "_scatter":
            scatter = True

        fig, ax = plt.subplots(
            len(conditions_list), len(conditions_list), figsize=(15, 15)
        )
        for idx_i, cond_i in enumerate(conditions_list):
            print(f"{idx_i} / {len(conditions_list)}", end="\r")
            x_label = True if idx_i == len(conditions_list) - 1 else False
            title = True if idx_i == 0 else False

            for idx_ii, cond_ii in enumerate(conditions_list):
                y_label = True if idx_ii == 0 else False
                sub_plot_cohens_kappa_two_conditions(
                    kappa_cond_i_cond_ii_dict[cond_i][cond_ii],
                    cond_i,
                    cond_ii,
                    ax[idx_i, idx_ii],
                    extended_mode_label_dict,
                    scatter=scatter,
                    mean_and_std=mean_and_std,
                    mean_and_sem=mean_and_sem,
                    y_label=y_label,
                    x_label=x_label,
                    title=title,
                )

        if save_fig:
            plot_name = "cohens_kappa"

            for version in range(100):
                file_name = os.path.join(
                    figures_folder,
                    f"{plot_name}{fig_name_suffix}_{exp_str}_{version}{file_type}",
                )
                # if file_name does not yet exist, use it
                if not os.path.exists(file_name):
                    break
            print(f"figure saved under {file_name}")
            plt.savefig(file_name, bbox_inches="tight")


def sub_plot_cohens_kappa_two_conditions(
    kappa_dict,
    cond_i,
    cond_ii,
    ax,
    extended_mode_label_dict,
    scatter=False,
    mean_and_std=False,
    mean_and_sem=False,
    y_label=False,
    x_label=False,
    title=False,
    seed=1,
):
    rng = np.random.default_rng(seed=seed)
    all_batches_list = []
    for batch_i in range(10):
        flat_list = [
            item_i
            for subject_i_dict in kappa_dict[f"batch_{batch_i}"].values()
            for item_i in list(subject_i_dict.values())
        ]
        all_batches_list.append(flat_list)
        if scatter:
            ax.scatter(
                rand_jitter([batch_i] * len(flat_list), rng), flat_list, marker="."
            )
        if mean_and_std:
            flat_array = np.array(flat_list)
            ax.errorbar(batch_i, flat_array.mean(), yerr=flat_array.std(), marker="x")
        if mean_and_sem:
            flat_array = np.array(flat_list)
            ax.errorbar(
                batch_i,
                flat_array.mean(),
                yerr=2 * (flat_array.std() / np.sqrt(flat_array.shape[0])),
                marker="x",
            )
    # no axis on top and right
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim([-1.1, 1.1])
    if x_label:
        ax.set_xlabel("Image Set", fontsize=12)
        ax.set_xticks([0, 3, 6, 9])
        ax.set_xticklabels([1, 4, 7, 10], fontsize=10)
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])
    if y_label:
        ax.set_ylabel(f"{extended_mode_label_dict[cond_i]}", fontsize=12)
    else:
        ax.set_yticks([])
        ax.set_yticklabels([])
    if title:
        ax.set_title(f"{extended_mode_label_dict[cond_ii]}", fontsize=12)
