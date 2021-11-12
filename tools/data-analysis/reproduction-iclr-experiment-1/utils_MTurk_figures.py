import pandas as pd
import numpy as np
import os
import warnings
import utils_ICLR_figures_helper as ut_helper

import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm
import matplotlib

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
colors = [
    [71 / 255, 120 / 255, 158 / 255],  # syn
    [255 / 255, 172 / 255, 116 / 255],  # nat
    [172 / 255, 167 / 255, 166 / 255],  # none
]


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
            "row_variability_details_details_upper_extracted"
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
    # plt.title(f"Histogram over Total Response Time\n{criteria_str} Data\nExclusion Criterion 90 s < x < 600 s")

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


def plot_catch_trials_details_ratio_extracted(
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
            "catch_trials_details_ratio_extracted"
        ]
    )
    plt.tight_layout()
    # plt.title(f"Histogram over Correctly Answered Catch Trials\n{criteria_str} Data\nExclusion Criterion x > 0.6 (out of three)")

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


def plot_exclusion_criteria(df_checks, proportion, results_folder, save_fig):
    plt.figure(figsize=(4, 3))
    rects = plt.bar(
        np.arange(2),
        [
            (len(df_checks) - df_checks["passed_checks"].sum()),
            df_checks["passed_checks"].sum(),
        ],
    )
    ut_helper.autolabel_counts(rects, plt.gca())
    plt.xticks(np.arange(2), ["Failed", "Passed"], fontsize=12)
    plt.ylim(0, 345)
    plt.tight_layout()
    # plt.title("All Exclusion Criteria")
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
        ut_helper.autolabel_counts(rects, plt.gca())
        plt.xticks(np.arange(2), ["Failed", "Passed"], fontsize=12)
        plt.ylim(0, 345)
        plt.tight_layout()
        title = f"{criterion.replace('_result', '').replace('_', ' ').title()}"
        # plt.title(title)
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
    ut_helper.autolabel_counts(rects, plt.gca())
    plt.xticks(np.arange(n_bins), np.arange(1, n_bins + 1), fontsize=12)
    plt.tight_layout()
    # plt.title("Histogram over Postings")
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
