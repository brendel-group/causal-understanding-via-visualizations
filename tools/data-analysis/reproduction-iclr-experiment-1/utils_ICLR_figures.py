import pandas as pd
import numpy as np
import os
import warnings
import utils_ICLR_figures_helper as ut_helper

from matplotlib import rc
from matplotlib import cm
import matplotlib

from matplotlib import rcParams

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["DejaVu Sans"]

# output text as text and not paths
rcParams["svg.fonttype"] = "none"
rcParams["pdf.fonttype"] = "truetype"

import matplotlib.pyplot as plt

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


def make_plot_synthetic_imgs_are_helpful(
    data_df_main_exp_main,
    results_folder="figures_ICLR_2020",
    exp_str="main_exp",
    fig_1=False,
    twitter=False,
    save_fig=True,
):

    if exp_str == "main_exp":
        fig_size_width_factor = 1
        instr_type_str = "instr_type_joint_none"
        instr_type_joint_none_list_in_nice_order = ["optimized", "natural", "none"]
        if fig_1 or twitter:
            instr_type_joint_none_list_in_nice_order = ["optimized", "natural"]
            fig_size_width_factor = 0.7
    else:
        fig_size_width_factor = 0.7
        instr_type_str = "instr_type"
        instr_type_joint_none_list_in_nice_order = ["optimized", "natural"]

    dict_acc = {}
    dict_sem = {}
    for reference_type_i in instr_type_joint_none_list_in_nice_order:
        df_factor_i = data_df_main_exp_main[
            data_df_main_exp_main[instr_type_str] == reference_type_i
        ].copy()
        accuracy = (df_factor_i["correct"] == True).sum() / df_factor_i.shape[0]
        dict_acc[reference_type_i] = accuracy
        # evaluate standard error of the mean
        dict_acc_subject_i = {}
        for subject_i in df_factor_i["subject_id"].unique():
            df_factor_i_subject = df_factor_i[
                df_factor_i["subject_id"] == subject_i
            ].copy()
            accuracy_subject_i = (
                df_factor_i_subject["correct"] == True
            ).sum() / df_factor_i_subject.shape[0]
            dict_acc_subject_i[subject_i] = accuracy_subject_i
        n_data_points = len(dict_acc_subject_i)
        standard_deviation = np.std(
            np.fromiter(dict_acc_subject_i.values(), dtype=float)
        )
        sem = standard_deviation / np.sqrt(n_data_points)
        dict_sem[reference_type_i] = sem

    ### figure
    fig, ax = plt.subplots(1, 1, figsize=(2.2 * fig_size_width_factor, 3))
    if twitter:
        fig, ax = plt.subplots(1, 1, figsize=(3, 1.675))

    # 2 SEM
    dict_2sem = {x: y * 2 for (x, y) in dict_sem.items()}
    rect = ax.bar(
        range(len(dict_acc)),
        list(dict_acc.values()),
        color=colors,
        yerr=dict_2sem.values(),
        error_kw=dict(lw=error_bar_linewidth),
        align="center",
    )
    ut_helper.autolabel(rect, ax, dict_2sem)
    ax.axhline(0.5, color="k", linestyle="--", linewidth=1, label="Chance")

    # significance
    if fig_1 == False and twitter == False:
        height_bar_and_error_list = []
        for rect_i, error_value_i in zip(rect, dict_2sem.values()):
            height = rect_i.get_height()
            height_bar_and_error_list.append(height + error_value_i)
        if exp_str == "main_exp":
            ut_helper.barplot_annotate_brackets(
                ax,
                0,
                1,
                "p $=$ 0.015",
                np.arange(3),
                height_bar_and_error_list,
                dh=dh_01,
            )
            ut_helper.barplot_annotate_brackets(
                ax,
                1,
                2,
                "p $<$ 0.001",
                np.arange(3),
                height_bar_and_error_list,
                dh=dh_12,
            )
            ut_helper.barplot_annotate_brackets(
                ax,
                0,
                2,
                "p $<$ 0.001",
                np.arange(3),
                height_bar_and_error_list,
                dh=dh_02,
            )
        elif exp_str == "ablation_exp":
            ut_helper.barplot_annotate_brackets(
                ax, 0, 1, "p $<$ 0.001", np.arange(3), height_bar_and_error_list
            )
        else:  # exp_str == "rebuttal_exp" or "pilot_ks_0_exp" or "full_exp_ks_1"
            pass  # TODO

    ax.set_ylabel("Proportion Correct", fontsize=fontsize_axes_labels)
    if fig_1:
        ax.set_ylim(0.5, 1)
    elif twitter:
        ax.set_ylim(0.5, 1)
    #         yticks_and_labels = [0.5, 0.75, 1]
    #         ax.set_yticks(yticks_and_labels)
    #         ax.set_yticklabels(yticks_and_labels)
    else:
        yticks_and_labels = [0, 0.2, 0.4, 0.6, 0.8, 1]
        ax.set_yticks(yticks_and_labels)
        ax.set_yticklabels(yticks_and_labels)

    # xticks
    ax.set_xticks(range(len(dict_acc)))
    ax.set_xticklabels(["Synthetic", "Natural", "None"][: len(dict_acc)])

    ax.tick_params(labelsize=fontsize_tick_labels)

    # no axis on top and right
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if fig_1 == False and twitter == False:
        ax.set_xlabel("Reference Images", fontsize=fontsize_axes_labels)

    plot_name = "synthetic_imgs_are_helpful"
    if fig_1:
        plot_name += "_fig_1"
    if twitter:
        plot_name += "twitter"
    for version in range(100):
        file_name = os.path.join(
            results_folder, f"{plot_name}_{exp_str}_{version}{file_type}"
        )
        # if file_name does not yet exist, use it
        if not os.path.exists(file_name):
            break
    if save_fig:
        print(f"figure saved under {file_name}")
        plt.savefig(file_name, bbox_inches="tight")

    plt.show()


def make_plot_natural_are_better_wrt_confidence(
    data_df_main_exp_main,
    results_folder="figures_ICLR_2020",
    exp_str="main_exp",
    conditioned_on_correctness=False,
    conditioned_on_falseness=False,
    conditioned_str="_",
    save_fig=True,
):

    y_axis_label = "Proportion of Total Trials"

    if conditioned_on_correctness == True and conditioned_on_falseness == True:
        warnings.warn("Check the conditioning settings!!!")
    elif conditioned_on_correctness:
        data_df_main_exp_main = data_df_main_exp_main[
            data_df_main_exp_main["correct"] == True
        ].copy()
        conditioned_str = "_conditioned_on_correctness_"
        y_axis_label = "Proportion of Correct Trials"
    elif conditioned_on_falseness:
        data_df_main_exp_main = data_df_main_exp_main[
            data_df_main_exp_main["correct"] == False
        ].copy()
        conditioned_str = "_conditioned_on_falseness_"
        y_axis_label = "Proportion of Incorrect Trials"

    if exp_str == "main_exp":
        fig_size_width_factor = 1
        instr_type_str = "instr_type_joint_none"
        instr_type_list = ["optimized", "natural", "none"]
    else:  # exp_str == "ablation_exp" or exp_str == "rebuttal_exp", ...
        fig_size_width_factor = 0.7
        instr_type_str = "instr_type"
        instr_type_list = ["optimized", "natural"]

    labels = ["Synthetic", "Natural", "None"]
    data = {}
    dict_sem = {}
    for reference_type_i in instr_type_list:
        denominator = data_df_main_exp_main[
            data_df_main_exp_main[instr_type_str] == reference_type_i
        ].shape[0]
        print(denominator)
        reference_type_i_data = {}
        reference_type_i_dict_sem = {}
        for abs_conf in data_df_main_exp_main["abs_conf_rating"].unique():
            reference_type_i_data[abs_conf] = (
                sum(
                    (data_df_main_exp_main["abs_conf_rating"] == abs_conf)
                    & (data_df_main_exp_main[instr_type_str] == reference_type_i)
                )
                / denominator
            )

            # evaluate standard error of the mean
            dict_conf_rating_subject_i = {}
            for subject_i in data_df_main_exp_main["subject_id"].unique():
                df_factor_i_subject = data_df_main_exp_main[
                    data_df_main_exp_main["subject_id"] == subject_i
                ].copy()
                denominator_subject_i = df_factor_i_subject[
                    df_factor_i_subject[instr_type_str] == reference_type_i
                ].shape[0]
                numerator_subject_i = sum(
                    (df_factor_i_subject["abs_conf_rating"] == abs_conf)
                    & (df_factor_i_subject[instr_type_str] == reference_type_i)
                )
                confidence_rating_subject_i = numerator_subject_i / (
                    denominator_subject_i + 0.0000000001
                )
                dict_conf_rating_subject_i[subject_i] = confidence_rating_subject_i
            n_data_points = len(dict_conf_rating_subject_i)
            standard_deviation = np.std(
                np.fromiter(dict_conf_rating_subject_i.values(), dtype=float)
            )
            sem = standard_deviation / np.sqrt(n_data_points)
            reference_type_i_dict_sem[abs_conf] = sem
        data[reference_type_i] = reference_type_i_data
        dict_sem[reference_type_i] = reference_type_i_dict_sem

    ### figure
    fig, axes = plt.subplots(
        1, len(instr_type_list), figsize=(3.5 * fig_size_width_factor, 3), sharey=sharey
    )

    for i, (factor_i, factor_i_data) in enumerate(data.items()):
        # 2 SEM
        dict_2sem = {x: y * 2 for (x, y) in dict_sem[factor_i].items()}
        rects = axes[i].bar(
            list(factor_i_data.keys()),
            list(factor_i_data.values()),
            color=colors[i],
            yerr=dict_2sem.values(),
            error_kw=dict(lw=error_bar_linewidth),
        )
        axes[i].set_xlabel(labels[i], fontsize=fontsize_tick_labels)
        # no axis on top and right
        axes[i].spines["top"].set_visible(False)
        axes[i].spines["right"].set_visible(False)
        axes[i].tick_params(labelsize=fontsize_tick_labels)
        if exp_str == "main_exp":
            axes[i].set_ylim(0, 0.79)
        elif exp_str == "full_exp_ks_1":
            axes[i].set_ylim(0, 0.62)
        if i == 0:
            axes[i].set_ylabel(y_axis_label, fontsize=fontsize_axes_labels)
        else:
            axes[i].yaxis.set_visible(False)
            axes[i].spines["left"].set_visible(False)
    #         ut_helper.autolabel_counts(rects, axes[i])

    if exp_str == "main_exp":
        plt.suptitle(
            "Confidence Rating", x=0.57, y=0.012, fontsize=fontsize_axes_labels
        )
    else:  # if exp_str == "ablation_exp" or exp_str == "rebuttal_exp":
        plt.suptitle(
            "Confidence Rating", x=0.61, y=0.012, fontsize=fontsize_axes_labels
        )
    plt.tight_layout()

    plot_name = "natural_are_better_wrt_confidence"
    for version in range(100):
        file_name = os.path.join(
            results_folder,
            f"{plot_name}{conditioned_str}{exp_str}_{version}{file_type}",
        )
        # if file_name does not yet exist, use it
        if not os.path.exists(file_name):
            break
    if save_fig:
        print(f"figure saved under {file_name}")
        plt.savefig(file_name, bbox_inches="tight")

    plt.show()


def make_plot_natural_are_better_wrt_reaction_time(
    data_df_main_exp_main,
    results_folder="figures_ICLR_2020",
    exp_str="main_exp",
    conditioned_on_correctness=False,
    conditioned_on_falseness=False,
    conditioned_str="_",
    save_fig=True,
):

    y_axis_label = "Reaction Time [msec] of Total Trials"

    if conditioned_on_correctness == True and conditioned_on_falseness == True:
        warnings.warn("Check the conditioning settings!!!")
    elif conditioned_on_correctness:
        data_df_main_exp_main = data_df_main_exp_main[
            data_df_main_exp_main["correct"] == True
        ].copy()
        conditioned_str = "_conditioned_on_correctness_"
        y_axis_label = "Reaction Time [msec] of Correct Trials"
    elif conditioned_on_falseness:
        data_df_main_exp_main = data_df_main_exp_main[
            data_df_main_exp_main["correct"] == False
        ].copy()
        conditioned_str = "_conditioned_on_falseness_"
        y_axis_label = "Reaction Time [msec] of Incorrect Trials"

    if exp_str == "main_exp":
        fig_size_width_factor = 1
        instr_type_str = "instr_type_joint_none"
        instr_type_list = ["optimized", "natural", "none"]
        x_tick_labels = ["Synthetic", "Natural", "None"]
    else:  # if exp_str == "ablation_exp" or exp_str == "rebuttal_exp":
        fig_size_width_factor = 0.7
        instr_type_str = "instr_type"
        instr_type_list = ["optimized", "natural"]
        x_tick_labels = ["Synthetic", "Natural"]

    dict_RT = {}
    dict_RT_sem = {}
    for reference_type_i in instr_type_list:
        df_factor_i = data_df_main_exp_main[
            data_df_main_exp_main[instr_type_str] == reference_type_i
        ].copy()
        RT_mean = df_factor_i["RT"].mean()
        dict_RT[reference_type_i] = RT_mean
        # evaluate standard error of the mean
        dict_RT_subject_i = {}
        for subject_i in df_factor_i["subject_id"].unique():
            df_factor_i_subject = df_factor_i[
                df_factor_i["subject_id"] == subject_i
            ].copy()
            RT_mean_subject_i = df_factor_i_subject["RT"].mean()
            dict_RT_subject_i[subject_i] = RT_mean_subject_i
        n_data_points = len(dict_RT_subject_i)
        standard_deviation = np.std(
            np.fromiter(dict_RT_subject_i.values(), dtype=float)
        )
        sem = standard_deviation / np.sqrt(n_data_points)
        dict_RT_sem[reference_type_i] = sem

    ### figure
    fig, ax = plt.subplots(1, 1, figsize=(2.2 * fig_size_width_factor, 3))

    # 2 SEM
    dict_2sem = {x: y * 2 for (x, y) in dict_RT_sem.items()}
    rect = ax.bar(
        range(len(dict_RT)),
        list(dict_RT.values()),
        color=colors,
        yerr=dict_2sem.values(),
        error_kw=dict(lw=error_bar_linewidth),
        align="center",
        label="All subjects",
    )
    ut_helper.autolabel(rect, ax, dict_2sem)

    # significance
    height_bar_and_error_list = []
    for rect_i, error_value_i in zip(rect, dict_2sem.values()):
        height = rect_i.get_height()
        height_bar_and_error_list.append(height + error_value_i)
    if conditioned_on_correctness == False and conditioned_on_falseness == False:
        if exp_str == "main_exp":
            ut_helper.barplot_annotate_brackets(
                ax,
                0,
                1,
                "p $=$ 0.002",
                np.arange(3),
                height_bar_and_error_list,
                dh=dh_01,
            )
            ut_helper.barplot_annotate_brackets(
                ax,
                0,
                2,
                "p $<$ 0.001",
                np.arange(3),
                height_bar_and_error_list,
                dh=dh_02,
            )
            ut_helper.barplot_annotate_brackets(
                ax,
                1,
                2,
                "p $=$ 0.002",
                np.arange(3),
                height_bar_and_error_list,
                dh=0.40,
            )
        elif exp_str == "ablation_exp":
            ut_helper.barplot_annotate_brackets(
                ax, 0, 1, "p $<$ 0.001", np.arange(3), height_bar_and_error_list
            )
        else:  # if exp_str == "rebuttal_exp":
            pass  # TODO
    elif conditioned_on_correctness == True:
        if exp_str == "main_exp":
            ut_helper.barplot_annotate_brackets(
                ax,
                0,
                1,
                "p $=$ 0.006",
                np.arange(3),
                height_bar_and_error_list,
                dh=dh_01,
            )
            ut_helper.barplot_annotate_brackets(
                ax,
                0,
                2,
                "p $<$ 0.001",
                np.arange(3),
                height_bar_and_error_list,
                dh=dh_02,
            )
            ut_helper.barplot_annotate_brackets(
                ax,
                1,
                2,
                "p $=$ 0.005",
                np.arange(3),
                height_bar_and_error_list,
                dh=0.365,
            )
        elif exp_str == "ablation_exp":
            ut_helper.barplot_annotate_brackets(
                ax, 0, 1, "p $<$ 0.001", np.arange(3), height_bar_and_error_list
            )
        else:  # if exp_str == "rebuttal_exp":
            pass  # TODO
    elif conditioned_on_falseness == True:
        if exp_str == "main_exp":
            ut_helper.barplot_annotate_brackets(
                ax,
                0,
                1,
                "p $=$ 0.035",
                np.arange(3),
                height_bar_and_error_list,
                dh=dh_01,
            )
            ut_helper.barplot_annotate_brackets(
                ax,
                0,
                2,
                "p $<$ 0.001",
                np.arange(3),
                height_bar_and_error_list,
                dh=dh_02,
            )
            ut_helper.barplot_annotate_brackets(
                ax,
                1,
                2,
                "p $<$ 0.001",
                np.arange(3),
                height_bar_and_error_list,
                dh=0.37,
            )
        elif exp_str == "ablation_exp":
            ut_helper.barplot_annotate_brackets(
                ax, 0, 1, "p $=$ 0.638", np.arange(3), height_bar_and_error_list
            )
        else:  # if exp_str == "rebuttal_exp":
            pass  # TODO

    ax.set_ylabel(y_axis_label, fontsize=fontsize_axes_labels)

    # xticks
    ax.set_xticks(range(len(dict_RT)))
    ax.set_xticklabels(x_tick_labels)
    ax.tick_params(labelsize=fontsize_tick_labels)

    # make y axes comparable across correct, incorrect and total trials
    if exp_str == "full_exp_ks_1":
        ax.set_ylim(0, 8200)

    # no axis on top and right
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlabel("Reference Images", fontsize=fontsize_axes_labels)

    plot_name = "natural_are_better_wrt_reaction_time"
    for version in range(100):
        file_name = os.path.join(
            results_folder,
            f"{plot_name}{conditioned_str}{exp_str}_{version}{file_type}",
        )
        # if file_name does not yet exist, use it
        if not os.path.exists(file_name):
            break
    if save_fig:
        print(f"figure saved under {file_name}")
        plt.savefig(file_name, bbox_inches="tight")

    plt.show()


def make_plot_no_trend_across_layers(
    data_df_main_exp_main, exp_str="main_exp", save_fig=True
):

    if exp_str == "main_exp":
        fig_size_width_factor = 1
        instr_type_str = "instr_type_joint_none"
        instr_type_list = ["optimized", "natural", "none"]
        layer_labels_list = ["3a", "3b", "4a", "4b", "4c", "4d", "4e", "5a", "5b"]
        labels = ["Synthetic", "Natural", "None"]
    elif exp_str == "ablation_exp" or exp_str == "rebuttal_exp":
        fig_size_width_factor = 0.7
        instr_type_str = "instr_type"
        instr_type_list = ["optimized", "natural"]
        layer_labels_list = ["3a", "4a", "4c", "4e", "5b"]
        labels = ["Synthetic", "Natural"]

    dict_acc = {}
    dict_sem = {}
    for reference_type_i in instr_type_list:
        df_reference_type_i = data_df_main_exp_main[
            data_df_main_exp_main[instr_type_str] == reference_type_i
        ].copy()
        dict_acc_layer_number_i = {}
        dict_sem_layer_number_i = {}
        for layer_number_i in sorted(df_reference_type_i.layer.unique()):
            df_reference_type_i_layer_number_i = df_reference_type_i[
                df_reference_type_i["layer"] == layer_number_i
            ].copy()
            accuracy = (
                df_reference_type_i_layer_number_i["correct"] == True
            ).sum() / df_reference_type_i_layer_number_i.shape[0]
            dict_acc_layer_number_i[layer_number_i] = accuracy
            # evaluate standard error of the mean
            dict_acc__subject_i = {}
            for subject_i in df_reference_type_i_layer_number_i["subject_id"].unique():
                df_reference_type_i_layer_number_i_subject_i = df_reference_type_i_layer_number_i[
                    df_reference_type_i_layer_number_i["subject_id"] == subject_i
                ].copy()
                accuracy_subject_i = (
                    df_reference_type_i_layer_number_i_subject_i["correct"] == True
                ).sum() / df_reference_type_i_layer_number_i_subject_i.shape[0]
                dict_acc__subject_i[subject_i] = accuracy_subject_i
            n_data_points = len(dict_acc__subject_i)
            standard_deviation = np.std(
                np.fromiter(dict_acc__subject_i.values(), dtype=float)
            )
            sem = standard_deviation / np.sqrt(n_data_points)
            dict_sem_layer_number_i[layer_number_i] = sem
        dict_acc[reference_type_i] = dict_acc_layer_number_i
        dict_sem[reference_type_i] = dict_sem_layer_number_i

    fig, axes = plt.subplots(
        1, len(instr_type_list), figsize=(8 * fig_size_width_factor, 2.5), sharey=sharey
    )

    for i, (factor_i, factor_i_data) in enumerate(dict_acc.items()):
        dict_2sem = {x: y * 2 for (x, y) in dict_sem[factor_i].items()}
        rect = axes[i].bar(
            np.arange(len(factor_i_data)),
            list(factor_i_data.values()),
            yerr=dict_2sem.values(),
            error_kw=dict(lw=error_bar_linewidth),
            color=colors[i],
        )
        #         ut_helper.autolabel(rect, axes, dict_2sem)
        axes[i].set_xticks(np.arange(len(factor_i_data)))
        if exp_str == "rebuttal_exp":
            axes[i].set_xticklabels(["1", "2", "3", "4", "5", "6"])
        else:
            axes[i].set_xticklabels(layer_labels_list)
        axes[i].set_xlabel(labels[i], fontsize=fontsize_tick_labels)
        # no axis on top and right
        axes[i].spines["top"].set_visible(False)
        axes[i].spines["right"].set_visible(False)
        axes[i].tick_params(labelsize=fontsize_tick_labels)
        axes[i].set_ylim(0, 1)

        axes[i].axhline(0.5, color="k", linestyle="--", linewidth=1, label="Chance")
        if i == 0:
            axes[i].set_ylabel("Proportion Correct", fontsize=fontsize_axes_labels)
        else:
            axes[i].yaxis.set_visible(False)
            axes[i].spines["left"].set_visible(False)

    if exp_str == "main_exp":
        plt.suptitle("Layer", x=0.526, y=0.05, fontsize=fontsize_axes_labels)
    elif exp_str == "ablation_exp" or exp_str == "rebuttal_exp":
        plt.suptitle("Layer", x=0.545, y=0.05, fontsize=fontsize_axes_labels)

    yticks_and_labels = [0, 0.2, 0.4, 0.6, 0.8, 1]
    axes[0].set_yticks(yticks_and_labels)
    axes[0].set_yticklabels(yticks_and_labels)
    plt.tight_layout()

    plot_name = "no_trend_across_layers"
    for version in range(100):
        file_name = os.path.join(
            "figures_ICLR_2020", f"{plot_name}_{exp_str}_{version}{file_type}"
        )
        # if file_name does not yet exist, use it
        if not os.path.exists(file_name):
            break
    if save_fig:
        print(f"figure saved under {file_name}")
        plt.savefig(file_name, bbox_inches="tight")

    plt.show()


def make_plot_no_trend_across_layers_alternating(
    data_df_main_exp_main,
    exp_str="main_exp",
    excluded_none=True,
    cropped_range=True,
    save_fig=True,
):

    if exp_str == "main_exp":
        fig_size_width_factor = 1
        instr_type_str = "instr_type_joint_none"
        instr_type_list = ["optimized", "natural", "none"]
        layer_labels_list = ["3a", "3b", "4a", "4b", "4c", "4d", "4e", "5a", "5b"]
        labels = ["Synthetic", "Natural", "None"]
    elif exp_str == "ablation_exp" or exp_str == "rebuttal_exp":
        fig_size_width_factor = 0.7
        instr_type_str = "instr_type"
        instr_type_list = ["optimized", "natural"]
        layer_labels_list = ["3a", "4a", "4c", "4e", "5b"]
        labels = ["Synthetic", "Natural"]

    dict_acc = {}
    dict_sem = {}
    for reference_type_i in instr_type_list:
        df_reference_type_i = data_df_main_exp_main[
            data_df_main_exp_main[instr_type_str] == reference_type_i
        ].copy()
        dict_acc_layer_number_i = {}
        dict_sem_layer_number_i = {}
        for layer_number_i in sorted(df_reference_type_i.layer.unique()):
            df_reference_type_i_layer_number_i = df_reference_type_i[
                df_reference_type_i["layer"] == layer_number_i
            ].copy()
            accuracy = (
                df_reference_type_i_layer_number_i["correct"] == True
            ).sum() / df_reference_type_i_layer_number_i.shape[0]
            dict_acc_layer_number_i[layer_number_i] = accuracy
            # evaluate standard error of the mean
            dict_acc__subject_i = {}
            for subject_i in df_reference_type_i_layer_number_i["subject_id"].unique():
                df_reference_type_i_layer_number_i_subject_i = df_reference_type_i_layer_number_i[
                    df_reference_type_i_layer_number_i["subject_id"] == subject_i
                ].copy()
                accuracy_subject_i = (
                    df_reference_type_i_layer_number_i_subject_i["correct"] == True
                ).sum() / df_reference_type_i_layer_number_i_subject_i.shape[0]
                dict_acc__subject_i[subject_i] = accuracy_subject_i
            n_data_points = len(dict_acc__subject_i)
            standard_deviation = np.std(
                np.fromiter(dict_acc__subject_i.values(), dtype=float)
            )
            sem = standard_deviation / np.sqrt(n_data_points)
            dict_sem_layer_number_i[layer_number_i] = sem
        dict_acc[reference_type_i] = dict_acc_layer_number_i
        dict_sem[reference_type_i] = dict_sem_layer_number_i

    fig, ax = plt.subplots(
        1, 1, figsize=(8 * fig_size_width_factor, 2.5), sharey=sharey
    )

    none_excluded = False
    if excluded_none:
        if "none" in dict_acc:
            none_excluded = True
            del dict_acc["none"]

    if len(dict_acc.items()) == 3:
        offsets = (-0.5 * 3 / 2, 0, 0.5 * 3 / 2)
    else:
        if none_excluded:
            offsets = (-0.4, 0.4)
        else:
            offsets = (-0.5, 0.5)

    for i, (factor_i, factor_i_data) in enumerate(dict_acc.items()):
        x = np.arange(len(factor_i_data)) * len(offsets)
        x = x + offsets[i]

        dict_2sem = {x: y * 2 for (x, y) in dict_sem[factor_i].items()}
        rect = ax.bar(
            x,
            list(factor_i_data.values()),
            yerr=dict_2sem.values(),
            error_kw=dict(lw=error_bar_linewidth),
            color=colors[i],
        )
    # ut_helper.autolabel(rect, axes, dict_2sem)
    ax.set_xticks(np.arange(len(factor_i_data)) * len(offsets))
    if exp_str == "rebuttal_exp":
        ax.set_xticklabels(["1", "2", "3", "4", "5", "6"])
    else:
        ax.set_xticklabels(layer_labels_list)
    # no axis on top and right
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=fontsize_tick_labels)
    if cropped_range:
        ax.set_ylim(0.5, 1)
    else:
        ax.set_ylim(0.0, 1)

    ax.axhline(0.5, color="k", linestyle="--", linewidth=1, label="Chance")
    ax.set_ylabel("Proportion Correct", fontsize=fontsize_axes_labels)

    if exp_str == "main_exp":
        plt.suptitle("Layer", x=0.526, y=0.05, fontsize=fontsize_axes_labels)
    elif exp_str == "ablation_exp" or exp_str == "rebuttal_exp":
        plt.suptitle("Layer", x=0.545, y=0.05, fontsize=fontsize_axes_labels)

    if cropped_range:
        yticks_and_labels = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    else:
        yticks_and_labels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_yticks(yticks_and_labels)
    ax.set_yticklabels(yticks_and_labels)
    plt.tight_layout()

    plot_name = "no_trend_across_layers_alternating"
    if cropped_range:
        plot_name += "_cropped"
    for version in range(100):
        file_name = os.path.join(
            "figures_ICLR_2020", f"{plot_name}_{exp_str}_{version}{file_type}"
        )
        # if file_name does not yet exist, use it
        if not os.path.exists(file_name):
            break
    if save_fig:
        print(f"figure saved under {file_name}")
        plt.savefig(file_name, bbox_inches="tight")

    plt.show()


def make_plot_no_trend_across_branches(
    data_df_main_exp_main, exp_str="main_exp", cropped_range=True, save_fig=True
):

    branches_labels_list = ["1x1", "3x3", "5x5", "pool"]
    if exp_str == "main_exp":
        fig_size_width_factor = 1
        instr_type_str = "instr_type_joint_none"
        instr_type_list = ["optimized", "natural", "none"]
        kernel_size_list = ["0", "1", "2", "3"]
        labels = ["Synthetic", "Natural", "None"]
    elif exp_str == "ablation_exp":
        fig_size_width_factor = 0.7
        instr_type_str = "instr_type"
        instr_type_list = ["optimized", "natural"]
        kernel_size_list = [0, 1, 2, 3]
        labels = ["Synthetic", "Natural"]

    dict_acc = {}
    dict_sem = {}
    for reference_type_i in instr_type_list:
        df_reference_type_i = data_df_main_exp_main[
            data_df_main_exp_main[instr_type_str] == reference_type_i
        ].copy()
        dict_acc_kernel_size_number_i = {}
        dict_sem_kernel_size_number_i = {}
        for kernel_size_number_i in kernel_size_list:
            df_reference_type_i_kernel_size_number_i = df_reference_type_i[
                df_reference_type_i["kernel_size"] == kernel_size_number_i
            ].copy()
            accuracy = (
                df_reference_type_i_kernel_size_number_i["correct"] == True
            ).sum() / df_reference_type_i_kernel_size_number_i.shape[0]
            dict_acc_kernel_size_number_i[kernel_size_number_i] = accuracy
            # evaluate standard error of the mean
            dict_acc__subject_i = {}
            for subject_i in df_reference_type_i_kernel_size_number_i[
                "subject_id"
            ].unique():
                df_reference_type_i_kernel_size_number_i_subject_i = df_reference_type_i_kernel_size_number_i[
                    df_reference_type_i_kernel_size_number_i["subject_id"] == subject_i
                ].copy()
                accuracy_subject_i = (
                    df_reference_type_i_kernel_size_number_i_subject_i["correct"]
                    == True
                ).sum() / df_reference_type_i_kernel_size_number_i_subject_i.shape[0]
                dict_acc__subject_i[subject_i] = accuracy_subject_i
            n_data_points = len(dict_acc__subject_i)
            standard_deviation = np.std(
                np.fromiter(dict_acc__subject_i.values(), dtype=float)
            )
            sem = standard_deviation / np.sqrt(n_data_points)
            dict_sem_kernel_size_number_i[kernel_size_number_i] = sem
        dict_acc[reference_type_i] = dict_acc_kernel_size_number_i
        dict_sem[reference_type_i] = dict_sem_kernel_size_number_i

    fig, axes = plt.subplots(
        1, len(instr_type_list), figsize=(5 * fig_size_width_factor, 2.5), sharey=sharey
    )

    for i, (factor_i, factor_i_data) in enumerate(dict_acc.items()):
        if i == 0:
            offset = 0
            label = "Synthetic"
        else:
            offset = 4.5
            label = "Natural"
        dict_2sem = {x: y * 2 for (x, y) in dict_sem[factor_i].items()}
        rect = axes[i].bar(
            np.arange(len(factor_i_data)),
            list(factor_i_data.values()),
            yerr=dict_2sem.values(),
            error_kw=dict(lw=error_bar_linewidth),
            color=colors[i],
            label=label,
        )
        #         ut_helper.autolabel(rect, axes, dict_2sem)
        axes[i].set_xticks(np.arange(len(factor_i_data)))
        axes[i].set_xticklabels(branches_labels_list)
        axes[i].set_xlabel(labels[i], fontsize=fontsize_tick_labels)

        # no axis on top and right
        axes[i].spines["top"].set_visible(False)
        axes[i].spines["right"].set_visible(False)
        axes[i].tick_params(labelsize=fontsize_tick_labels)

        if cropped_range:
            axes[i].set_ylim(0.5, 1)
        else:
            axes[i].set_ylim(0.0, 1)

        # axes[i].axhline(0.5,
        #           color='k',
        #           linestyle = '--',
        #           linewidth = 1,
        #           label='Chance'
        #           )

        if i == 0:
            axes[i].set_ylabel("Proportion Correct", fontsize=fontsize_axes_labels)
        else:
            axes[i].yaxis.set_visible(False)
            axes[i].spines["left"].set_visible(False)

    if exp_str == "main_exp":
        plt.suptitle(
            "Branch in Inception Module", x=0.55, y=0.02, fontsize=fontsize_axes_labels
        )
    elif exp_str == "ablation_exp":
        plt.suptitle(
            "Branch in Inception Module", x=0.565, y=0.02, fontsize=fontsize_axes_labels
        )

    if cropped_range:
        yticks_and_labels = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    else:
        yticks_and_labels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    axes[0].set_yticks(yticks_and_labels)
    axes[0].set_yticklabels(yticks_and_labels)
    plt.tight_layout()

    plot_name = "no_trend_across_branches"
    if cropped_range:
        plot_name += "_cropped"

    for version in range(100):
        file_name = os.path.join(
            "figures_ICLR_2020", f"{plot_name}_{exp_str}_{version}{file_type}"
        )
        # if file_name does not yet exist, use it
        if not os.path.exists(file_name):
            break
    if save_fig:
        print(f"figure saved under {file_name}")
        plt.savefig(file_name, bbox_inches="tight")

    plt.show()


def make_plot_no_trend_across_branches_alternating(
    data_df_main_exp_main,
    exp_str="main_exp",
    cropped_range=True,
    excluded_none=True,
    save_fig=True,
):

    branches_labels_list = ["1x1", "3x3", "5x5", "pool"]
    if exp_str == "main_exp":
        instr_type_str = "instr_type_joint_none"
        if excluded_none:
            fig_size_width_factor = 0.7
            instr_type_list = ["optimized", "natural"]
            labels = ["Synthetic", "Natural"]
        else:
            fig_size_width_factor = 1
            instr_type_list = ["optimized", "natural", "none"]
            labels = ["Synthetic", "Natural", "None"]
        kernel_size_list = ["0", "1", "2", "3"]
    elif exp_str == "ablation_exp":
        fig_size_width_factor = 0.7
        instr_type_str = "instr_type"
        instr_type_list = ["optimized", "natural"]
        kernel_size_list = [0, 1, 2, 3]
        labels = ["Synthetic", "Natural"]

    dict_acc = {}
    dict_sem = {}
    for reference_type_i in instr_type_list:
        df_reference_type_i = data_df_main_exp_main[
            data_df_main_exp_main[instr_type_str] == reference_type_i
        ].copy()
        dict_acc_kernel_size_number_i = {}
        dict_sem_kernel_size_number_i = {}
        for kernel_size_number_i in kernel_size_list:
            df_reference_type_i_kernel_size_number_i = df_reference_type_i[
                df_reference_type_i["kernel_size"] == kernel_size_number_i
            ].copy()
            accuracy = (
                df_reference_type_i_kernel_size_number_i["correct"] == True
            ).sum() / df_reference_type_i_kernel_size_number_i.shape[0]
            dict_acc_kernel_size_number_i[kernel_size_number_i] = accuracy
            # evaluate standard error of the mean
            dict_acc__subject_i = {}
            for subject_i in df_reference_type_i_kernel_size_number_i[
                "subject_id"
            ].unique():
                df_reference_type_i_kernel_size_number_i_subject_i = df_reference_type_i_kernel_size_number_i[
                    df_reference_type_i_kernel_size_number_i["subject_id"] == subject_i
                ].copy()
                accuracy_subject_i = (
                    df_reference_type_i_kernel_size_number_i_subject_i["correct"]
                    == True
                ).sum() / df_reference_type_i_kernel_size_number_i_subject_i.shape[0]
                dict_acc__subject_i[subject_i] = accuracy_subject_i
            n_data_points = len(dict_acc__subject_i)
            standard_deviation = np.std(
                np.fromiter(dict_acc__subject_i.values(), dtype=float)
            )
            sem = standard_deviation / np.sqrt(n_data_points)
            dict_sem_kernel_size_number_i[kernel_size_number_i] = sem
        dict_acc[reference_type_i] = dict_acc_kernel_size_number_i
        dict_sem[reference_type_i] = dict_sem_kernel_size_number_i

    fig, ax = plt.subplots(
        1, 1, figsize=(8 * fig_size_width_factor, 2.5), sharey=sharey
    )

    none_excluded = False
    if excluded_none:
        if "none" in dict_acc:
            none_excluded = True
            del dict_acc["none"]

    if len(dict_acc.items()) == 3:
        offsets = (-0.5 * 3 / 2, 0, 0.5 * 3 / 2)
    else:
        if none_excluded:
            offsets = (-0.4, 0.4)
        else:
            offsets = (-0.5, 0.5)

    for i, (factor_i, factor_i_data) in enumerate(dict_acc.items()):
        x = np.arange(len(factor_i_data)) * len(offsets)
        x = x + offsets[i]

        dict_2sem = {x: y * 2 for (x, y) in dict_sem[factor_i].items()}
        rect = ax.bar(
            x,
            list(factor_i_data.values()),
            yerr=dict_2sem.values(),
            error_kw=dict(lw=error_bar_linewidth),
            color=colors[i],
        )
        #         ut_helper.autolabel(rect, axes, dict_2sem)
        ax.set_xticks(np.arange(len(factor_i_data)) * len(offsets))
        if exp_str == "rebuttal_exp":
            ax.set_xticklabels(["1", "2", "3", "4", "5", "6"])
        else:
            ax.set_xticklabels(branches_labels_list)
        ax.set_xlabel(labels[i], fontsize=fontsize_tick_labels)

        # no axis on top and right
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=fontsize_tick_labels)

        if cropped_range:
            ax.set_ylim(0.5, 1)
        else:
            ax.set_ylim(0.0, 1)

        ax.set_ylabel("Proportion Correct", fontsize=fontsize_axes_labels)

    if exp_str == "main_exp":
        plt.suptitle(
            "Branch in Inception Module", x=0.55, y=0.02, fontsize=fontsize_axes_labels
        )
    elif exp_str == "ablation_exp":
        plt.suptitle(
            "Branch in Inception Module", x=0.565, y=0.02, fontsize=fontsize_axes_labels
        )

    if cropped_range:
        yticks_and_labels = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    else:
        yticks_and_labels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_yticks(yticks_and_labels)
    ax.set_yticklabels(yticks_and_labels)
    plt.tight_layout()

    plot_name = "no_trend_across_branches_alternating"
    if cropped_range:
        plot_name += "_cropped"

    for version in range(100):
        file_name = os.path.join(
            "figures_ICLR_2020", f"{plot_name}_{exp_str}_{version}{file_type}"
        )
        # if file_name does not yet exist, use it
        if not os.path.exists(file_name):
            break
    if save_fig:
        print(f"figure saved under {file_name}")
        plt.savefig(file_name, bbox_inches="tight")

    plt.show()


def make_plot_presentation_style(data_df_ablation_exp_main, save_fig=True):

    presentation_condition_labels_list = ["Max 1", "Max 9", "Min+Max 1", "Min+Max 9"]
    labels = ["Synthetic", "Natural"]

    dict_acc = {}
    dict_sem = {}
    for factor_i in ["optimized", "natural"]:
        df_factor_i = data_df_ablation_exp_main[
            data_df_ablation_exp_main["instr_type"] == factor_i
        ].copy()

        dict_acc_factor_ii = {}
        dict_sem_factor_ii = {}
        for factor_ii in ["Max-1", "Max-9", "Min-Max-1", "Min-Max-9"]:
            df_factor_i_factor_ii = df_factor_i[
                df_factor_i["condition_name"] == factor_ii
            ].copy()
            accuracy = (
                df_factor_i_factor_ii["correct"] == True
            ).sum() / df_factor_i_factor_ii.shape[0]
            dict_acc_factor_ii[factor_ii] = accuracy
            # evaluate standard error of the mean
            dict_acc__subject_i = {}
            for subject_i in df_factor_i_factor_ii["subject_id"].unique():
                df_factor_i_factor_ii_subject_i = df_factor_i_factor_ii[
                    df_factor_i_factor_ii["subject_id"] == subject_i
                ].copy()
                accuracy = (
                    df_factor_i_factor_ii_subject_i["correct"] == True
                ).sum() / df_factor_i_factor_ii_subject_i.shape[0]
                dict_acc__subject_i[subject_i] = accuracy
            n_data_points = len(dict_acc__subject_i)
            standard_deviation = np.std(
                np.fromiter(dict_acc__subject_i.values(), dtype=float)
            )
            sem = standard_deviation / np.sqrt(n_data_points)
            dict_sem_factor_ii[factor_ii] = sem
        dict_acc[factor_i] = dict_acc_factor_ii
        dict_sem[factor_i] = dict_sem_factor_ii

    fig, axes = plt.subplots(1, 2 + 1, figsize=(6.75, 2.5), sharey=sharey)

    for i, (factor_i, factor_i_data) in enumerate(dict_acc.items()):
        dict_2sem = {x: y * 2 for (x, y) in dict_sem[factor_i].items()}
        rect = axes[i].bar(
            np.arange(len(factor_i_data)),
            list(factor_i_data.values()),
            yerr=dict_2sem.values(),
            error_kw=dict(lw=error_bar_linewidth),
            color=colors[i],
        )
        #         ut_helper.autolabel(rect, axes, dict_2sem)

        axes[i].set_xticks(np.arange(len(factor_i_data)))
        axes[i].set_xticklabels(
            presentation_condition_labels_list, rotation=x_tick_label_rotation
        )
        axes[i].set_xlabel(labels[i], fontsize=fontsize_tick_labels)
        # no axis on top and right
        axes[i].spines["top"].set_visible(False)
        axes[i].spines["right"].set_visible(False)
        axes[i].tick_params(labelsize=fontsize_tick_labels)
        axes[i].set_ylim(0, 1)

        axes[i].axhline(0.5, color="k", linestyle="--", linewidth=1, label="Chance")

        if i == 0:
            axes[i].set_ylabel("Proportion Correct", fontsize=fontsize_axes_labels)
        else:
            axes[i].yaxis.set_visible(False)
            axes[i].spines["left"].set_visible(False)

        axes[i].set_ylim(0, 1.2)
        yticks_and_labels = [0, 0.2, 0.4, 0.6, 0.8, 1]
        axes[i].set_yticks(yticks_and_labels)
        axes[i].set_yticklabels(yticks_and_labels)

    axes[2].bar([0, 1], [1, 1])
    axes[2].set_xticks(np.arange(len([1, 1])))
    axes[2].set_xticklabels(uniform_sizing_labels_list, rotation=x_tick_label_rotation)
    axes[2].set_xlabel(labels[i], fontsize=fontsize_tick_labels)

    plt.suptitle("Presentation Scheme", x=0.54, y=0.02, fontsize=fontsize_axes_labels)

    plt.tight_layout()

    plot_name = "presentation_style"
    exp_str = "ablation_exp"
    for version in range(100):
        file_name = os.path.join(
            "figures_ICLR_2020", f"{plot_name}_{exp_str}_{version}{file_type}"
        )
        # if file_name does not yet exist, use it
        if not os.path.exists(file_name):
            break
    if save_fig:
        print(f"figure saved under {file_name}")
        plt.savefig(file_name, bbox_inches="tight")

    plt.show()


def make_plot_random_vs_cherry_picked(data_df_main_exp_main, save_fig=True):
    data_df_main_exp_main_only_pool = data_df_main_exp_main[
        data_df_main_exp_main["kernel_size"] == "3"
    ].copy()
    labels = ["Synthetic", "Natural"]
    random_cherry_picked_labels_list = ["Random", "Hand-\nPicked"]

    dict_acc = {}
    dict_sem = {}
    for factor_i in ["optimized", "natural"]:
        df_factor_i = data_df_main_exp_main_only_pool[
            data_df_main_exp_main_only_pool["instr_type_joint_none"] == factor_i
        ].copy()

        dict_acc_factor_ii = {}
        dict_sem_factor_ii = {}
        for factor_ii in ["channel_0", "channel_1"]:
            df_factor_i_factor_ii = df_factor_i[
                df_factor_i["channel"] == factor_ii
            ].copy()
            accuracy = (
                df_factor_i_factor_ii["correct"] == True
            ).sum() / df_factor_i_factor_ii.shape[0]
            dict_acc_factor_ii[factor_ii] = accuracy
            # evaluate standard error of the mean
            dict_acc__subject_i = {}
            for subject_i in df_factor_i_factor_ii["subject_id"].unique():
                df_factor_i_factor_ii_subject_i = df_factor_i_factor_ii[
                    df_factor_i_factor_ii["subject_id"] == subject_i
                ].copy()
                accuracy = (
                    df_factor_i_factor_ii_subject_i["correct"] == True
                ).sum() / df_factor_i_factor_ii_subject_i.shape[0]
                dict_acc__subject_i[subject_i] = accuracy
            n_data_points = len(dict_acc__subject_i)
            standard_deviation = np.std(
                np.fromiter(dict_acc__subject_i.values(), dtype=float)
            )
            sem = standard_deviation / np.sqrt(n_data_points)
            dict_sem_factor_ii[factor_ii] = sem
        dict_acc[factor_i] = dict_acc_factor_ii
        dict_sem[factor_i] = dict_sem_factor_ii

    fig, axes = plt.subplots(1, 2 + 1, figsize=(3.8, 2.5), sharey=sharey)

    for i, (factor_i, factor_i_data) in enumerate(dict_acc.items()):
        dict_2sem = {x: y * 2 for (x, y) in dict_sem[factor_i].items()}
        rect = axes[i].bar(
            np.arange(len(factor_i_data)),
            list(factor_i_data.values()),
            yerr=dict_2sem.values(),
            error_kw=dict(lw=error_bar_linewidth),
            color=colors[i],
        )
        #         ut_helper.autolabel(rect, axes, dict_2sem)

        axes[i].set_xticks(np.arange(len(factor_i_data)))
        axes[i].set_xticklabels(
            random_cherry_picked_labels_list, rotation=x_tick_label_rotation
        )
        axes[i].set_xlabel(labels[i], fontsize=fontsize_tick_labels)
        # no axis on top and right
        axes[i].spines["top"].set_visible(False)
        axes[i].spines["right"].set_visible(False)
        axes[i].tick_params(labelsize=fontsize_tick_labels)

        axes[i].axhline(0.5, color="k", linestyle="--", linewidth=1, label="Chance")
        # significance
        height_bar_and_error_list = []
        for rect_i, error_value_i in zip(rect, dict_2sem.values()):
            height = rect_i.get_height()
            height_bar_and_error_list.append(height + error_value_i)
        ut_helper.barplot_annotate_brackets(
            axes[i], 0, 1, "ns", np.arange(2), height_bar_and_error_list
        )

        if i == 0:
            axes[i].set_ylabel("Proportion Correct", fontsize=fontsize_axes_labels)
        else:
            axes[i].yaxis.set_visible(False)
            axes[i].spines["left"].set_visible(False)

        axes[i].set_ylim(0, 1.2)
        yticks_and_labels = [0, 0.2, 0.4, 0.6, 0.8, 1]
        axes[i].set_yticks(yticks_and_labels)
        axes[i].set_yticklabels(yticks_and_labels)

    axes[2].bar([0, 1], [1, 1])
    axes[2].set_xticks(np.arange(len([1, 1])))
    axes[2].set_xticklabels(uniform_sizing_labels_list, rotation=x_tick_label_rotation)
    axes[2].set_xlabel(labels[i], fontsize=fontsize_tick_labels)

    plt.suptitle("Selection Mode", x=0.59, y=0.02, fontsize=fontsize_axes_labels)

    yticks_and_labels = [0, 0.2, 0.4, 0.6, 0.8, 1]
    axes[0].set_yticks(yticks_and_labels)
    axes[0].set_yticklabels(yticks_and_labels)
    plt.tight_layout()

    plot_name = "random_vs_hand_picked"
    exp_str = "main_exp"
    for version in range(100):
        file_name = os.path.join(
            "figures_ICLR_2020", f"{plot_name}_{exp_str}_{version}{file_type}"
        )
        # if file_name does not yet exist, use it
        if not os.path.exists(file_name):
            break
    if save_fig:
        print(f"figure saved under {file_name}")
        plt.savefig(file_name, bbox_inches="tight")

    plt.show()


def make_plot_expert_vs_lay(data_df_ablation_exp_main, save_fig=True):
    dict_acc = {}
    dict_sem = {}
    labels = ["Synthetic", "Natural"]
    expert_level_labels_list = ["Expert", "Lay"]

    for factor_i in ["optimized", "natural"]:
        df_factor_i = data_df_ablation_exp_main[
            data_df_ablation_exp_main["instr_type"] == factor_i
        ].copy()

        dict_acc_factor_ii = {}
        dict_sem_factor_ii = {}
        for factor_ii in ["expert", "naive"]:
            df_factor_i_factor_ii = df_factor_i[
                df_factor_i["expert_level"] == factor_ii
            ].copy()
            accuracy = (
                df_factor_i_factor_ii["correct"] == True
            ).sum() / df_factor_i_factor_ii.shape[0]
            dict_acc_factor_ii[factor_ii] = accuracy
            # evaluate standard error of the mean
            dict_acc__subject_i = {}
            for subject_i in df_factor_i_factor_ii["subject_id"].unique():
                df_factor_i_factor_ii_subject_i = df_factor_i_factor_ii[
                    df_factor_i_factor_ii["subject_id"] == subject_i
                ].copy()
                accuracy = (
                    df_factor_i_factor_ii_subject_i["correct"] == True
                ).sum() / df_factor_i_factor_ii_subject_i.shape[0]
                dict_acc__subject_i[subject_i] = accuracy
            n_data_points = len(dict_acc__subject_i)
            standard_deviation = np.std(
                np.fromiter(dict_acc__subject_i.values(), dtype=float)
            )
            sem = standard_deviation / np.sqrt(n_data_points)
            dict_sem_factor_ii[factor_ii] = sem
        dict_acc[factor_i] = dict_acc_factor_ii
        dict_sem[factor_i] = dict_sem_factor_ii

    fig, axes = plt.subplots(1, 2 + 1, figsize=(3.8, 2.5), sharey=sharey)

    for i, (factor_i, factor_i_data) in enumerate(dict_acc.items()):
        dict_2sem = {x: y * 2 for (x, y) in dict_sem[factor_i].items()}
        rect = axes[i].bar(
            np.arange(len(factor_i_data)),
            list(factor_i_data.values()),
            yerr=dict_2sem.values(),
            error_kw=dict(lw=error_bar_linewidth),
            color=colors[i],
        )
        #         ut_helper.autolabel(rect, axes, dict_2sem)

        axes[i].set_xticks(np.arange(len(factor_i_data)))
        axes[i].set_xticklabels(
            expert_level_labels_list, rotation=x_tick_label_rotation
        )
        axes[i].set_xlabel(labels[i], fontsize=fontsize_tick_labels)
        # no axis on top and right
        axes[i].spines["top"].set_visible(False)
        axes[i].spines["right"].set_visible(False)
        axes[i].tick_params(labelsize=fontsize_tick_labels)

        axes[i].axhline(0.5, color="k", linestyle="--", linewidth=1, label="Chance")

        # significance
        height_bar_and_error_list = []
        for rect_i, error_value_i in zip(rect, dict_2sem.values()):
            height = rect_i.get_height()
            height_bar_and_error_list.append(height + error_value_i)
        ut_helper.barplot_annotate_brackets(
            axes[i], 0, 1, "ns", np.arange(2), height_bar_and_error_list
        )

        if i == 0:
            axes[i].set_ylabel("Proportion Correct", fontsize=fontsize_axes_labels)
        else:
            axes[i].yaxis.set_visible(False)
            axes[i].spines["left"].set_visible(False)

        axes[i].set_ylim(0, 1.2)
        yticks_and_labels = [0, 0.2, 0.4, 0.6, 0.8, 1]
        axes[i].set_yticks(yticks_and_labels)
        axes[i].set_yticklabels(yticks_and_labels)

    axes[2].bar([0, 1], [1, 1])
    axes[2].set_xticks(np.arange(len([1, 1])))
    axes[2].set_xticklabels(uniform_sizing_labels_list, rotation=x_tick_label_rotation)
    axes[2].set_xlabel(labels[i], fontsize=fontsize_tick_labels)

    plt.suptitle("Expert Level", x=0.59, y=0.02, fontsize=fontsize_axes_labels)

    plt.tight_layout()

    plot_name = "expert_vs_lay"
    exp_str = "ablation_exp"
    for version in range(100):
        file_name = os.path.join(
            "figures_ICLR_2020", f"{plot_name}_{exp_str}_{version}{file_type}"
        )
        # if file_name does not yet exist, use it
        if not os.path.exists(file_name):
            break
    if save_fig:
        print(f"figure saved under {file_name}")
        plt.savefig(file_name, bbox_inches="tight")

    plt.show()


### Intuitiveness


def get_mean_of_list(list_of_means):
    list_mean = sum(list_of_means) / len(list_of_means)
    return list_mean


def get_sem_of_list(list_of_means):
    n_data_points = len(list_of_means)
    standard_deviation = np.std(np.array(list_of_means))
    sem = standard_deviation / np.sqrt(n_data_points)
    return 2 * sem


def make_plot_differences_in_intuitiveness(
    warm_up_data_beginning_df_final, warm_up_data_end_df_final, save_fig=True
):

    # insert difference between ratings per subject
    difference_bw_rating_before_and_after_dict = {}
    mean_list_a = []
    mean_list_b = []
    mean_list_c = []
    for subject_i in sorted(warm_up_data_beginning_df_final.subject_id.unique()):
        warm_up_data_beginning_df_final_subject_i = warm_up_data_beginning_df_final[
            warm_up_data_beginning_df_final["subject_id"] == subject_i
        ].copy()
        warm_up_data_end_df_final_subject_i = warm_up_data_end_df_final[
            warm_up_data_end_df_final["subject_id"] == subject_i
        ].copy()
        difference_subject_i_dict = {}
        for trial_i in ["a", "b", "c"]:
            rating_beginning = warm_up_data_beginning_df_final_subject_i[
                warm_up_data_beginning_df_final_subject_i["trial_name"] == trial_i
            ].rating
            rating_end = warm_up_data_end_df_final_subject_i[
                warm_up_data_end_df_final_subject_i["trial_name"] == trial_i
            ].rating
            difference_subject_i_trial_i = rating_end.item() - rating_beginning.item()
            difference_subject_i_dict[trial_i] = difference_subject_i_trial_i
            if trial_i == "a":
                mean_list_a.append(difference_subject_i_trial_i)
            elif trial_i == "b":
                mean_list_b.append(difference_subject_i_trial_i)
            elif trial_i == "c":
                mean_list_c.append(difference_subject_i_trial_i)
        difference_bw_rating_before_and_after_dict[
            subject_i
        ] = difference_subject_i_dict

    fig, ax = plt.subplots(1, 1, figsize=(3, 2.5))

    for (
        subject_i,
        difference_subject_i_dict,
    ) in difference_bw_rating_before_and_after_dict.items():
        ax.plot(
            np.arange(len(difference_subject_i_dict)) + np.random.rand(1) / 10 - 0.05,
            list(difference_subject_i_dict.values()),
            color="gray",
            alpha=0.4,
            marker=".",
            linestyle="None",
        )

    plt.errorbar(
        np.arange(len(difference_subject_i_dict)),
        [
            get_mean_of_list(mean_list_a),
            get_mean_of_list(mean_list_b),
            get_mean_of_list(mean_list_c),
        ],
        yerr=[
            get_sem_of_list(mean_list_a),
            get_sem_of_list(mean_list_b),
            get_sem_of_list(mean_list_c),
        ],
        color=colors[0],
        label="mean",
        fmt="o",
    )

    ax.axhline(
        0,
        color="k",
        # linestyle = ':',
        linewidth=1,
        alpha=0.35,
    )

    ax.set_xticks(np.arange(warm_up_data_end_df_final.trial_name.unique().shape[0]))
    ax.set_xticklabels(["3a", "4b", "5b"])

    # no axis on top and right
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=fontsize_tick_labels)

    ax.set_xlabel("Layer", fontsize=fontsize_axes_labels)
    ax.set_ylabel("Difference in Intuitiveness", fontsize=fontsize_axes_labels)

    plot_name = "differences_in_intuitiveness"
    exp_str = "ablation_exp"
    for version in range(100):
        file_name = os.path.join(
            "figures_ICLR_2020", f"{plot_name}_{exp_str}_{version}{file_type}"
        )
        # if file_name does not yet exist, use it
        if not os.path.exists(file_name):
            break
    if save_fig:
        print(f"figure saved under {file_name}")
        plt.savefig(file_name, bbox_inches="tight")

    plt.show()


def get_dict_rating_and_dict_sem(warm_up_data_df, factor_i_str):

    dict_rating = {}
    dict_sem = {}
    for factor_i in warm_up_data_df[factor_i_str].unique():
        df_factor_i = warm_up_data_df[warm_up_data_df[factor_i_str] == factor_i].copy()
        rating_mean = (df_factor_i["rating"]).mean()
        dict_rating[factor_i] = rating_mean
        # standard error of the mean
        standard_deviation = (df_factor_i["rating"]).std()
        n_data_points = df_factor_i.shape[0]
        sem = standard_deviation / np.sqrt(n_data_points)
        dict_sem[factor_i] = sem

    return dict_rating, dict_sem


def make_plot_absolute_intuitiveness(
    warm_up_data_beginning_df_final, warm_up_data_end_df_final, save_fig=True
):

    plot_name = "absolute_intuitiveness_of_feature_vis_method"
    exp_str = "ablation"
    factor_i_str = "trial_name"
    offset = 0.2

    fig, ax = plt.subplots(1, 1, figsize=(3, 2.5))

    # line for no difference
    ax.axhline(0, color="k", linewidth=1, alpha=0.35)

    # subject lines
    idx = 0
    for subject_i in warm_up_data_beginning_df_final.subject_id.unique():
        df_before_subject_i = warm_up_data_beginning_df_final[
            warm_up_data_beginning_df_final["subject_id"] == subject_i
        ].copy()
        df_after_subject_i = warm_up_data_end_df_final[
            warm_up_data_end_df_final["subject_id"] == subject_i
        ].copy()
        for trial_name_number_i in df_before_subject_i.trial_name_number.unique():
            df_before_subject_i_trial_i = df_before_subject_i[
                df_before_subject_i["trial_name_number"] == trial_name_number_i
            ]
            df_after_subject_i_trial_i = df_after_subject_i[
                df_after_subject_i["trial_name_number"] == trial_name_number_i
            ]
            rating_before = df_before_subject_i_trial_i.rating.item()
            rating_after = df_after_subject_i_trial_i.rating.item()
            ax.plot(
                [
                    trial_name_number_i - offset,
                    trial_name_number_i + offset,
                ],  # x-position
                [rating_before, rating_after],  # rating
                color="gray",
                alpha=0.25,
            )
            idx += 1

    # means
    dict_rating_before, dict_sem_before = get_dict_rating_and_dict_sem(
        warm_up_data_beginning_df_final, factor_i_str
    )
    dict_rating_after, dict_sem_after = get_dict_rating_and_dict_sem(
        warm_up_data_end_df_final, factor_i_str
    )

    dict_2sem_before = {x: y * 2 for (x, y) in dict_sem_before.items()}
    dict_2sem_after = {x: y * 2 for (x, y) in dict_sem_after.items()}

    idx = 0
    for rating_before_i, sem_before_i, rating_after_i, sem_after_i in zip(
        dict_rating_before.values(),
        dict_2sem_before.values(),
        dict_rating_after.values(),
        dict_2sem_after.values(),
    ):
        ax.errorbar(
            [idx - offset, idx + offset],
            [rating_before_i, rating_after_i],
            yerr=[sem_before_i, sem_after_i],
            #             error_kw=dict(lw=error_bar_linewidth),
            color=colors[0],
        )
        idx += 1

    ax.set_xticks(np.arange(warm_up_data_end_df_final.trial_name.unique().shape[0]))
    ax.set_xticklabels(["3a", "4b", "5b"])

    # no axis on top and right
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=fontsize_tick_labels)

    ax.set_xlabel("Layer", fontsize=fontsize_axes_labels)
    ax.set_ylabel("Intuitiveness", fontsize=fontsize_axes_labels)
    ax.set_ylim(-105, 105)

    for version in range(100):
        file_name = os.path.join(
            "figures_ICLR_2020", f"{plot_name}_{exp_str}_{version}{file_type}"
        )
        # if file_name does not yet exist, use it
        if not os.path.exists(file_name):
            break
    if save_fig:
        print(f"figure saved under {file_name}")
        plt.savefig(file_name, bbox_inches="tight")

    plt.show()


def add_trial_name_as_jittered_number(warm_up_data_df):
    """add column where trial_name is converted to number and jitter is added"""
    warm_up_data_df.insert(
        warm_up_data_df.shape[1], "trial_name_as_jittered_number", float(100)
    )
    for index, row in warm_up_data_df.iterrows():
        if row.trial_name == "a":
            warm_up_data_df.at[index, "trial_name_as_jittered_number"] = (
                0 + np.random.rand(1) / 10 - 0.05
            )
        elif row.trial_name == "b":
            warm_up_data_df.at[index, "trial_name_as_jittered_number"] = (
                1 + np.random.rand(1) / 10 - 0.05
            )
        elif row.trial_name == "c":
            warm_up_data_df.at[index, "trial_name_as_jittered_number"] = (
                2 + np.random.rand(1) / 10 - 0.05
            )
    return warm_up_data_df


def plot_intuitiveness_experiment_I(warm_up_data_df, jitter=False, save_fig=True):

    factor_i_str = "trial_name"

    dict_rating = {}
    dict_sem = {}
    for factor_i in warm_up_data_df[factor_i_str].unique():
        df_factor_i = warm_up_data_df[warm_up_data_df[factor_i_str] == factor_i].copy()
        rating_mean = (df_factor_i["rating"]).mean()
        dict_rating[factor_i] = rating_mean
        # standard error of the mean
        standard_deviation = (df_factor_i["rating"]).std()
        n_data_points = df_factor_i.shape[0]
        sem = standard_deviation / np.sqrt(n_data_points)
        dict_sem[factor_i] = sem

    fig, ax = plt.subplots(1, 1, figsize=(3, 2.5))

    rating = warm_up_data_df.rating
    if jitter:
        trial_name = warm_up_data_df.trial_name_as_jittered_number
    else:
        trial_name = warm_up_data_df.trial_name_corrected_by_before_and_after
    for subject_i in range(warm_up_data_df.subject_id.max()):
        ax.plot(
            trial_name[(subject_i * 3) : (3 + subject_i * 3)],
            rating[(subject_i * 3) : (3 + subject_i * 3)],
            ".",
            color="gray",
            alpha=0.6,
        )
    ax.axhline(0, color="k", linewidth=1, alpha=0.35)

    # 2 SEM
    dict_2sem = {x: y * 2 for (x, y) in dict_sem.items()}
    ax.errorbar(
        range(len(dict_rating)),
        list(dict_rating.values()),
        yerr=dict_2sem.values(),
        fmt="o",
        color=colors[0],
    )

    ax.set_xticks(np.arange(warm_up_data_df.trial_name.unique().shape[0]))
    ax.set_xticklabels(["3a", "4b", "5b"])

    # no axis on top and right
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=fontsize_tick_labels)

    ax.set_xlabel("Layer", fontsize=fontsize_axes_labels)
    ax.set_ylabel("Intuitiveness", fontsize=fontsize_axes_labels)
    ax.set_ylim(-105, 105)

    plot_name = "intuitiveness_beginning_only"
    exp_str = "main_exp"
    for version in range(100):
        file_name = os.path.join(
            "figures_ICLR_2020", f"{plot_name}_{exp_str}_{version}{file_type}"
        )
        # if file_name does not yet exist, use it
        if not os.path.exists(file_name):
            break
    if save_fig:
        print(f"figure saved under {file_name}")
        plt.savefig(file_name, bbox_inches="tight")

    plt.show()


def make_plot_by_expert_comparison(
    data_df_main_exp_main,
    exp_str="main_exp",
    expert_specification_string="some_expert",
    save_fig=True,
):

    if exp_str == "main_exp":
        fig_size_width_factor = 1
        instr_type_str = "instr_type_joint_none"
        instr_type_joint_none_list_in_nice_order = ["optimized", "natural", "none"]
    elif exp_str == "ablation_exp":
        fig_size_width_factor = 0.7
        instr_type_str = "instr_type"
        instr_type_joint_none_list_in_nice_order = ["optimized", "natural"]

    dict_acc = {}
    dict_sem = {}
    for reference_type_i in instr_type_joint_none_list_in_nice_order:
        df_factor_i = data_df_main_exp_main[
            data_df_main_exp_main[instr_type_str] == reference_type_i
        ].copy()
        accuracy = (df_factor_i["correct"] == True).sum() / df_factor_i.shape[0]
        dict_acc[reference_type_i] = accuracy
        # evaluate standard error of the mean
        dict_acc_subject_i = {}
        for subject_i in df_factor_i["subject_id"].unique():
            df_factor_i_subject = df_factor_i[
                df_factor_i["subject_id"] == subject_i
            ].copy()
            accuracy_subject_i = (
                df_factor_i_subject["correct"] == True
            ).sum() / df_factor_i_subject.shape[0]
            dict_acc_subject_i[subject_i] = accuracy_subject_i
        n_data_points = len(dict_acc_subject_i)
        standard_deviation = np.std(
            np.fromiter(dict_acc_subject_i.values(), dtype=float)
        )
        sem = standard_deviation / np.sqrt(n_data_points)
        dict_sem[reference_type_i] = sem

    ### figure
    fig, ax = plt.subplots(1, 1, figsize=(2.2 * fig_size_width_factor, 3))

    # 2 SEM
    dict_2sem = {x: y * 2 for (x, y) in dict_sem.items()}
    rect = ax.bar(
        range(len(dict_acc)),
        list(dict_acc.values()),
        color=colors,
        yerr=dict_2sem.values(),
        error_kw=dict(lw=error_bar_linewidth),
        align="center",
    )
    ut_helper.autolabel(rect, ax, dict_2sem)
    ax.axhline(0.5, color="k", linestyle="--", linewidth=1, label="Chance")

    ax.set_ylabel("Proportion Correct", fontsize=fontsize_axes_labels)

    yticks_and_labels = [0, 0.2, 0.4, 0.6, 0.8, 1]
    ax.set_yticks(yticks_and_labels)
    ax.set_yticklabels(yticks_and_labels)

    # xticks
    ax.set_xticks(range(len(dict_acc)))
    ax.set_xticklabels(["Synthetic", "Natural", "None"])

    ax.tick_params(labelsize=fontsize_tick_labels)

    # no axis on top and right
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlabel("Reference Images", fontsize=fontsize_axes_labels)

    plot_name = "by_expert_level_comparison"
    for version in range(100):
        file_name = os.path.join(
            "figures_ICLR_2020",
            f"{plot_name}_{expert_specification_string}_{exp_str}_{version}{file_type}",
        )
        # if file_name does not yet exist, use it
        if not os.path.exists(file_name):
            break
    if save_fig:
        print(f"figure saved under {file_name}")
        plt.savefig(file_name, bbox_inches="tight")

    plt.show()
