import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


###### Data Processing ##############


def separate_practice_main_repeated_and_catch_trials(
    data_df, main_ablation_rebuttal_exp
):

    data_df_practice = data_df[data_df["trial_type"] == "practice"].copy()
    data_df_main = data_df[data_df["trial_type"] == "main"].copy()
    # separate catch trials (only applicable to main trial)
    data_df_catch_trials = data_df_main[data_df_main["catch_trial"] == True].copy()
    data_df_main = data_df_main[data_df_main["catch_trial"] == False]

    if main_ablation_rebuttal_exp == "main_exp":
        data_df_repeated = data_df[data_df["trial_type"] == "repeated"].copy()
        data_df_repeated.old_trial_nr = data_df_repeated.old_trial_nr.map(
            lambda x: int(x)
        )
        return data_df_practice, data_df_main, data_df_catch_trials, data_df_repeated
    elif (
        main_ablation_rebuttal_exp == "ablation_exp"
        or main_ablation_rebuttal_exp == "rebuttal_exp"
    ):
        return data_df_practice, data_df_main, data_df_catch_trials


########## PLOTS ##############


def autolabel(rects, ax, error_bar, rotation=90):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect, error_value_i in zip(rects, error_bar.values()):
        height = rect.get_height()
        height_rounded = round(height, 2)
        error_rounded = round(error_value_i, 2)
        text_y = 0.02
        ax.annotate(
            f"{height_rounded}\u00B1{error_rounded}",
            xy=(rect.get_x() + rect.get_width() / 2, text_y),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            rotation=rotation,
        )


def autolabel_counts(rects, ax, rotation=0):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        height_rounded = round(height, 2)
        text_y = height
        ax.annotate(
            f"{height_rounded}",
            xy=(rect.get_x() + rect.get_width() / 2, text_y),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            rotation=rotation,
        )


def barplot_annotate_brackets(
    axes,
    num1,
    num2,
    data,
    center,
    height,
    yerr=None,
    dh=0.03,
    barh=0.03,
    fontsize=8,
    maxasterix=None,
):
    """ 
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fontsize: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    
    credit to https://stackoverflow.com/questions/11517986/indicating-the-statistically-significant-difference-in-bar-graph
    """

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ""
        p = 0.05

        while data < p:
            text += "*"
            p /= 10.0

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = "n. s."

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= ax_y1 - ax_y0
    barh *= ax_y1 - ax_y0

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y + barh, y + barh, y]
    mid = ((lx + rx) / 2, y + barh)

    axes.plot(barx, bary, c="black", linewidth=1)

    kwargs = dict(ha="center", va="bottom")
    if fontsize is not None:
        kwargs["fontsize"] = fontsize

    axes.text(*mid, text, **kwargs)


def add_trial_name_as_number(warm_up_data_df):
    """add column where trial_name is converted to number and before-/after-correction is added"""
    warm_up_data_df.insert(
        warm_up_data_df.shape[1], "trial_name_corrected_by_before_and_after", float(100)
    )
    offset = 0
    for index, row in warm_up_data_df.iterrows():
        if row.trial_name == "a":
            warm_up_data_df.at[index, "trial_name_number"] = 0 + offset
        elif row.trial_name == "b":
            warm_up_data_df.at[index, "trial_name_number"] = 1 + offset
        elif row.trial_name == "c":
            warm_up_data_df.at[index, "trial_name_number"] = 2 + offset
    return warm_up_data_df


def add_trial_name_as_jittered_number(warm_up_data_df):
    """add column where trial_name is converted to number and jitter is added"""
    warm_up_data_df.insert(
        warm_up_data_df.shape[1], "trial_name_as_jittered_number", float(100)
    )
    jitter_array = np.random.normal(0, 0.1, warm_up_data_df.shape[0])
    jitter_counter_i = 0
    for index, row in warm_up_data_df.iterrows():
        if row.trial_name == "a":
            warm_up_data_df.at[index, "trial_name_as_jittered_number"] = (
                0 + jitter_array[jitter_counter_i]
            )
        elif row.trial_name == "b":
            warm_up_data_df.at[index, "trial_name_as_jittered_number"] = (
                1 + jitter_array[jitter_counter_i]
            )
        elif row.trial_name == "c":
            warm_up_data_df.at[index, "trial_name_as_jittered_number"] = (
                2 + jitter_array[jitter_counter_i]
            )
        jitter_counter_i += 1
    return warm_up_data_df
