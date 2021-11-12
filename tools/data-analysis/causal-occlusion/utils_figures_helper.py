import numpy as np
import matplotlib.pyplot as plt


def autolabel(rects, ax, error_bar, rotation=90, color="black", fontsize=None):
    """Attach a text label above each bar in *rects*, displaying its height."""
    if isinstance(error_bar, dict):
        error_bar = error_bar.values()

    if error_bar is None:
        error_bar = [np.nan for _ in range(len(rects))]

    for rect, error_value_i in zip(rects, error_bar):
        height = rect.get_height()
        height_rounded = round(height, 2)
        error_rounded = round(error_value_i, 2)
        #         text_y = height + error_value_i
        text_y = 0.02
        if np.isnan(height_rounded):
            continue
        if np.isnan(error_rounded):
            ax.annotate(
                f"{height_rounded:.02f}",
                xy=(rect.get_x() + rect.get_width() / 2, text_y),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                rotation=rotation,
                color=color,
                fontsize=fontsize,
            )
        else:
            ax.annotate(
                f"{height_rounded:.02f}\u00B1{error_rounded:.02f}",
                xy=(rect.get_x() + rect.get_width() / 2, text_y),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                rotation=rotation,
                color=color,
                fontsize=fontsize,
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


# cohen's kappa
def get_sum_of_equal_entries(
    df_condition_i_batch_i_subject_i,
    df_condition_ii_batch_i_subject_i,
    column_i,
    column_ii,
):
    sum_equal_entries = (
        df_condition_i_batch_i_subject_i.reset_index(drop=True)[column_i]
        == df_condition_ii_batch_i_subject_i.reset_index(drop=True)[column_ii]
    ).sum()
    return sum_equal_entries


def get_cohens_kappa(
    df_condition_i_batch_i_subject_i,
    df_condition_ii_batch_i_subject_i,
    column_i,
    column_ii,
    n_data_points=18,
):
    probability_observed_agreement = (
        get_sum_of_equal_entries(
            df_condition_i_batch_i_subject_i,
            df_condition_ii_batch_i_subject_i,
            column_i,
            column_ii,
        )
        / n_data_points
    )
    assert (
        probability_observed_agreement >= 0 and probability_observed_agreement <= 1
    ), f"probability_observed_agreement is outside of range 0-1: {probability_observed_agreement}"

    probability_correct_cond_i = (
        df_condition_i_batch_i_subject_i[column_i].sum() / n_data_points
    )
    assert (
        probability_correct_cond_i >= 0 and probability_correct_cond_i <= 1
    ), f"probability_correct_cond_i is outside of range 0-1: {probability_correct_cond_i}"

    probability_correct_cond_ii = (
        df_condition_ii_batch_i_subject_i[column_ii].sum() / n_data_points
    )
    assert (
        probability_correct_cond_ii >= 0 and probability_correct_cond_ii <= 1
    ), f"probability_correct_cond_ii is outside of range 0-1: {probability_correct_cond_ii}"

    probability_correct_independently = (
        probability_correct_cond_i * probability_correct_cond_ii
    )
    probability_incorrect_independently = (1 - probability_correct_cond_i) * (
        1 - probability_correct_cond_ii
    )
    probability_expected_chance_agreement = (
        probability_correct_independently + probability_incorrect_independently
    )

    assert (
        probability_expected_chance_agreement >= 0
        and probability_expected_chance_agreement <= 1
    ), f"probability_expected_chance_agreement is outside of range 0-1: {probability_expected_chance_agreement}"

    cohens_kappa = (
        probability_observed_agreement - probability_expected_chance_agreement
    ) / (1 - probability_expected_chance_agreement)
    return cohens_kappa


def get_cohens_kappa_all_batches(
    df_main_not_excluded_cond_i,
    df_main_not_excluded_cond_ii,
    cond_i,
    cond_ii,
    column_i,
    column_ii,
):
    # loop through 10 different image sets
    kappa_dict = {}
    for batch_i in range(10):
        df_condition_i_batch_i = df_main_not_excluded_cond_i[
            df_main_not_excluded_cond_i.batch == batch_i
        ].copy()
        df_condition_ii_batch_i = df_main_not_excluded_cond_ii[
            df_main_not_excluded_cond_ii.batch == batch_i
        ].copy()

        if cond_i == cond_ii and ("e_" in cond_i or "w_" in cond_i):
            kappa_cond_i = get_cohens_kappa_within_subject_conditions(
                df_condition_i_batch_i, df_condition_ii_batch_i, column_i, column_ii
            )
        elif ("w_" in cond_i or "e_" in cond_i) and (
            "w_" in cond_ii or "e_" in cond_ii
        ):
            kappa_cond_i = get_cohens_kappa_between_subject_conditions(
                df_condition_i_batch_i, df_condition_ii_batch_i, column_i, column_ii
            )
        elif "b_" in cond_i and "b_" in cond_ii:
            kappa_cond_i = get_cohens_kappa_between_baselines(
                df_condition_i_batch_i, df_condition_ii_batch_i, column_i, column_ii
            )
        elif "b_" in cond_i and ("e_" in cond_ii or "w_" in cond_ii):
            kappa_cond_i = get_cohens_kappa_between_subject_and_baseline(
                df_condition_i_batch_i, df_condition_ii_batch_i, column_i, column_ii
            )
        elif ("e_" in cond_i or "w_" in cond_i) and "b_" in cond_ii:
            # switch the order of conditions such that the first (second) one passed is the baseline (worker condition)
            kappa_cond_i = get_cohens_kappa_between_subject_and_baseline(
                df_condition_ii_batch_i, df_condition_i_batch_i, column_ii, column_i
            )
        else:
            raise ValueError("Cohen's kappa cannot be calculated")

        kappa_dict[f"batch_{batch_i}"] = kappa_cond_i

    return kappa_dict


def get_cohens_kappa_between_subject_and_baseline(
    df_condition_i_batch_i, df_condition_ii_batch_i, column_i, column_ii
):
    """When comparing between a subject condition and a baseline, it suffices to 
    compare each subject to the baseline once.
    This code assumes that condition_i is the baseline and that condition_ii is the subject condition.
    """
    kappa_cond_i = {}

    # set the one subject of the baseline that we always compare
    subject_i_of_cond_i = sorted(df_condition_i_batch_i.task_number.unique())[0]
    df_condition_i_batch_i_subject_i = df_condition_i_batch_i[
        df_condition_i_batch_i["task_number"] == subject_i_of_cond_i
    ].copy()
    df_condition_i_batch_i_subject_i.sort_values(
        by=["layer", "kernel_size"], inplace=True
    )

    kappa_cond_ii = {}
    # loop through subjects of the subject condition
    for subject_i_of_cond_ii in sorted(df_condition_ii_batch_i.task_number.unique()):
        df_condition_ii_batch_i_subject_i = df_condition_ii_batch_i[
            df_condition_ii_batch_i["task_number"] == subject_i_of_cond_ii
        ].copy()
        df_condition_ii_batch_i_subject_i.sort_values(
            by=["layer", "kernel_size"], inplace=True
        )

        assert (
            get_sum_of_equal_entries(
                df_condition_i_batch_i_subject_i,
                df_condition_ii_batch_i_subject_i,
                "layer",
                "layer",
            )
            == 18
        )
        assert (
            get_sum_of_equal_entries(
                df_condition_i_batch_i_subject_i,
                df_condition_ii_batch_i_subject_i,
                "kernel_size",
                "kernel_size",
            )
            == 18
        )

        cohens_kappa = get_cohens_kappa(
            df_condition_i_batch_i_subject_i,
            df_condition_ii_batch_i_subject_i,
            column_i,
            column_ii,
        )

        kappa_cond_ii[f"cond_ii_subject_{subject_i_of_cond_ii}"] = cohens_kappa
    kappa_cond_i[f"cond_i_subject_{subject_i_of_cond_i}"] = kappa_cond_ii

    return kappa_cond_i


def get_cohens_kappa_between_baselines(
    df_condition_i_batch_i, df_condition_ii_batch_i, column_i, column_ii
):
    """When comparing between baselines, it suffices to consider one subject in each df"""
    kappa_cond_i = {}

    # pick first subject
    subject_i_of_cond_i = sorted(df_condition_i_batch_i.task_number.unique())[0]
    df_condition_i_batch_i_subject_i = df_condition_i_batch_i[
        df_condition_i_batch_i["task_number"] == subject_i_of_cond_i
    ].copy()
    df_condition_i_batch_i_subject_i.sort_values(
        by=["layer", "kernel_size"], inplace=True
    )

    kappa_cond_ii = {}
    subject_i_of_cond_ii = sorted(df_condition_ii_batch_i.task_number.unique())[0]
    df_condition_ii_batch_i_subject_i = df_condition_ii_batch_i[
        df_condition_ii_batch_i["task_number"] == subject_i_of_cond_ii
    ].copy()
    df_condition_ii_batch_i_subject_i.sort_values(
        by=["layer", "kernel_size"], inplace=True
    )

    assert (
        get_sum_of_equal_entries(
            df_condition_i_batch_i_subject_i,
            df_condition_ii_batch_i_subject_i,
            "layer",
            "layer",
        )
        == 18
    )
    assert (
        get_sum_of_equal_entries(
            df_condition_i_batch_i_subject_i,
            df_condition_ii_batch_i_subject_i,
            "kernel_size",
            "kernel_size",
        )
        == 18
    )

    cohens_kappa = get_cohens_kappa(
        df_condition_i_batch_i_subject_i,
        df_condition_ii_batch_i_subject_i,
        column_i,
        column_ii,
    )

    kappa_cond_ii[f"cond_ii_subject_{subject_i_of_cond_ii}"] = cohens_kappa
    kappa_cond_i[f"cond_i_subject_{subject_i_of_cond_i}"] = kappa_cond_ii

    return kappa_cond_i


def get_cohens_kappa_within_subject_conditions(
    df_condition_i_batch_i, df_condition_ii_batch_i, column_i, column_ii
):
    """When comparing within the same subject condition, compare all subjects with each other once except for the subject
    with itself. I.e. compare subject 1 with subjects 2-5, subject 2 with subjects 3-5, etc. """
    kappa_cond_i = {}

    cond_i_list = sorted(df_condition_i_batch_i.task_number.unique())
    cond_ii_list = sorted(df_condition_ii_batch_i.task_number.unique())
    cond_i_list.remove(cond_i_list[-1])
    cond_ii_list.remove(cond_ii_list[0])

    # loop through subjects
    for subject_i_of_cond_i in cond_i_list:
        df_condition_i_batch_i_subject_i = df_condition_i_batch_i[
            df_condition_i_batch_i["task_number"] == subject_i_of_cond_i
        ].copy()
        df_condition_i_batch_i_subject_i.sort_values(
            by=["layer", "kernel_size"], inplace=True
        )

        kappa_cond_ii = {}
        for subject_i_of_cond_ii in cond_ii_list:
            df_condition_ii_batch_i_subject_i = df_condition_ii_batch_i[
                df_condition_ii_batch_i["task_number"] == subject_i_of_cond_ii
            ].copy()
            df_condition_ii_batch_i_subject_i.sort_values(
                by=["layer", "kernel_size"], inplace=True
            )

            assert (
                get_sum_of_equal_entries(
                    df_condition_i_batch_i_subject_i,
                    df_condition_ii_batch_i_subject_i,
                    "layer",
                    "layer",
                )
                == 18
            )
            assert (
                get_sum_of_equal_entries(
                    df_condition_i_batch_i_subject_i,
                    df_condition_ii_batch_i_subject_i,
                    "kernel_size",
                    "kernel_size",
                )
                == 18
            )

            cohens_kappa = get_cohens_kappa(
                df_condition_i_batch_i_subject_i,
                df_condition_ii_batch_i_subject_i,
                column_i,
                column_ii,
            )

            kappa_cond_ii[f"cond_ii_subject_{subject_i_of_cond_ii}"] = cohens_kappa
        cond_ii_list.remove(cond_ii_list[0])
        kappa_cond_i[f"cond_i_subject_{subject_i_of_cond_i}"] = kappa_cond_ii

    return kappa_cond_i


def get_cohens_kappa_between_subject_conditions(
    df_condition_i_batch_i, df_condition_ii_batch_i, column_i, column_ii
):
    """When comparing between two different subject conditions, compare all subjects with each other once. 
    I.e. compare subject 1 with subjects a-e, subject 2 with subjects a-e, etc. """
    kappa_cond_i = {}

    # loop through subjects
    for subject_i_of_cond_i in sorted(df_condition_i_batch_i.task_number.unique()):
        df_condition_i_batch_i_subject_i = df_condition_i_batch_i[
            df_condition_i_batch_i["task_number"] == subject_i_of_cond_i
        ].copy()
        df_condition_i_batch_i_subject_i.sort_values(
            by=["layer", "kernel_size"], inplace=True
        )

        kappa_cond_ii = {}
        for subject_i_of_cond_ii in sorted(
            df_condition_ii_batch_i.task_number.unique()
        ):
            df_condition_ii_batch_i_subject_i = df_condition_ii_batch_i[
                df_condition_ii_batch_i["task_number"] == subject_i_of_cond_ii
            ].copy()
            df_condition_ii_batch_i_subject_i.sort_values(
                by=["layer", "kernel_size"], inplace=True
            )

            assert (
                get_sum_of_equal_entries(
                    df_condition_i_batch_i_subject_i,
                    df_condition_ii_batch_i_subject_i,
                    "layer",
                    "layer",
                )
                == 18
            )
            assert (
                get_sum_of_equal_entries(
                    df_condition_i_batch_i_subject_i,
                    df_condition_ii_batch_i_subject_i,
                    "kernel_size",
                    "kernel_size",
                )
                == 18
            )

            cohens_kappa = get_cohens_kappa(
                df_condition_i_batch_i_subject_i,
                df_condition_ii_batch_i_subject_i,
                column_i,
                column_ii,
            )

            kappa_cond_ii[f"cond_ii_subject_{subject_i_of_cond_ii}"] = cohens_kappa
        kappa_cond_i[f"cond_i_subject_{subject_i_of_cond_i}"] = kappa_cond_ii

    return kappa_cond_i


def get_cohens_kappa_all_conditions_with_each_other(
    df, conditions_list, name_of_condition
):
    # check that either workers or experts are at the beginning of the conditions_list
    assert conditions_list[0].split("_")[0] in "we"

    kappa_cond_i_cond_ii_dict = {}
    for idx_i, cond_i in enumerate(conditions_list):
        print(f"{idx_i+1} / {len(conditions_list)}", end="\r")

        kappa_cond_ii_dict = {}
        for idx_ii, cond_ii in enumerate(conditions_list):

            # df for cond_i
            # pick any human condition such that a non-empty df is returned, e.g.here w_natural
            if "b_" in cond_i:
                df_main_not_excluded_cond_i = df[df[name_of_condition] == f"w_natural"]
            else:
                df_main_not_excluded_cond_i = df[df[name_of_condition] == cond_i]
            column_i = (
                "correct_" + cond_i.split("b_")[1] if "b_" in cond_i else "correct"
            )

            # df for cond_ii
            # pick any human condition such that a non-empty df is returned, e.g. here w_natural
            if "b_" in cond_ii:
                df_main_not_excluded_cond_ii = df[df[name_of_condition] == f"w_natural"]
            else:
                df_main_not_excluded_cond_ii = df[df[name_of_condition] == cond_ii]
            column_ii = (
                "correct_" + cond_ii.split("b_")[1] if "b_" in cond_ii else "correct"
            )

            kappa_dict = get_cohens_kappa_all_batches(
                df_main_not_excluded_cond_i,
                df_main_not_excluded_cond_ii,
                cond_i,
                cond_ii,
                column_i=column_i,
                column_ii=column_ii,
            )

            kappa_cond_ii_dict[cond_ii] = kappa_dict
        kappa_cond_i_cond_ii_dict[cond_i] = kappa_cond_ii_dict

    return kappa_cond_i_cond_ii_dict


def get_cohens_kappa_matrix_all_conditions_with_each_other(
    conditions_list, kappa_cond_i_cond_ii_dict
):

    cohens_kappa_matrix = np.ones((len(conditions_list), len(conditions_list)))
    cohens_kappa_std_matrix = np.ones((len(conditions_list), len(conditions_list)))
    cohens_kappa_sem_matrix = np.ones((len(conditions_list), len(conditions_list)))

    for idx_i, cond_i in enumerate(conditions_list):
        kappa_cond_i_dict = kappa_cond_i_cond_ii_dict[cond_i]
        for idx_ii, cond_ii in enumerate(conditions_list):
            kappa_dict = kappa_cond_i_dict[cond_ii]
            all_batches_list = []
            for batch_i in range(10):
                flat_list = [
                    item_i
                    for subject_i_dict in kappa_dict[f"batch_{batch_i}"].values()
                    for item_i in list(subject_i_dict.values())
                ]
                all_batches_list.append(flat_list)
            all_batches_flat_list = [
                item_i for sublist in all_batches_list for item_i in sublist
            ]
            all_batches_flat_array = np.array(all_batches_flat_list)

            cohens_kappa_matrix[idx_i, idx_ii] = all_batches_flat_array.mean()
            cohens_kappa_std_matrix[idx_i, idx_ii] = all_batches_flat_array.std()
            cohens_kappa_sem_matrix[
                idx_i, idx_ii
            ] = all_batches_flat_array.std() / np.sqrt(all_batches_flat_array.shape[0])
    return cohens_kappa_matrix, cohens_kappa_std_matrix, cohens_kappa_sem_matrix
