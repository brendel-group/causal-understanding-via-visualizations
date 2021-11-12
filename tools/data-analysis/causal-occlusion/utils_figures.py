import numpy as np
import os
import warnings
import utils_figures_helper as ut_helper

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

sharey = True
dh_01 = 0.03
dh_12 = 0.17
dh_02 = 0.33
colors = {
    "optimized": [71 / 255, 120 / 255, 158 / 255],  # syn
    "natural": [255 / 255, 172 / 255, 116 / 255],  # nat
    "blur": [114 / 255, 82 / 255, 159 / 255],  # nat blur
    "none": [172 / 255, 167 / 255, 166 / 255],  # none
    "mixed": [48 / 255, 136 / 255, 31 / 255],  # mixed
    "center": [153 / 255, 123 / 255, 102 / 255],  # center-bias
}


def setup_turbo_colormap():
    # Author: Anton Mikhailov
    turbo_colormap_data = [
        [0.18995, 0.07176, 0.23217],
        [0.19483, 0.08339, 0.26149],
        [0.19956, 0.09498, 0.29024],
        [0.20415, 0.10652, 0.31844],
        [0.20860, 0.11802, 0.34607],
        [0.21291, 0.12947, 0.37314],
        [0.21708, 0.14087, 0.39964],
        [0.22111, 0.15223, 0.42558],
        [0.22500, 0.16354, 0.45096],
        [0.22875, 0.17481, 0.47578],
        [0.23236, 0.18603, 0.50004],
        [0.23582, 0.19720, 0.52373],
        [0.23915, 0.20833, 0.54686],
        [0.24234, 0.21941, 0.56942],
        [0.24539, 0.23044, 0.59142],
        [0.24830, 0.24143, 0.61286],
        [0.25107, 0.25237, 0.63374],
        [0.25369, 0.26327, 0.65406],
        [0.25618, 0.27412, 0.67381],
        [0.25853, 0.28492, 0.69300],
        [0.26074, 0.29568, 0.71162],
        [0.26280, 0.30639, 0.72968],
        [0.26473, 0.31706, 0.74718],
        [0.26652, 0.32768, 0.76412],
        [0.26816, 0.33825, 0.78050],
        [0.26967, 0.34878, 0.79631],
        [0.27103, 0.35926, 0.81156],
        [0.27226, 0.36970, 0.82624],
        [0.27334, 0.38008, 0.84037],
        [0.27429, 0.39043, 0.85393],
        [0.27509, 0.40072, 0.86692],
        [0.27576, 0.41097, 0.87936],
        [0.27628, 0.42118, 0.89123],
        [0.27667, 0.43134, 0.90254],
        [0.27691, 0.44145, 0.91328],
        [0.27701, 0.45152, 0.92347],
        [0.27698, 0.46153, 0.93309],
        [0.27680, 0.47151, 0.94214],
        [0.27648, 0.48144, 0.95064],
        [0.27603, 0.49132, 0.95857],
        [0.27543, 0.50115, 0.96594],
        [0.27469, 0.51094, 0.97275],
        [0.27381, 0.52069, 0.97899],
        [0.27273, 0.53040, 0.98461],
        [0.27106, 0.54015, 0.98930],
        [0.26878, 0.54995, 0.99303],
        [0.26592, 0.55979, 0.99583],
        [0.26252, 0.56967, 0.99773],
        [0.25862, 0.57958, 0.99876],
        [0.25425, 0.58950, 0.99896],
        [0.24946, 0.59943, 0.99835],
        [0.24427, 0.60937, 0.99697],
        [0.23874, 0.61931, 0.99485],
        [0.23288, 0.62923, 0.99202],
        [0.22676, 0.63913, 0.98851],
        [0.22039, 0.64901, 0.98436],
        [0.21382, 0.65886, 0.97959],
        [0.20708, 0.66866, 0.97423],
        [0.20021, 0.67842, 0.96833],
        [0.19326, 0.68812, 0.96190],
        [0.18625, 0.69775, 0.95498],
        [0.17923, 0.70732, 0.94761],
        [0.17223, 0.71680, 0.93981],
        [0.16529, 0.72620, 0.93161],
        [0.15844, 0.73551, 0.92305],
        [0.15173, 0.74472, 0.91416],
        [0.14519, 0.75381, 0.90496],
        [0.13886, 0.76279, 0.89550],
        [0.13278, 0.77165, 0.88580],
        [0.12698, 0.78037, 0.87590],
        [0.12151, 0.78896, 0.86581],
        [0.11639, 0.79740, 0.85559],
        [0.11167, 0.80569, 0.84525],
        [0.10738, 0.81381, 0.83484],
        [0.10357, 0.82177, 0.82437],
        [0.10026, 0.82955, 0.81389],
        [0.09750, 0.83714, 0.80342],
        [0.09532, 0.84455, 0.79299],
        [0.09377, 0.85175, 0.78264],
        [0.09287, 0.85875, 0.77240],
        [0.09267, 0.86554, 0.76230],
        [0.09320, 0.87211, 0.75237],
        [0.09451, 0.87844, 0.74265],
        [0.09662, 0.88454, 0.73316],
        [0.09958, 0.89040, 0.72393],
        [0.10342, 0.89600, 0.71500],
        [0.10815, 0.90142, 0.70599],
        [0.11374, 0.90673, 0.69651],
        [0.12014, 0.91193, 0.68660],
        [0.12733, 0.91701, 0.67627],
        [0.13526, 0.92197, 0.66556],
        [0.14391, 0.92680, 0.65448],
        [0.15323, 0.93151, 0.64308],
        [0.16319, 0.93609, 0.63137],
        [0.17377, 0.94053, 0.61938],
        [0.18491, 0.94484, 0.60713],
        [0.19659, 0.94901, 0.59466],
        [0.20877, 0.95304, 0.58199],
        [0.22142, 0.95692, 0.56914],
        [0.23449, 0.96065, 0.55614],
        [0.24797, 0.96423, 0.54303],
        [0.26180, 0.96765, 0.52981],
        [0.27597, 0.97092, 0.51653],
        [0.29042, 0.97403, 0.50321],
        [0.30513, 0.97697, 0.48987],
        [0.32006, 0.97974, 0.47654],
        [0.33517, 0.98234, 0.46325],
        [0.35043, 0.98477, 0.45002],
        [0.36581, 0.98702, 0.43688],
        [0.38127, 0.98909, 0.42386],
        [0.39678, 0.99098, 0.41098],
        [0.41229, 0.99268, 0.39826],
        [0.42778, 0.99419, 0.38575],
        [0.44321, 0.99551, 0.37345],
        [0.45854, 0.99663, 0.36140],
        [0.47375, 0.99755, 0.34963],
        [0.48879, 0.99828, 0.33816],
        [0.50362, 0.99879, 0.32701],
        [0.51822, 0.99910, 0.31622],
        [0.53255, 0.99919, 0.30581],
        [0.54658, 0.99907, 0.29581],
        [0.56026, 0.99873, 0.28623],
        [0.57357, 0.99817, 0.27712],
        [0.58646, 0.99739, 0.26849],
        [0.59891, 0.99638, 0.26038],
        [0.61088, 0.99514, 0.25280],
        [0.62233, 0.99366, 0.24579],
        [0.63323, 0.99195, 0.23937],
        [0.64362, 0.98999, 0.23356],
        [0.65394, 0.98775, 0.22835],
        [0.66428, 0.98524, 0.22370],
        [0.67462, 0.98246, 0.21960],
        [0.68494, 0.97941, 0.21602],
        [0.69525, 0.97610, 0.21294],
        [0.70553, 0.97255, 0.21032],
        [0.71577, 0.96875, 0.20815],
        [0.72596, 0.96470, 0.20640],
        [0.73610, 0.96043, 0.20504],
        [0.74617, 0.95593, 0.20406],
        [0.75617, 0.95121, 0.20343],
        [0.76608, 0.94627, 0.20311],
        [0.77591, 0.94113, 0.20310],
        [0.78563, 0.93579, 0.20336],
        [0.79524, 0.93025, 0.20386],
        [0.80473, 0.92452, 0.20459],
        [0.81410, 0.91861, 0.20552],
        [0.82333, 0.91253, 0.20663],
        [0.83241, 0.90627, 0.20788],
        [0.84133, 0.89986, 0.20926],
        [0.85010, 0.89328, 0.21074],
        [0.85868, 0.88655, 0.21230],
        [0.86709, 0.87968, 0.21391],
        [0.87530, 0.87267, 0.21555],
        [0.88331, 0.86553, 0.21719],
        [0.89112, 0.85826, 0.21880],
        [0.89870, 0.85087, 0.22038],
        [0.90605, 0.84337, 0.22188],
        [0.91317, 0.83576, 0.22328],
        [0.92004, 0.82806, 0.22456],
        [0.92666, 0.82025, 0.22570],
        [0.93301, 0.81236, 0.22667],
        [0.93909, 0.80439, 0.22744],
        [0.94489, 0.79634, 0.22800],
        [0.95039, 0.78823, 0.22831],
        [0.95560, 0.78005, 0.22836],
        [0.96049, 0.77181, 0.22811],
        [0.96507, 0.76352, 0.22754],
        [0.96931, 0.75519, 0.22663],
        [0.97323, 0.74682, 0.22536],
        [0.97679, 0.73842, 0.22369],
        [0.98000, 0.73000, 0.22161],
        [0.98289, 0.72140, 0.21918],
        [0.98549, 0.71250, 0.21650],
        [0.98781, 0.70330, 0.21358],
        [0.98986, 0.69382, 0.21043],
        [0.99163, 0.68408, 0.20706],
        [0.99314, 0.67408, 0.20348],
        [0.99438, 0.66386, 0.19971],
        [0.99535, 0.65341, 0.19577],
        [0.99607, 0.64277, 0.19165],
        [0.99654, 0.63193, 0.18738],
        [0.99675, 0.62093, 0.18297],
        [0.99672, 0.60977, 0.17842],
        [0.99644, 0.59846, 0.17376],
        [0.99593, 0.58703, 0.16899],
        [0.99517, 0.57549, 0.16412],
        [0.99419, 0.56386, 0.15918],
        [0.99297, 0.55214, 0.15417],
        [0.99153, 0.54036, 0.14910],
        [0.98987, 0.52854, 0.14398],
        [0.98799, 0.51667, 0.13883],
        [0.98590, 0.50479, 0.13367],
        [0.98360, 0.49291, 0.12849],
        [0.98108, 0.48104, 0.12332],
        [0.97837, 0.46920, 0.11817],
        [0.97545, 0.45740, 0.11305],
        [0.97234, 0.44565, 0.10797],
        [0.96904, 0.43399, 0.10294],
        [0.96555, 0.42241, 0.09798],
        [0.96187, 0.41093, 0.09310],
        [0.95801, 0.39958, 0.08831],
        [0.95398, 0.38836, 0.08362],
        [0.94977, 0.37729, 0.07905],
        [0.94538, 0.36638, 0.07461],
        [0.94084, 0.35566, 0.07031],
        [0.93612, 0.34513, 0.06616],
        [0.93125, 0.33482, 0.06218],
        [0.92623, 0.32473, 0.05837],
        [0.92105, 0.31489, 0.05475],
        [0.91572, 0.30530, 0.05134],
        [0.91024, 0.29599, 0.04814],
        [0.90463, 0.28696, 0.04516],
        [0.89888, 0.27824, 0.04243],
        [0.89298, 0.26981, 0.03993],
        [0.88691, 0.26152, 0.03753],
        [0.88066, 0.25334, 0.03521],
        [0.87422, 0.24526, 0.03297],
        [0.86760, 0.23730, 0.03082],
        [0.86079, 0.22945, 0.02875],
        [0.85380, 0.22170, 0.02677],
        [0.84662, 0.21407, 0.02487],
        [0.83926, 0.20654, 0.02305],
        [0.83172, 0.19912, 0.02131],
        [0.82399, 0.19182, 0.01966],
        [0.81608, 0.18462, 0.01809],
        [0.80799, 0.17753, 0.01660],
        [0.79971, 0.17055, 0.01520],
        [0.79125, 0.16368, 0.01387],
        [0.78260, 0.15693, 0.01264],
        [0.77377, 0.15028, 0.01148],
        [0.76476, 0.14374, 0.01041],
        [0.75556, 0.13731, 0.00942],
        [0.74617, 0.13098, 0.00851],
        [0.73661, 0.12477, 0.00769],
        [0.72686, 0.11867, 0.00695],
        [0.71692, 0.11268, 0.00629],
        [0.70680, 0.10680, 0.00571],
        [0.69650, 0.10102, 0.00522],
        [0.68602, 0.09536, 0.00481],
        [0.67535, 0.08980, 0.00449],
        [0.66449, 0.08436, 0.00424],
        [0.65345, 0.07902, 0.00408],
        [0.64223, 0.07380, 0.00401],
        [0.63082, 0.06868, 0.00401],
        [0.61923, 0.06367, 0.00410],
        [0.60746, 0.05878, 0.00427],
        [0.59550, 0.05399, 0.00453],
        [0.58336, 0.04931, 0.00486],
        [0.57103, 0.04474, 0.00529],
        [0.55852, 0.04028, 0.00579],
        [0.54583, 0.03593, 0.00638],
        [0.53295, 0.03169, 0.00705],
        [0.51989, 0.02756, 0.00780],
        [0.50664, 0.02354, 0.00863],
        [0.49321, 0.01963, 0.00955],
        [0.47960, 0.01583, 0.01055],
    ]

    def interpolate(colormap, x):
        x = max(0.0, min(1.0, x))
        a = int(x * 255.0)
        b = min(255, a + 1)
        f = x * 255.0 - a
        return [
            colormap[a][0] + (colormap[b][0] - colormap[a][0]) * f,
            colormap[a][1] + (colormap[b][1] - colormap[a][1]) * f,
            colormap[a][2] + (colormap[b][2] - colormap[a][2]) * f,
        ]

    def interpolate_or_clip(colormap, x):
        if x < 0.0:
            return [0.0, 0.0, 0.0]
        elif x > 1.0:
            return [1.0, 1.0, 1.0]
        else:
            return interpolate(colormap, x)

    import matplotlib
    from matplotlib.colors import ListedColormap

    matplotlib.cm.register_cmap("turbo", cmap=ListedColormap(turbo_colormap_data))


def make_plot_synthetic_imgs_are_helpful(
    data_df_main_exp_main,
    results_folder="figures_ICLR_2020",
    exp_str="main_exp",
    instr_type_list=["optimized", "natural", "mixed"],
    labels=["Synthetic", "Natural", "Mixed"],
    fig_1=False,
    twitter=False,
    save_fig=True,
    show_fig=True,
    ax=None,
    ylim_min=None,
    ylim_max=None,
    extra_dict=None,
    fig_name_suffix="",
):

    instr_type_str = "mode"

    dict_acc = {}
    dict_sem = {}
    for reference_type_i in instr_type_list:
        df_factor_i = data_df_main_exp_main[
            data_df_main_exp_main[instr_type_str] == reference_type_i
        ].copy()

        if len(df_factor_i) == 0:
            dict_acc[reference_type_i] = np.nan
            dict_sem[reference_type_i] = np.nan
        else:
            accuracy = (df_factor_i["correct"] == True).sum() / df_factor_i.shape[0]
            dict_acc[reference_type_i] = accuracy
            # evaluate standard error of the mean
            dict_acc_subject_i = {}
            for subject_i in df_factor_i["task_response_id"].unique():
                df_factor_i_subject = df_factor_i[
                    df_factor_i["task_response_id"] == subject_i
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
    # add additional baselines
    if extra_dict is not None:
        for key_i, dict_i in extra_dict.items():
            dict_acc[key_i] = dict_i["accuracy"]
            dict_sem[key_i] = dict_i["sem"]

    ##figure
    if ax is None:
        if fig_1:
            fig, ax = plt.subplots(1, 1, figsize=(2, 2.3))
        else:
            fig, ax = plt.subplots(1, 1, figsize=(2.2, 3))

    color_list = [colors[it] for it in instr_type_list]
    if extra_dict is not None:
        for key_i in extra_dict.keys():
            color_list.append(colors[key_i])

    # 2 SEM
    dict_2sem = {x: y * 2 for (x, y) in dict_sem.items()}
    rect = ax.bar(
        range(len(dict_acc)),
        list(dict_acc.values()),
        color=color_list,
        yerr=dict_2sem.values(),
        error_kw=dict(lw=error_bar_linewidth),
        align="center",
    )
    if not fig_1:
        ut_helper.autolabel(rect, ax, dict_2sem)
    ax.axhline(0.5, color="k", linestyle="--", linewidth=1, label="Chance")

    # significance
    if fig_1 == False and twitter == False:
        height_bar_and_error_list = []
        """
        for rect_i, error_value_i in zip(rect, dict_2sem.values()):
            height = rect_i.get_height()
            height_bar_and_error_list.append(height + error_value_i)
        if exp_str == "main_exp":
            ut_helper.barplot_annotate_brackets(ax, 0, 1, "p $=$ 0.015", np.arange(3), height_bar_and_error_list, dh=dh_01) 
            ut_helper.barplot_annotate_brackets(ax, 1, 2, "p $<$ 0.001", np.arange(3), height_bar_and_error_list, dh=dh_12)
            ut_helper.barplot_annotate_brackets(ax, 0, 2, "p $<$ 0.001", np.arange(3), height_bar_and_error_list, dh=dh_02)
        elif exp_str == "ablation_exp":
            ut_helper.barplot_annotate_brackets(ax, 0, 1, "p $<$ 0.001", np.arange(3), height_bar_and_error_list)
        else: #exp_str == "rebuttal_exp" or "pilot_ks_0_exp"
            pass #TODO
        """

    ax.set_ylabel("Proportion Correct", fontsize=fontsize_axes_labels)
    if fig_1:
        ax.set_ylim(0.5, 0.75)
    else:
        if ylim_min is not None:
            ax.set_ylim(ylim_min, ylim_max)
        yticks_and_labels = [0, 0.2, 0.4, 0.6, 0.8, 1]
        ax.set_yticks(yticks_and_labels)
        ax.set_yticklabels(yticks_and_labels)

    # xticks
    ax.set_xticks(range(len(dict_acc)))
    ax.set_xticklabels(labels[: len(dict_acc)], rotation=25)

    ax.tick_params(labelsize=fontsize_tick_labels)

    # no axis on top and right
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if fig_1 == False and twitter == False:
        ax.set_xlabel("Reference Images", fontsize=fontsize_axes_labels)

    plot_name = "accuracy_averaged_all_per_condition"
    if fig_1:
        plot_name += "_fig_1"
    if twitter:
        plot_name += "twitter"
    for version in range(100):
        file_name = os.path.join(
            results_folder,
            f"{plot_name}{fig_name_suffix}_{exp_str}_{version}{file_type}",
        )
        # if file_name does not yet exist, use it
        if not os.path.exists(file_name):
            break
    if save_fig:
        print(f"figure saved under {file_name}")
        plt.savefig(file_name, bbox_inches="tight")

    if show_fig:
        plt.show()
    else:
        return ax


def make_plot_workers_understood_task(
    data_df_main_exp_main,
    results_folder="figures_ICLR_2020",
    exp_str="main_exp",
    instr_type_list=["optimized", "natural", "mixed"],
    labels=["Synthetic", "Natural", "Mixed"],
    include_experts=True,
    fig_1=False,
    twitter=False,
    save_fig=True,
    show_fig=True,
    ax=None,
    ylim_min=None,
    ylim_max=None,
    extra_dict=None,
):

    instr_type_str = "mode"

    dict_acc = []
    dict_sem = []

    for reference_type_i in instr_type_list:
        for expert_baseline_value in (False, True) if include_experts else (False,):
            df_factor_i = data_df_main_exp_main[
                data_df_main_exp_main[instr_type_str] == reference_type_i
            ].copy()
            df_factor_i = df_factor_i[
                df_factor_i["expert_baseline"] == expert_baseline_value
            ].copy()

            if len(df_factor_i) == 0:
                dict_acc.append(np.nan)
                dict_sem.append(np.nan)
            else:
                accuracy = (df_factor_i["correct"] == True).sum() / df_factor_i.shape[0]
                dict_acc.append(accuracy)
                # evaluate standard error of the mean
                dict_acc_subject_i = {}
                for subject_i in df_factor_i["task_response_id"].unique():
                    df_factor_i_subject = df_factor_i[
                        df_factor_i["task_response_id"] == subject_i
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
                dict_sem.append(sem)

    ##figure
    if ax is None:
        if fig_1:
            fig, ax = plt.subplots(1, 1, figsize=(2, 2.3))
        else:
            fig, ax = plt.subplots(1, 1, figsize=(3.8, 3))

    if include_experts:
        color_list = [[colors[it]] * 2 for it in instr_type_list]
        color_list = [it for lst in color_list for it in lst]
    else:
        color_list = [colors[it] for it in instr_type_list]

    # 2 SEM
    dict_2sem = [y * 2 for y in dict_sem]

    bar_width = 1.0
    bar_gap = 0.8

    x_values = bar_width * np.arange(len(dict_acc))
    x_gap_values = np.ones_like(x_values) * bar_gap
    if include_experts:
        x_gap_values[1::2] = 0
    x_gap_values = np.cumsum(x_gap_values)
    x_values += x_gap_values
    y_values = dict_acc
    y_err = dict_2sem
    alpha_list = [1.0, 0.5]
    stepsize = 2 if include_experts else 1
    for start_idx in (0, 1) if include_experts else (0,):
        rect = ax.bar(
            x_values[start_idx::stepsize],
            y_values[start_idx::stepsize],
            alpha=alpha_list[start_idx],
            color=color_list[start_idx::stepsize],
            yerr=y_err[start_idx::stepsize],
            error_kw=dict(lw=error_bar_linewidth),
            align="center",
            width=bar_width,
        )

        ut_helper.autolabel(rect, ax, y_err[start_idx::stepsize])
    ax.axhline(0.5, color="k", linestyle="--", linewidth=1, label="Chance")

    # significance
    if fig_1 == False and twitter == False:
        height_bar_and_error_list = []
        """
        for rect_i, error_value_i in zip(rect, dict_2sem.values()):
            height = rect_i.get_height()
            height_bar_and_error_list.append(height + error_value_i)
        if exp_str == "main_exp":
            ut_helper.barplot_annotate_brackets(ax, 0, 1, "p $=$ 0.015", np.arange(3), height_bar_and_error_list, dh=dh_01) 
            ut_helper.barplot_annotate_brackets(ax, 1, 2, "p $<$ 0.001", np.arange(3), height_bar_and_error_list, dh=dh_12)
            ut_helper.barplot_annotate_brackets(ax, 0, 2, "p $<$ 0.001", np.arange(3), height_bar_and_error_list, dh=dh_02)
        elif exp_str == "ablation_exp":
            ut_helper.barplot_annotate_brackets(ax, 0, 1, "p $<$ 0.001", np.arange(3), height_bar_and_error_list)
        else: #exp_str == "rebuttal_exp" or "pilot_ks_0_exp"
            pass #TODO
        """

    ax.set_ylabel("Proportion Correct", fontsize=fontsize_axes_labels)
    if fig_1:
        ax.set_ylim(0.5, 0.70)
    else:
        if ylim_min is not None:
            ax.set_ylim(ylim_min, ylim_max)
        yticks_and_labels = [0, 0.2, 0.4, 0.6, 0.8, 1]
        ax.set_yticks(yticks_and_labels)
        ax.set_yticklabels(yticks_and_labels)

    # xticks
    if include_experts:
        ax.set_xticks((x_values[::2] + x_values[1::2]) / 2)
    else:
        ax.set_xticks(x_values)

    ax.set_xticklabels(labels, rotation=0)

    ax.tick_params(labelsize=fontsize_tick_labels)

    # no axis on top and right
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if fig_1 == False and twitter == False:
        ax.set_xlabel("Reference Images", fontsize=fontsize_axes_labels)

    plot_name = "accuracy_averaged_all_per_condition_with_experts"
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

    if show_fig:
        plt.show()
    else:
        return ax


def make_plot_natural_are_better_wrt_confidence(
    data_df_main_exp_main,
    results_folder="figures_ICLR_2020",
    exp_str="main_exp",
    instr_type_list=["optimized", "natural", "mixed"],
    labels=["Synthetic", "Natural", "Mixed"],
    conditioned_on=None,  # None, "correctness", "falseness"
    save_fig=True,
):

    y_axis_label = "Proportion of Trials"

    conditioned_str = "_"

    assert conditioned_on is None or conditioned_on in ["correctness", "falseness"]
    if conditioned_on == "correctness":
        data_df_main_exp_main = data_df_main_exp_main[
            data_df_main_exp_main["correct"] == True
        ].copy()
        conditioned_str = "_conditioned_on_correctness_"
        y_axis_label = "Proportion of Correct Trials"
    elif conditioned_on == "falseness":
        data_df_main_exp_main = data_df_main_exp_main[
            data_df_main_exp_main["correct"] == False
        ].copy()
        conditioned_str = "_conditioned_on_falseness_"
        y_axis_label = "Proportion of Incorrect Trials"

    fig_size_width_factor = 1
    instr_type_str = "mode"

    data = {}
    dict_sem = {}
    for reference_type_i in instr_type_list:
        denominator = data_df_main_exp_main[
            data_df_main_exp_main[instr_type_str] == reference_type_i
        ].shape[0]

        reference_type_i_data = {}
        reference_type_i_dict_sem = {}
        for abs_conf in data_df_main_exp_main["confidence"].unique():
            reference_type_i_data[abs_conf] = (
                sum(
                    (data_df_main_exp_main["confidence"] == abs_conf)
                    & (data_df_main_exp_main[instr_type_str] == reference_type_i)
                )
                / denominator
            )

            # evaluate standard error of the mean
            dict_conf_rating_subject_i = {}
            for subject_i in data_df_main_exp_main["task_response_id"].unique():
                df_factor_i_subject = data_df_main_exp_main[
                    data_df_main_exp_main["task_response_id"] == subject_i
                ].copy()
                denominator_subject_i = df_factor_i_subject[
                    df_factor_i_subject[instr_type_str] == reference_type_i
                ].shape[0]
                numerator_subject_i = sum(
                    (df_factor_i_subject["confidence"] == abs_conf)
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
            color=colors[factor_i],
            yerr=dict_2sem.values(),
            error_kw=dict(lw=error_bar_linewidth),
        )
        axes[i].set_xlabel(labels[i], fontsize=fontsize_tick_labels)
        # no axis on top and right
        axes[i].spines["top"].set_visible(False)
        axes[i].spines["right"].set_visible(False)
        axes[i].tick_params(labelsize=fontsize_tick_labels)
        if exp_str == "main_exp":
            axes[i].set_ylim(0, 0.59)
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

    plot_name = "confidence_averaged_all_per_condition"
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
    instr_type_list=["optimized", "natural", "mixed"],
    labels=["Synthetic", "Natural", "Mixed"],
    conditioned_on=None,  # None, "correctness", "falseness"
    save_fig=True,
):

    time_divider = 1000

    y_axis_label = "Reaction Time [sec]"

    conditioned_str = "_"

    assert conditioned_on is None or conditioned_on in ["correctness", "falseness"]
    if conditioned_on == "correctness":
        data_df_main_exp_main = data_df_main_exp_main[
            data_df_main_exp_main["correct"] == True
        ].copy()
        conditioned_str = "_conditioned_on_correctness_"
        y_axis_label = "Reaction Time [sec] of Correct Trials"
    elif conditioned_on == "falseness":
        data_df_main_exp_main = data_df_main_exp_main[
            data_df_main_exp_main["correct"] == False
        ].copy()
        conditioned_str = "_conditioned_on_falseness_"
        y_axis_label = "Reaction Time [sec] of Incorrect Trials"

    fig_size_width_factor = 1
    instr_type_str = "mode"

    dict_rt = {}
    dict_rt_sem = {}
    for reference_type_i in instr_type_list:
        df_factor_i = data_df_main_exp_main[
            data_df_main_exp_main[instr_type_str] == reference_type_i
        ].copy()
        df_factor_i["rt"] /= time_divider
        rt_mean = df_factor_i["rt"].mean()
        dict_rt[reference_type_i] = rt_mean
        # evaluate standard error of the mean
        dict_rt_subject_i = {}
        for subject_i in df_factor_i["task_response_id"].unique():
            df_factor_i_subject = df_factor_i[
                df_factor_i["task_response_id"] == subject_i
            ].copy()
            rt_mean_subject_i = df_factor_i_subject["rt"].mean()
            dict_rt_subject_i[subject_i] = rt_mean_subject_i
        n_data_points = len(dict_rt_subject_i)
        standard_deviation = np.std(
            np.fromiter(dict_rt_subject_i.values(), dtype=float)
        )
        sem = standard_deviation / np.sqrt(n_data_points)
        dict_rt_sem[reference_type_i] = sem

    ### figure
    fig, ax = plt.subplots(1, 1, figsize=(3.5 * fig_size_width_factor, 3))

    # 2 SEM
    dict_2sem = {x: y * 2 for (x, y) in dict_rt_sem.items()}
    bar_gap = 0.25
    rect = ax.bar(
        (1 + bar_gap) * np.arange(len(dict_rt)),
        list(dict_rt.values()),
        color=[colors[it] for it in instr_type_list],
        yerr=dict_2sem.values(),
        error_kw=dict(lw=error_bar_linewidth),
        align="center",
        label="All subjects",
    )
    ut_helper.autolabel(rect, ax, dict_2sem, fontsize=9)

    ax.set_ylabel(y_axis_label, fontsize=fontsize_axes_labels)

    # xticks
    ax.set_xticks((1 + bar_gap) * np.arange(len(dict_rt)))
    ax.set_xticklabels(labels)
    ax.tick_params(labelsize=fontsize_tick_labels)

    # no axis on top and right
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlabel("Reference Images", fontsize=fontsize_axes_labels)

    plot_name = "reaction_time_averaged_all_per_condition"
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


def make_plot_branch_comparison(
    data_df_main_exp_main,
    results_folder="figures_ICLR_2020",
    exp_str="main_exp",
    instr_type_list=["optimized", "natural", "mixed"],
    labels=["Synthetic", "Natural", "Mixed"],
    branches_labels=["1x1", "3x3", "5x5", "pool"],
    branches=["0", "1", "2", "3"],
    cropped_range=True,
    save_fig=True,
):
    fig_size_width_factor = len(instr_type_list) * 0.35
    instr_type_str = "mode"

    dict_acc = {}
    dict_sem = {}
    for reference_type_i in instr_type_list:
        df_reference_type_i = data_df_main_exp_main[
            data_df_main_exp_main[instr_type_str] == reference_type_i
        ].copy()
        dict_acc_kernel_size_number_i = {}
        dict_sem_kernel_size_number_i = {}
        for kernel_size_number_i in branches:
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
                "task_response_id"
            ].unique():
                df_reference_type_i_kernel_size_number_i_subject_i = df_reference_type_i_kernel_size_number_i[
                    df_reference_type_i_kernel_size_number_i["task_response_id"]
                    == subject_i
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
        dict_2sem = {x: y * 2 for (x, y) in dict_sem[factor_i].items()}
        rect = axes[i].bar(
            np.arange(len(factor_i_data)),
            list(factor_i_data.values()),
            yerr=dict_2sem.values(),
            error_kw=dict(lw=error_bar_linewidth),
            color=[colors[factor_i]] * len(branches),
            label=labels[i],
        )

        # ut_helper.autolabel(rect, axes, dict_2sem)
        axes[i].set_xticks(np.arange(len(factor_i_data)))
        axes[i].set_xticklabels(branches_labels)
        axes[i].set_xlabel(labels[i], fontsize=fontsize_tick_labels)

        # no axis on top and right
        axes[i].spines["top"].set_visible(False)
        axes[i].spines["right"].set_visible(False)
        axes[i].tick_params(labelsize=fontsize_tick_labels)

        if cropped_range:
            axes[i].set_ylim(0.5, 1)
        else:
            axes[i].set_ylim(0.0, 1)
            axes[i].axhline(0.5, color="k", linestyle="--", linewidth=1, label="Chance")

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

    plot_name = "comparison_across_branches"
    if cropped_range:
        plot_name += "_cropped"

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


def make_plot_branch_comparison_alternating(
    data_df_main_exp_main,
    results_folder="figures_ICLR_2020",
    exp_str="main_exp",
    instr_type_list=["optimized", "natural", "mixed"],
    branches_labels=["1x1", "3x3", "5x5", "pool"],
    branches=["0", "1", "2", "3"],
    cropped_range=True,
    save_fig=True,
):
    instr_type_str = "mode"
    fig_size_width_factor = len(instr_type_list) * 0.35

    dict_acc = {}
    dict_sem = {}
    for reference_type_i in instr_type_list:
        df_reference_type_i = data_df_main_exp_main[
            data_df_main_exp_main[instr_type_str] == reference_type_i
        ].copy()
        dict_acc_kernel_size_number_i = {}
        dict_sem_kernel_size_number_i = {}
        for kernel_size_number_i in branches:
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
                "task_response_id"
            ].unique():
                df_reference_type_i_kernel_size_number_i_subject_i = df_reference_type_i_kernel_size_number_i[
                    df_reference_type_i_kernel_size_number_i["task_response_id"]
                    == subject_i
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

    if len(instr_type_list) == 2:
        offsets = (-0.2, 0.2)
        bar_width = 0.4
        x_factor = 1.3
    elif len(instr_type_list) == 3:
        offsets = (-0.3, 0, 0.3)
        bar_width = 0.3
        x_factor = 1.2
    elif len(instr_type_list) == 4:
        offsets = (-0.5625, -0.1875, 0.1875, 0.5625)
        bar_width = 0.3
        x_factor = 2
    elif len(instr_type_list) == 5:
        offsets = (-0.6, -0.3, 0, 0.3, 0.6)
        bar_width = 0.3
        x_factor = 2

    for i, (factor_i, factor_i_data) in enumerate(dict_acc.items()):
        x = np.arange(len(factor_i_data)) * x_factor
        x = x + offsets[i]

        dict_2sem = {x: y * 2 for (x, y) in dict_sem[factor_i].items()}
        rect = ax.bar(
            x,
            list(factor_i_data.values()),
            width=bar_width,
            yerr=dict_2sem.values(),
            error_kw=dict(lw=error_bar_linewidth),
            color=[colors[factor_i]] * len(branches),
        )
        #         ut_helper.autolabel(rect, axes, dict_2sem)
        ax.set_xticks(np.arange(len(factor_i_data)) * x_factor)
        ax.set_xticklabels(branches_labels)

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

    plot_name = "comparison_across_branches_alternating"
    if cropped_range:
        plot_name += "_cropped"

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


def plot_baseline_accuracy(
    accuracies,
    label_order,
    sems=None,
    results_folder="figures_ICLR_2020",
    save_fig=False,
    show_fig=True,
):
    if sems is None:
        sems = {}

    two_sems = [np.nan if l not in sems else 2 * sems[l] for l in label_order]

    ##figure
    fig, ax = plt.subplots(1, 1, figsize=(2.5, 3))

    bar_gap = 0.25
    rect = ax.bar(
        (1.0 + bar_gap) * np.arange(len(label_order)),
        [accuracies[l] for l in label_order],
        color="black",
        yerr=two_sems,
        error_kw=dict(lw=error_bar_linewidth),
        align="center",
    )
    ut_helper.autolabel(rect, ax, None, color="white")
    ax.axhline(0.5, color="k", linestyle="--", linewidth=1, label="Chance")

    ax.set_ylabel("Proportion Correct", fontsize=fontsize_axes_labels)

    ax.set_ylim(0, 1)
    yticks_and_labels = [0, 0.2, 0.4, 0.6, 0.8, 1]
    ax.set_yticks(yticks_and_labels)
    ax.set_yticklabels(yticks_and_labels)

    # xticks
    ax.set_xticks((1.0 + bar_gap) * np.arange(len(label_order)))
    ax.set_xticklabels(label_order, rotation=0)

    ax.tick_params(labelsize=fontsize_tick_labels)

    # no axis on top and right
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlabel("Baselines", fontsize=fontsize_axes_labels)

    plot_name = "accuracy_baselines"
    for version in range(100):
        file_name = os.path.join(results_folder, f"{plot_name}_{version}{file_type}")
        # if file_name does not yet exist, use it
        if not os.path.exists(file_name):
            break
    if save_fig:
        print(f"figure saved under {file_name}")
        plt.savefig(file_name, bbox_inches="tight")

    if show_fig:
        plt.show()
    else:
        return ax
