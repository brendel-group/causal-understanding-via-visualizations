from glob import glob
import os
import json
import pandas as pd
import pickle
import numpy as np


def load_results(data_folder):
    """Load experiment results as pickled RepeatedTaskResult object"""
    result_fns = glob(os.path.join(data_folder, "result_task_*.pkl"))

    all_results = []
    for result_fn in result_fns:
        with open(result_fn, "rb") as f:
            result = pickle.load(f)
        if len(result) == 1:
            all_results += result
        else:
            all_results.append(result)

    return all_results


def parse_results(tasks, mode):
    """Convert list of RepeatedTaskResult objects to pandas dataframe"""
    dfs = []
    for i_task, task_data in enumerate(tasks):
        dfs_per_task = []

        for i, response_data in enumerate(task_data.responses):
            response_df = pd.DataFrame(response_data["main_data"])
            response_df["response_index"] = i
            response_df["worker_id"] = task_data[1][i][
                "worker_id"
            ]  # reading out from raw_responses
            response_df.loc[response_df["is_demo"] == True, "trial_index"] = np.arange(
                -len(response_df[response_df["is_demo"] == True]), 0
            )
            response_df.loc[response_df["is_demo"] == False, "trial_index"] = np.arange(
                len(response_df[response_df["is_demo"] == False])
            )
            dfs_per_task.append(response_df)

        task_df = pd.concat(dfs_per_task, 0)
        task_df["task_number"] = int(task_data.task_id.split("-")[-1])
        task_df["task_id"] = task_data.task_id
        dfs.append(task_df)

    if len(dfs) > 0:
        df = pd.concat(dfs, 0)

        df["mode"] = mode
        df = df.reset_index().drop("index", axis=1)

        df["choice_number"] = df["choice"].map(lambda x: -1 if x == "a" else 1)

        return df
    return None


def parse_feedback(tasks, mode):
    dfs = []
    for i_task, task_data in enumerate(tasks):
        for i, response_data in enumerate(task_data.responses):
            surveys = [
                it
                for it in response_data["raw_data"]
                if it["trial_type"] == "survey-text"
            ]
            feedback = "\n".join(
                [json.loads(it["responses"])["feedback"] for it in surveys]
            )
            dfs.append(
                pd.DataFrame(
                    {
                        "response_index": i,
                        "task_number": int(task_data.task_id.split("-")[-1]),
                        "task_id": task_data.task_id,
                        "feedback": feedback,
                        "mode": mode,
                    },
                    index=[len(dfs)],
                )
            )

    if len(dfs) > 0:
        df = pd.concat(dfs, 0)

        df = df.reset_index().drop("index", axis=1)

        return df
    return None


def parse_check_results(tasks, mode="natural"):
    df = []
    for task in tasks:
        for response_idx, response in enumerate(task.raw_responses):
            check_results = response["check_results"]
            df.append(
                {
                    "task_id": task.task_id,
                    "response_index": response_idx,
                    "passed_checks": response["passed_checks"],
                    **{f"{k}_result": check_results[k][0] for k in check_results},
                    **{f"{k}_details": check_results[k][1] for k in check_results},
                }
            )
    df = pd.DataFrame(df)
    df["mode"] = mode

    return df


def load_and_parse_all_results(base_folder, instr_type_list):
    results = {
        it: load_results(os.path.join(base_folder, it)) for it in instr_type_list
    }

    df_checks = [parse_check_results(results[it], it) for it in instr_type_list]
    df_checks = pd.concat(df_checks).reset_index().drop("index", axis=1)

    df_feedback = [parse_feedback(results[it], it) for it in instr_type_list]
    df_feedback = pd.concat(df_feedback).reset_index().drop("index", axis=1)

    dfs = [parse_results(results[it], it) for it in instr_type_list]
    dfs = [it for it in dfs if it is not None]
    df = pd.concat(dfs).reset_index().drop("index", axis=1)

    return df, df_checks, df_feedback


def load_and_parse_trial_structure(folder, structure_names):
    def parse_trials_structure(trials):
        results = []
        for trial in trials:
            query_path = trial["queries"]
            parts = query_path.split("/")
            batch = int(parts[-2].split("_")[-1])
            channel = parts[-4].split("_")[-1]
            kernel_size = parts[-5].split("_")[-1]
            layer = parts[-6].split("_")[-1]

            value = dict(
                batch=batch,
                channel=channel,
                kernel_size=kernel_size,
                layer=layer,
                mode=trial["mode"],
            )
            if "base_query_activation" in trial:
                for k in (
                    "base_query_activation",
                    "min_query_activation",
                    "max_query_activation",
                ):
                    value[k] = trial[k]
            if "min_query_center_distance" in trial:
                value["min_query_center_distance"] = trial["min_query_center_distance"]
                value["max_query_center_distance"] = trial["max_query_center_distance"]
            if "min_query_patch_std" in trial:
                value["min_query_patch_std"] = trial["min_query_patch_std"]
                value["max_query_patch_std"] = trial["max_query_patch_std"]
            if "min_query_patch_saliency" in trial:
                value["min_query_patch_saliency"] = trial["min_query_patch_saliency"]
                value["max_query_patch_saliency"] = trial["max_query_patch_saliency"]
            results.append(value)
        return results

    def process_structure_file(filename):
        with open(os.path.join(folder, filename), "r") as f:
            raw_structure = json.load(f)

        structure = {}
        for item in raw_structure["tasks"]:
            structure[item["index"]] = {
                k: parse_trials_structure(item[k]) for k in item if k != "index"
            }

        return structure

    return [process_structure_file(it) for it in structure_names]


def append_trial_structure_to_results(df, structures):
    df = df.copy(deep=True)

    # merge structure with df
    batch_column = []
    channel_column = []
    kernel_size_column = []
    layer_column = []

    base_query_activation_column = []
    min_query_activation_column = []
    max_query_activation_column = []

    min_query_center_distance_column = []
    max_query_center_distance_column = []

    min_query_patch_std_column = []
    max_query_patch_std_column = []

    min_query_patch_saliency_column = []
    max_query_patch_saliency_column = []
    for i in range(len(df)):
        task_number = df.task_number[i]
        trial_number = df.trial_index[i]

        if df["mode"][i] in structures:
            structure = structures[df["mode"][i]]
        else:
            raise ValueError(f"unknown mode: {df.mode[i]}")

        info = structure[task_number]["trials"][trial_number]
        batch_column.append(info["batch"])
        channel_column.append(info["channel"])
        kernel_size_column.append(info["kernel_size"])
        layer_column.append(info["layer"])

        if "base_query_activation" in info:
            base_query_activation_column.append(float(info["base_query_activation"]))
            min_query_activation_column.append(float(info["min_query_activation"]))
            max_query_activation_column.append(float(info["max_query_activation"]))
        if "min_query_center_distance" in info:
            min_query_center_distance_column.append(
                float(info["min_query_center_distance"])
            )
            max_query_center_distance_column.append(
                float(info["max_query_center_distance"])
            )
        if "min_query_patch_std" in info:
            min_query_patch_std_column.append(float(info["min_query_patch_std"]))
            max_query_patch_std_column.append(float(info["max_query_patch_std"]))
        if "min_query_patch_saliency" in info:
            min_query_patch_saliency_column.append(
                float(info["min_query_patch_saliency"])
            )
            max_query_patch_saliency_column.append(
                float(info["max_query_patch_saliency"])
            )

    df["batch"] = batch_column
    df["channel"] = channel_column
    df["kernel_size"] = kernel_size_column
    df["layer"] = layer_column

    if len(base_query_activation_column) > 0:
        df["base_query_activation"] = base_query_activation_column
        df["min_query_activation"] = min_query_activation_column
        df["max_query_activation"] = max_query_activation_column

    if len(min_query_center_distance_column) > 0:
        df["min_query_center_distance"] = min_query_center_distance_column
        df["max_query_center_distance"] = max_query_center_distance_column

    if len(min_query_patch_std_column) > 0:
        df["min_query_patch_std"] = min_query_patch_std_column
        df["max_query_patch_std"] = max_query_patch_std_column

    if len(min_query_patch_saliency_column) > 0:
        df["min_query_patch_saliency"] = min_query_patch_saliency_column
        df["max_query_patch_saliency"] = max_query_patch_saliency_column

    return df


def add_task_response_id(df, df_checks):
    """Create a unique column based on task id and response id (unique within each task)"""
    df["task_response_id"] = df.apply(
        lambda row: row["task_id"] + "_" + str(row["response_index"]), axis=1
    )
    df_checks["task_response_id"] = df_checks.apply(
        lambda row: row["task_id"] + "_" + str(row["response_index"]), axis=1
    )

    return df, df_checks


def checks_add_demo_trial_repetitions(df_demo_trials, df_checks):
    """Calculate how often the demo trials had to be repeated. Since the demo trials get a negative
    trial index (from -#demo trials to -1) we can take the minimum trial index per response and divide
    this number by the number of demo trials per demo trial block (i.e., 4) to get the number of repetitions."""

    # for all tasks that contained demo trials, get the max number of demo trials within the task
    df_temp = (
        df_demo_trials[["task_response_id", "trial_index"]]
        .groupby("task_response_id", as_index=False)
        .min()
    )

    # now either return the calculated number or return 0 if no demo trial is contaiend in the task
    def get_demo_trial_repetitions(row):
        selected_df = df_temp[df_temp["task_response_id"] == row["task_response_id"]]

        if len(selected_df) == 0:
            return 0
        else:
            return selected_df.iloc[0]["trial_index"] / (-4)

    df_checks["demo_trials_repetitions"] = df_checks.apply(
        get_demo_trial_repetitions, axis=1
    )

    return df_checks


def _get_map_excluded_responses(df_checks, column_name="passed_checks"):
    def map_excluded_responses(row):
        rows = df_checks[
            (df_checks["task_id"] == row["task_id"])
            & (df_checks["response_index"] == row["response_index"])
        ]
        result = not rows[column_name].item()
        return result

    return map_excluded_responses


def process_checks(df, df_checks):
    df["excluded_response"] = df.apply(
        _get_map_excluded_responses(df_checks, "passed_checks"), axis=1
    )

    # simplify existing columns
    df_checks["instruction_time_details_extracted"] = df_checks.apply(
        lambda row: row["instruction_time_details"]["total_time"], axis=1
    )

    df_checks["total_response_time_details_extracted"] = df_checks.apply(
        lambda row: row["total_response_time_details"]["total_time"], axis=1
    )

    df_checks["row_variability_details_details_upper_exctracted"] = df_checks.apply(
        lambda row: row["row_variability_details"]["n_upper_row"], axis=1
    )

    df_checks["row_variability_details_details_lower_exctracted"] = df_checks.apply(
        lambda row: row["row_variability_details"]["n_lower_row"], axis=1
    )

    df_checks["catch_trials_details_ratio_exctracted"] = df_checks.apply(
        lambda row: row["catch_trials_details"]["ratio"], axis=1
    )

    df_checks["catch_trials_details_correctly_answered_exctracted"] = df_checks.apply(
        lambda row: row["catch_trials_details"]["correctly_answered"], axis=1
    )
    df_checks["passed_checks"] = df_checks["passed_checks"].astype(bool)

    df_checks["passed_checks_except_catch"] = (
        (df_checks["instruction_time_details_extracted"] >= 15)
        & (df_checks["total_response_time_details_extracted"] <= 900)
        & (df_checks["total_response_time_details_extracted"] >= 90)
        & (df_checks["row_variability_details_details_upper_exctracted"] >= 1)
    )

    return df, df_checks


def get_catch_trials_as_main_data(df_catch_trials, df_checks):
    df_catch_trials = df_catch_trials.copy()
    df_checks = df_checks.copy()
    df_catch_trials["excluded_response_ignoring_catch_trials"] = df_catch_trials.apply(
        _get_map_excluded_responses(df_checks, "passed_checks_except_catch"), axis=1
    )
    df_catch_trials_not_excluded_ignoring_catch_trials = df_catch_trials[
        ~df_catch_trials["excluded_response_ignoring_catch_trials"]
    ]

    return df_catch_trials_not_excluded_ignoring_catch_trials


def append_hit_id_and_times_return_all_assignments(
    df, mturk_csv_datetime="20210601_090251"
):
    assignments_df = pd.read_csv(f"data/mturk_{mturk_csv_datetime}_assignments.csv")
    hits_df = pd.read_csv(f"data/mturk_{mturk_csv_datetime}_hits.csv")

    assignments_df["HITRequesterAnnotation"] = assignments_df.apply(
        axis=1,
        func=lambda row: hits_df[hits_df["HITId"] == row["HITId"]].iloc[0][
            "RequesterAnnotation"
        ],
    )

    def get_hit_id_and_times(row):
        worker_id = row["worker_id"]
        task_id = row["task_id"]

        # the requester annotation has the format "task namespace-experiment-task" and we need only the "experiment-task" part of it
        task_ids = assignments_df["HITRequesterAnnotation"].map(
            lambda x: "-".join(x.split("-")[1:])
        )
        selected_assignments = assignments_df[
            (assignments_df["WorkerId"] == worker_id) & (task_ids == task_id)
        ]

        # there should only be exactly 1 assignment in the data from mturk that matches the criteria
        assert len(selected_assignments) == 1, len(selected_assignments)
        return (
            selected_assignments.iloc[0]["HITId"],
            selected_assignments.iloc[0]["SubmitTime"],
            selected_assignments.iloc[0]["AcceptTime"],
            selected_assignments.iloc[0]["ApprovalTime"],
        )

    # insert additional columns
    (
        df["hit_id"],
        df["assignment_submit_time"],
        df["assignment_accept_time"],
        df["assignment_approval_time"],
    ) = zip(*df.apply(axis=1, func=get_hit_id_and_times))

    # convert from object to proper datetime column
    df["assignment_submit_time"] = pd.to_datetime(df["assignment_submit_time"])
    df["assignment_accept_time"] = pd.to_datetime(df["assignment_accept_time"])
    df["assignment_approval_time"] = pd.to_datetime(df["assignment_approval_time"])

    assignments_df["SubmitTime"] = pd.to_datetime(assignments_df["SubmitTime"])
    assignments_df["AcceptTime"] = pd.to_datetime(assignments_df["AcceptTime"])
    assignments_df["ApprovalTime"] = pd.to_datetime(assignments_df["ApprovalTime"])

    return df, assignments_df
