"""Repeatedly post HITs on MTurk until one accepted answer to each task has been collected."""
import glob

from mturk import HITSpawner
import argparse
import logging
import requests
from urllib.parse import urljoin
import os
import pickle
import sys
import time


def parse_args():
    """The default values correspond to the most lenient setup possible."""
    parser = argparse.ArgumentParser(
        "Experiment Spawner", description="Spawn experiments on AWS MTurk"
    )
    parser.add_argument(
        "--experiment-name", required=True, help="name of experiment for url"
    )
    parser.add_argument(
        "--task-namespace",
        required=True,
        help="condition and number of " "reference images for url",
    )
    parser.add_argument("--n-tasks", required=True, type=int)
    parser.add_argument("--n-repetitions", required=True, type=int)
    parser.add_argument(
        "--base-data-url",
        required=False,
        default="https://yourdomain.tld/mturk-data/experiments/",
    )
    parser.add_argument("--single-participation-identifier", default=None, type=str)
    parser.add_argument("--environment", required=True, choices=("sandbox", "real"))
    parser.add_argument(
        "--reward",
        required=True,
        type=float,
        help="how many USD a worker makes with this HIT",
    )
    parser.add_argument("--output-folder", required=True)
    parser.add_argument(
        "--catch-trial-ratio-threshold",
        default=0,
        type=float,
        help="ratio of catch trials that must " "be passed",
    )
    parser.add_argument(
        "--min-total-response-time", default=0, type=float, help="time in [sec]"
    )
    parser.add_argument(
        "--max-total-response-time", default=60 * 60, type=float, help="time in [sec]"
    )
    parser.add_argument(
        "--min-instruction-time", default=0, type=float, help="time in [sec]"
    )
    parser.add_argument(
        "--max-instruction-time", default=-1, type=float, help="time in [sec]"
    )
    parser.add_argument(
        "--row-variability-threshold",
        default=0,
        type=float,
        help="number of trials that must be "
        "chosen from the less frequently "
        "selected row",
    )
    parser.add_argument(
        "--max-demo-trials-attempts",
        default=-1,
        type=int,
        help="number of allowed attempts to pass demo trials such that data is included in analysis",
    )
    parser.add_argument(
        "--previous-single-participation-qualifications",
        type=str,
        nargs="+",
        default=[],
        help="Exclude all workers who participated in previous experiments that used these qualifications",
    )
    parser.add_argument(
        "--hit-lifetime",
        default=1.0,
        type=float,
        help="Max lifetime of a HIT before to restart it (in hours)",
    )

    parser.add_argument(
        "--no-practice-trials", action="store_true", help="do not show practice trials"
    )
    parser.add_argument(
        "--no-instruction", action="store_true", help="do not show instruction"
    )
    parser.add_argument(
        "--no-bouncer", action="store_true", help="disable bouncer check"
    )

    parser.add_argument(
        "--task-type", required=True, type=str, choices=("2afc", "cf"), help=""
    )
    parser.add_argument(
        "--max-total-assignments",
        default=10,
        type=int,
        help="How often a HIT can be posted if the exclusion criteria were not met before.",
    )

    args = parser.parse_args()

    assert 0 <= args.catch_trial_ratio_threshold <= 1

    return args


def verify_tasknamespace_available(
    base_data_url, experiment_name, task_namespace, n_tasks
):
    def _url_file_exists(url):
        r = requests.head(url)
        return r.status_code == requests.codes.ok

    tns_url = urljoin(base_data_url, f"{experiment_name}/{task_namespace}/")

    for i in range(1, n_tasks + 1):
        if not _url_file_exists(urljoin(tns_url, f"task_{i}/index.json")):
            logging.error(f"Task {i} not found in task namespace")
            return False

    return True


def get_verify_task_callback(
    task_type,
    catch_trial_ratio_threshold,
    min_total_response_time,
    max_total_response_time,
    min_instruction_time,
    max_instruction_time,
    row_variability_threshold,
    max_demo_trials_attempts,
):
    """Constructs a callback that checks whether a task response passes all exclusion criteria"""

    def verify_catch_trials(main_trials):
        """Verify that the user took the catch trials seriously"""
        n_catch_trial_total = 0
        n_solved_catch_trials = 0

        catch_trials_answers = []
        for trial in main_trials:
            if trial["catch_trial"] == True:
                n_catch_trial_total += 1
                catch_trials_answers.append(trial["correct"] == True)
                if trial["correct"] == True:
                    n_solved_catch_trials += 1

        if n_catch_trial_total == 0:
            result = True
            ratio = float("nan")
        else:
            ratio = n_solved_catch_trials / n_catch_trial_total
            result = ratio >= catch_trial_ratio_threshold

        return result, {"ratio": ratio, "correctly_answered": catch_trials_answers}

    def verify_total_response_time(response_data):
        """Verify that the user did not take too long to carefully finish the task"""

        # get total time in seconds
        total_time = response_data[-1]["time_elapsed"] / 1000

        # ignore thresholds if they are set to their default values 0 and -1
        return (
            (min_total_response_time <= total_time or min_total_response_time == 0)
            and (total_time <= max_total_response_time),
            {"total_time": total_time},
        )

    def verify_row_variability(main_trials):
        """Verify that the user did not always choose the same row"""

        n_upper_row = 0
        n_lower_row = 0

        for trial in main_trials:
            if trial["button_pressed"] in [0, 1, 2]:
                n_upper_row += 1
            elif trial["button_pressed"] in [3, 4, 5]:
                n_lower_row += 1

        # case that only one trial per worker is tested
        if n_upper_row + n_lower_row:
            result = True
        else:
            result = (n_upper_row >= row_variability_threshold) and (
                n_lower_row >= row_variability_threshold
            )

        return result, {"n_upper_row": n_upper_row, "n_lower_row": n_lower_row}

    def verify_instruction_time(instructions):
        """Verify that the user carefully read the instructions"""

        # get total time in seconds
        total_time = instructions[-1]["time_elapsed"] / 1000

        # ignore thresholds if they are set to their default values 0 and -1
        return (
            (min_instruction_time <= total_time or min_instruction_time == 0)
            and (total_time <= max_instruction_time or max_instruction_time == -1),
            {"total_time": total_time},
        )

    def verify_demo_trials(demo_trials):
        """Verify that the user did not have to repeat the demo trials too often"""

        trial_types = [trial["trial_type"] for trial in demo_trials]
        # first block does not start with call_function trial, thus, add 1 afterwards
        n_demo_trials_blocks = 1 + len(
            [1 for it in trial_types if it == "call-function"]
        )

        return (
            (
                max_demo_trials_attempts == -1
                or n_demo_trials_blocks <= max_demo_trials_attempts
            ),
            {"n_demo_trials_blocks": n_demo_trials_blocks},
        )

    def verify_task_callback(mturkhit, response_data):
        """Here, we split data up into the four blocks. To this end, we first search for the first block, then for the
        second, etc.
        Args:
            mturkhit:       raw data (currently not used)
            response_data:  4 blocks: instructions, demo trials, further instructions, real trials
        """

        def search_block(start_index, criterion):
            """criterion (Callable) : Evaluates whether this is the index we are looking for
            based on the current element and the next element"""

            for i in range(start_index + 1, len(response_data["raw_data"]) - 1):
                if criterion(
                    response_data["raw_data"][i], response_data["raw_data"][i + 1]
                ):
                    return i

        """
        For the CF task the data has a structure like:
        ['instructions', 'instructions', 'fullscreen', 'call-function', 'cf-image-confidence-response',
         'cf-image-confidence-response', 'cf-image-confidence-response', 'cf-image-confidence-response', 'instructions',
         'call-function', 'cf-image-confidence-response', 'cf-image-confidence-response', 'cf-image-confidence-response',
         'cf-image-confidence-response', 'instructions', 'cf-image-confidence-response']
        """

        demo_trials_block_start_idx = search_block(
            0, lambda x, _: x["trial_type"] == f"{task_type}-image-confidence-response"
        )
        further_instructions_block_start_idx = search_block(
            demo_trials_block_start_idx,
            lambda x, x_next: (x["trial_type"] == "fullscreen")
            if task_type == "2afc"
            else (
                x["trial_type"] == "instructions"
                and x_next["trial_type"] == "cf-image-confidence-response"
            ),
        )
        if further_instructions_block_start_idx is None:
            # subtract -1 to make sure the we do not miss the first entry of the main type trials
            further_instructions_block_start_idx = demo_trials_block_start_idx - 1
        main_trials_block_start_idx = search_block(
            further_instructions_block_start_idx,
            lambda x, _: x["trial_type"] == f"{task_type}-image-confidence-response",
        )

        instructions = response_data["raw_data"][:demo_trials_block_start_idx]
        main_trials_block_end_idx = len(response_data["raw_data"]) - 1
        if task_type == "cf":
            # ignore last item for exclusion criteria since this is the feedback trial
            main_trials_block_end_idx -= 1
        main_trials = response_data["raw_data"][
            main_trials_block_start_idx:main_trials_block_end_idx
        ]
        demo_trials = response_data["raw_data"][
            demo_trials_block_start_idx:further_instructions_block_start_idx
        ]

        check_results = {}
        check_results["demo_trials"] = verify_demo_trials(demo_trials)
        check_results["catch_trials"] = verify_catch_trials(main_trials)
        check_results["total_response_time"] = verify_total_response_time(
            response_data["raw_data"]
        )
        check_results["row_variability"] = verify_row_variability(main_trials)
        check_results["instruction_time"] = verify_instruction_time(instructions)

        result = all([check_results[k][0] for k in check_results])

        return result, check_results

    return verify_task_callback


def main():
    args = parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    stdout_handler = logging.StreamHandler(sys.stdout)
    general_file_handler = logging.FileHandler(
        filename=os.path.join(args.output_folder, "log.log")
    )
    general_handlers = [stdout_handler, general_file_handler]
    logging.basicConfig(
        handlers=general_handlers,
        level=logging.INFO,
        format="[%(asctime)s] " + logging.BASIC_FORMAT,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info("Parsed arguments:")
    logging.info(args)

    sandbox = args.environment == "sandbox"
    if not sandbox:
        print("Running in real MTurk environment will cost real money.")
        if (
            input("If you are aware of that and want to continue, type 'yes': ")
            != "yes"
        ):
            print("Abort")
            return

    verify_tasknamespace_available(
        args.base_data_url, args.experiment_name, args.task_namespace, args.n_tasks
    )

    spawner = HITSpawner(
        args.reward,
        sandbox,
        None
        if args.single_participation_identifier is None
        else f"{args.experiment_name}-{args.single_participation_identifier}",
        ignore_qualifications=sandbox,
        previous_single_participation_qualifications=args.previous_single_participation_qualifications,
        hit_lifetime_in_hours=args.hit_lifetime,
        no_instruction=args.no_instruction,
        no_demo_trials=args.no_practice_trials,
        no_bouncer=args.no_bouncer,
    )

    tasks_to_spawn = list(range(1, args.n_tasks + 1))

    # check for already executed and saved tasks
    tasks_completed = set(
        [
            int(os.path.splitext(fn)[0].split("_")[-1])
            for fn in glob.glob(os.path.join(args.output_folder, "result_task_*.pkl"))
        ]
    )
    tasks_to_spawn = [it for it in tasks_to_spawn if it not in tasks_completed]

    if len(tasks_to_spawn) == 0:
        print("All tasks already answered.")
        sys.exit(0)

    print(f"Spawning {len(tasks_to_spawn)} tasks...")
    time.sleep(15)

    for task_id in tasks_to_spawn:
        # spawn tasks on MTurk
        logging.info(f"Posting task {task_id}")
        spawner.post_repeated_task(
            task_id,
            args.task_namespace,
            args.experiment_name,
            args.n_repetitions,
            int(args.max_total_response_time),
            task_type=args.task_type,
            max_total_assignments=args.max_total_assignments,
        )

    def save_single_task(hit, tr):
        # save results per task:
        logging.info(f"Save results for task {hit.task_id}")
        with open(
            os.path.join(args.output_folder, f"result_task_{hit.task_id}.pkl"), "wb"
        ) as f:
            pickle.dump(tr, f)

    # wait for results
    logging.info("Waiting for results...")
    results = spawner.get_repeated_task_results(
        get_verify_task_callback(
            args.task_type,
            args.catch_trial_ratio_threshold,
            args.min_total_response_time,
            args.max_total_response_time,
            args.min_instruction_time,
            args.max_instruction_time,
            args.row_variability_threshold,
            args.max_demo_trials_attempts,
        ),
        received_single_task_result=save_single_task,
    )

    # save results
    logging.info(f"Save results for all tasks")
    with open(os.path.join(args.output_folder, f"results_all_tasks.pkl"), "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()
