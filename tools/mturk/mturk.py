"""API handling the communication between AWS MTurk and our code."""
import datetime

import boto3
from aws_credentials import AWSCredentials
import warnings
import logging
import time
import xml.etree
import json
from collections import namedtuple
from typing import List, Optional, Callable, Tuple
from typing_extensions import Literal


class HITSpawner:
    """
    Args:
        task_reward (float): reward per HIT in USD
        sandbox (bool): ùse the sandbox instead of the real environment
        single_participation_qualification (str): name of mturk qualification type that will be created
            and awarded to each participant to restrict them to only participate once
        ignore_qualifications (bool): ignore previous qualifications
        previous_single_participation_qualifications (list of str): list of all previous qualifications which will be
            used as a blacklist for participants
        hit_lifetime_in_hours (int): how long the hit exists, until it will be posted again
        no_instruction (bool): skip task instructions
        no_demo_trials (bool): skip practice/demo trials
        no_bouncer (bool): turn of the Bouncer tool that makes sure every participant can really
            only participate once
    """

    def __init__(
        self,
        task_reward: float,
        sandbox: bool,
        single_participation_qualification: Optional[str] = None,
        ignore_qualifications: Optional[bool] = False,
        previous_single_participation_qualifications: Optional[List[str]] = None,
        hit_lifetime_in_hours=24 * 14,  # default on MTurk is 14 days
        no_instruction=False,
        no_demo_trials=False,
        no_bouncer=False,
    ):
        if sandbox:
            warnings.warn("Using MTurk Sandbox environment - this will be free")
            endpoint_url = "https://mturk-requester-sandbox.us-east-1.amazonaws.com"
        else:
            warnings.warn("Using real MTurk environment - this will charge you money!")
            endpoint_url = "https://mturk-requester.us-east-1.amazonaws.com"

        region_name = "us-east-1"

        self._client = boto3.client(
            "mturk",
            endpoint_url=endpoint_url,
            region_name=region_name,
            aws_access_key_id=AWSCredentials.aws_access_key_id(),
            aws_secret_access_key=AWSCredentials.aws_secret_access_key(),
        )

        assert task_reward > 0, "Task reward must be positive number."
        self._task_reward = task_reward
        self._single_participation_qualification = single_participation_qualification
        self._single_participation_qualification_id = None
        if previous_single_participation_qualifications is None:
            previous_single_participation_qualifications = []
        self._previous_single_participation_qualification_ids = []
        self._setup_previous_single_participation_qualifications(
            previous_single_participation_qualifications
        )
        self._ignore_qualifications = ignore_qualifications
        assert hit_lifetime_in_hours > 0
        self._hit_lifetime_in_hours = hit_lifetime_in_hours
        self._hits: List[MTurkHIT] = []

        self.sandbox = sandbox

        self.no_instruction = no_instruction
        self.no_demo_trials = no_demo_trials
        self.no_bouncer = no_bouncer

        if single_participation_qualification is not None:
            self._setup_single_participation_qualification()
        else:
            self.__logger.info(
                "No participation restriction qualifier submitted; participation will "
                "be unrestricted"
            )

    def _setup_previous_single_participation_qualifications(
        self, previous_single_participation_qualifications
    ):
        """For each of the previous qualification names, get the internal ID
        needed to assign the qualification to workers"""

        self._previous_single_participation_qualification_ids += self._get_previous_single_participation_qualification_ids(
            previous_single_participation_qualifications
        )

    def _get_previous_single_participation_qualification_ids(
        self, previous_single_participation_qualifications
    ):
        """For each of the previous qualification names, get the internal ID
        needed to assign the qualification to workers"""

        ids = []
        for qualification in previous_single_participation_qualifications:
            try:
                self.__logger.info(
                    "Getting ID for qualification type `{0}` to restrict participation".format(
                        qualification
                    )
                )
                response = self._client.list_qualification_types(
                    Query="bethgelab-hap-" + qualification, MustBeRequestable=False
                )
                for item in response["QualificationTypes"]:
                    ids.append(item["QualificationTypeId"])
                if len(response["QualificationTypes"]) > 0:
                    self.__logger.info(
                        "Successfully got ID {0} for qualification type {1}".format(
                            ids[-1], qualification
                        )
                    )
                else:
                    raise ValueError(
                        f"Could not find qualification ID for qualification `{qualification}`"
                    )
            except Exception as ex:
                self.__logger.warning(
                    f"Failed to get ID qualification type `{qualification}`: {repr(ex)}"
                )

        return ids

    def _setup_single_participation_qualification(self):
        """Create a qualification type to assign later to workers"""

        try:
            self.__logger.info(
                "Creating qualification type `{0}` to restrict participation".format(
                    self._single_participation_qualification
                )
            )

            response = self._client.create_qualification_type(
                Name="bethgelab-hap-" + self._single_participation_qualification,
                Description="This qualification indicates that a worker has participated "
                "in a recent study executed by the Bethgelab with id "
                + self._single_participation_qualification,
                QualificationTypeStatus="Active",
            )
            self._single_participation_qualification_id = response["QualificationType"][
                "QualificationTypeId"
            ]
            self.__logger.info(
                "Successfully created qualification type `{0}` with ID `{1}`".format(
                    self._single_participation_qualification,
                    self._single_participation_qualification_id,
                )
            )
        except Exception as ex:
            self.__logger.warning("Failed to create qualification type: " + repr(ex))
            self.__logger.info("Trying to treat it as a previous qualification type...")
            self._single_participation_qualification_id = self._get_previous_single_participation_qualification_ids(
                [self._single_participation_qualification]
            )[
                0
            ]

        if self._single_participation_qualification_id is None:
            self.__logger.warn("No single participation qualification ID used")
        else:
            self.__logger.info(
                f"Using single participation qualification ID `{self._single_participation_qualification_id}`"
            )

    def _award_participation_qualification(self, worker_id: str):
        """Awards a qualification to an MTurk worker"""

        if self._single_participation_qualification_id is None:
            return

        while True:
            try:
                self.__logger.info(
                    "Assigning participation qualification to worker: " + worker_id
                )

                self._client.associate_qualification_with_worker(
                    QualificationTypeId=self._single_participation_qualification_id,
                    WorkerId=worker_id,
                    SendNotification=False,
                    IntegerValue=1,
                )

                return
            except Exception as ex:
                self.__logger.info("Failed to award qualification type: " + repr(ex))
                time.sleep(5)

    @property
    def __logger(self):
        return logging.getLogger("mturk")

    def post_repeated_task(
        self,
        task_id: str,
        task_namespace: str,
        experiment_name: str,
        n_repetitions: int,
        assignment_duration: int,
        max_total_assignments: int,
        task_type: Tuple[Literal["2afc"], Literal["cf"]],
        retry_on_failure: Optional[bool] = True,
    ):
        """
        Post a task online on MTurk.

        Args:
            task_id (str): id of the task
            task_namespace (str): namespace of the task, used to refer to different types of experiments
            experiment_name (str): name of the experiment
            n_repetitions (str): how often the same task should be posted
            assignment_duration (int): duration in seconds
            max_total_assignments (int): maximal number of assignments for the HIT; once reached, no more will be posted
                even if not enough accepted responses have been collected yet
            task_type (str): controls the type of the task; "2afc" for the 2-alternative-forced-choice task used to
                replicate Borowski et al. (2021) or "cf" for the new counter-factual task
            retry_on_failure (bool): whether the HIT should be posted again if posting it
                failed (e.g., due to network errors)
        """

        question_pattern = """
            <ExternalQuestion xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2006-07-14/ExternalQuestion.xsd">
               <ExternalURL>https://yourdomain.tld/mturk/{3}/start.html?tid={0}&amp;tns={1}&amp;exp={2}{4}{5}{6}</ExternalURL>
               <FrameHeight>800</FrameHeight>
            </ExternalQuestion>
            """

        question = question_pattern.format(
            task_id,
            task_namespace,
            experiment_name,
            task_type,
            "&amp;ni" if self.no_instruction else "",
            "&amp;nd" if self.no_demo_trials else "",
            "&amp;nb" if self.no_bouncer else "",
        )

        while True:
            try:
                if self._ignore_qualifications:
                    qualification_requirements = []
                else:
                    qualification_requirements = [
                        {
                            # Worker_Locale
                            "QualificationTypeId": "00000000000000000071",
                            "Comparator": "In",
                            "LocaleValues": [
                                {"Country": "US",},
                                {"Country": "CA",},
                                {"Country": "GB",},
                                {"Country": "AU",},
                                {"Country": "NZ",},
                                {"Country": "IE",},
                            ],
                            "ActionsGuarded": "DiscoverPreviewAndAccept",
                        },
                        {
                            # Worker_​NumberHITsApproved
                            "QualificationTypeId": "00000000000000000040",
                            "Comparator": "GreaterThanOrEqualTo",
                            "IntegerValues": [1000],
                            "ActionsGuarded": "DiscoverPreviewAndAccept",
                        },
                        {
                            # Worker_​PercentAssignmentsApproved
                            "QualificationTypeId": "000000000000000000L0",
                            "Comparator": "GreaterThanOrEqualTo",
                            "IntegerValues": [99],
                            "ActionsGuarded": "DiscoverPreviewAndAccept",
                        },
                    ]

                    # prevent workers from participating twice in the same experiment
                    if self._single_participation_qualification_id is not None:
                        qualification_requirements += [
                            {
                                # Single participation only
                                "QualificationTypeId": self._single_participation_qualification_id,
                                "Comparator": "DoesNotExist",
                                "ActionsGuarded": "DiscoverPreviewAndAccept",
                            },
                        ]

                    # also prevent worker who participated in previous experiments to participate again
                    for (
                        qualification_id
                    ) in self._previous_single_participation_qualification_ids:
                        qualification_requirements += [
                            {
                                # Single participation only
                                "QualificationTypeId": qualification_id,
                                "Comparator": "DoesNotExist",
                                "ActionsGuarded": "DiscoverPreviewAndAccept",
                            },
                        ]

                self.__logger.info("Posting task.")

                result = self._client.create_hit(
                    MaxAssignments=n_repetitions,
                    # remove after 14 days
                    LifetimeInSeconds=int(60 * 60 * self._hit_lifetime_in_hours),
                    AssignmentDurationInSeconds=assignment_duration,
                    Reward=f"{self._task_reward:.2f}",
                    Title="Image Classification",
                    Description="Academic survey in which you have to classify images, i.e. "
                    " match them with another set of images.",
                    RequesterAnnotation=f"{experiment_name}-{task_namespace}-{task_id}",
                    Question=question,
                    AutoApprovalDelayInSeconds=0,  # automatically accept the assignment without checking it
                    QualificationRequirements=qualification_requirements,
                )

                self.__logger.info("Posting task succeeded.")

                hit = MTurkHIT.parse_server_response(
                    result["HIT"],
                    f"{task_namespace}-{task_id}",
                    task_id,
                    task_namespace,
                    experiment_name,
                    assignment_duration,
                    max_total_assignments,
                    task_type,
                )
                self._hits.append(hit)
            except Exception as ex:
                if retry_on_failure:
                    self.__logger.warning(
                        "Posting task failed (will retry): " + repr(ex)
                    )
                    time.sleep(1)
                else:
                    self.__logger.warning("Posting task failed: " + repr(ex))
                    raise ex
            else:
                break

    def get_repeated_task_results(
        self,
        verify_task: Optional[Callable] = lambda *_: True,
        received_single_task_result: Optional[Callable] = lambda *_: None,
    ):
        """Collect the results for the posted tasks from MTurk.

        Args:
            verify_task (callback): callback to verify whether a HIT satisfies all quality requirements
            received_single_task_result (callback): is called every time a response has been collected,
                even if it was rejected to due to poor quality
        """

        received_single_task_result_called_map = dict()

        while True:
            for hit in [x for x in self._hits if not x.required_responses_loaded]:
                hit.load_responses(self._client, verify_task)
                if hit.required_responses_loaded:
                    for worker_id in hit.assigned_worker_ids:
                        self._award_participation_qualification(worker_id)

                    if (
                        hit.max_total_assignments_reached
                        and hit.hit_id not in received_single_task_result_called_map
                    ):
                        received_single_task_result_called_map[hit.hit_id] = True
                        received_single_task_result(
                            hit,
                            RepeatedTaskResult(
                                task_id=hit.combined_task_id,
                                responses=hit.responses,
                                approved_responses=hit.approved_responses,
                                rejected_responses=hit.rejected_responses,
                                raw_responses=hit.raw_responses,
                            ),
                        )

                    if hit.hit_id not in received_single_task_result_called_map:
                        received_single_task_result_called_map[hit.hit_id] = True
                        received_single_task_result(
                            hit,
                            RepeatedTaskResult(
                                task_id=hit.combined_task_id,
                                responses=hit.responses,
                                approved_responses=hit.approved_responses,
                                rejected_responses=hit.rejected_responses,
                                raw_responses=hit.raw_responses,
                            ),
                        )

                if hit.is_expired and not hit.required_responses_loaded:
                    # only act if no worker is still working on the HIT
                    if not hit.check_if_assignments_pending(self._client):
                        # hit has expired
                        # check if we can still add new assignment
                        if not hit.max_total_assignments_reached:
                            # post it again and remove current hit from list
                            # then start loop again

                            # try to delete the HIT
                            deletion_success = hit.delete(self._client)
                            # only remove from our list and post again if it could be deleted
                            # otherwise there might still be a pending result coming in
                            if deletion_success:
                                self._hits.remove(hit)
                                self.post_repeated_task(
                                    hit.task_id,
                                    hit.task_namespace,
                                    hit.experiment_name,
                                    hit.n_approved_assignments_required,
                                    hit.assignment_duration,
                                    max_total_assignments=hit.max_total_assignments,
                                    task_type=hit.task_type,
                                )
                                break
                    else:
                        break

            if all(
                [
                    hit.required_responses_loaded
                    or hit.max_total_assignments_reached
                    and hit.assignments_open == 0
                    for hit in self._hits
                ]
            ):
                break
            else:
                time.sleep(10)

        responses = [
            RepeatedTaskResult(
                task_id=hit.combined_task_id,
                responses=hit.responses,
                approved_responses=hit.approved_responses,
                rejected_responses=hit.rejected_responses,
                raw_responses=hit.raw_responses,
            )
            for hit in self._hits
        ]

        self._hits.clear()

        return responses


RepeatedTaskResult = namedtuple(
    "RepeatedTaskResult",
    [
        "task_id",
        "raw_responses",
        "responses",
        "approved_responses",
        "rejected_responses",
    ],
)


class MTurkHIT:
    """

    Args:
        task_id: Internal ID used for referring to a specific task.
        hit_id: External ID used by AWS MTurk.
        n_approved_assignments_required: Number of assignments which have passed the catch trial check required.
        asked_assignments: How many assignments were initially asked.
    """

    def __init__(
        self,
        combined_task_id: str,
        task_id: str,
        task_namespace: str,
        experiment_name: str,
        hit_id: str,
        assignment_duration: int,
        creation_time: datetime.datetime,
        expiration_time: datetime.datetime,
        max_total_assignments: int,
        task_type: Tuple[Literal["2afc"], Literal["cf"]],
        n_approved_assignments_required: int = 1,
        asked_assignments: int = 1,
    ):
        self._combined_task_id = combined_task_id
        self._task_id = task_id
        self._task_namespace = task_namespace
        self._experiment_name = experiment_name
        self._hit_id = hit_id
        self._assignment_duration = assignment_duration
        self._response_data = []
        self._assigned_worker_ids = []
        self._creation_time = creation_time
        self._expiration_time = expiration_time
        self._max_total_assignments = max_total_assignments
        self._task_type = task_type
        self._n_approved_assignments_required = n_approved_assignments_required
        self._asked_assignments = asked_assignments
        self._processed_assignments = []

        self._assignments_open = asked_assignments

        if self._n_approved_assignments_required < 10:
            assert (
                self._asked_assignments <= 10
            ), "HITs with originally less than 10 assignments cannot be reposted more than 9 times."

    @staticmethod
    def parse_server_response(
        response: dict,
        combined_task_id: str,
        task_id: str,
        task_namespace: str,
        experiment_name: str,
        assignment_duration: int,
        max_total_assignments: int,
        task_type: Tuple[Literal["2afc"], Literal["cf"]],
    ):
        """Parse AWS MTurk's response"""
        hit_id = response["HITId"]
        max_assignments = response["MaxAssignments"]
        creation_time = response["CreationTime"]
        expiration_time = response["Expiration"]
        res = MTurkHIT(
            combined_task_id,
            task_id,
            task_namespace,
            experiment_name,
            hit_id,
            assignment_duration,
            creation_time=creation_time,
            expiration_time=expiration_time,
            max_total_assignments=max_total_assignments,
            task_type=task_type,
            n_approved_assignments_required=max_assignments,
            asked_assignments=max_assignments,
        )

        res.internal_data = response

        return res

    @property
    def is_expired(self):
        now = datetime.datetime.now(self._expiration_time.tzinfo)
        return now > self._expiration_time

    @property
    def task_type(self):
        return self._task_type

    @property
    def max_total_assignments(self):
        return self._max_total_assignments

    @property
    def approved_responses(self):
        return [it for it in self._response_data if it["passed_checks"]]

    @property
    def rejected_responses(self):
        return [it for it in self._response_data if not it["passed_checks"]]

    @property
    def assignment_duration(self):
        return self._assignment_duration

    @property
    def required_responses_loaded(self):
        return len(self.approved_responses) == self._n_approved_assignments_required

    @property
    def max_total_assignments_reached(self):
        return self._asked_assignments >= self._max_total_assignments

    @property
    def assignments_open(self):
        return self._assignments_open

    @property
    def n_approved_assignments_required(self):
        return self._n_approved_assignments_required

    @property
    def annotation_task(self):
        return self._annotation_task

    @property
    def assigned_worker_ids(self):
        return self._assigned_worker_ids

    @property
    def responses(self):
        return [
            dict(main_data=it["main_data"], raw_data=it["raw_data"])
            for it in self._response_data
        ]

    @property
    def raw_responses(self):
        return self._response_data

    @property
    def hit_id(self):
        return self._hit_id

    @property
    def combined_task_id(self):
        return self._combined_task_id

    @property
    def task_id(self):
        return self._task_id

    @property
    def experiment_name(self):
        return self._experiment_name

    @property
    def task_namespace(self):
        return self._task_namespace

    @property
    def __logger(self):
        return logging.getLogger("mturk")

    def check_if_assignments_pending(self, client):
        """Check if a worker has accepted the HIT but has not yet submitted the response"""
        try:
            self.__logger.info(
                f"Checking for pending assignments for HIT {self._hit_id}."
            )
            response = client.get_hit(HITId=self._hit_id)
            n_pending = response["HIT"]["NumberOfAssignmentsPending"]
            return n_pending != 0
        except Exception as ex:
            self.__logger.warning("Failed to check pending assignments: " + repr(ex))
            return

    def _parse_assignment_response(
        self, assignment: dict, verify_task: Optional[Callable] = lambda *_: True
    ):
        """Parse the JSON response from AWS to extract to relevant information"""
        worker_id = assignment["WorkerId"]
        answer = assignment["Answer"]

        response_document = xml.etree.ElementTree.fromstring(answer)
        response_data = {
            i.getchildren()[0].text: i.getchildren()[1].text
            for i in response_document.getchildren()
        }

        all_data_available = True
        if "raw_data" in response_data:
            response_data["raw_data"] = json.loads(response_data["raw_data"])
        else:
            response_data["raw_data"] = None
            all_data_available = False

        if "main_data" in response_data:
            response_data["main_data"] = json.loads(response_data["main_data"])
        else:
            response_data["main_data"] = None
            all_data_available = False

        if all_data_available:
            (
                response_data["passed_checks"],
                response_data["check_results"],
            ) = verify_task(self, response_data)
        else:
            response_data["passed_checks"] = False
            response_data["check_results"] = None

        response_data["worker_id"] = worker_id

        return response_data, worker_id

    def load_responses(self, client, verify_task: Optional[Callable] = lambda *_: True):
        """Load the responses form AWS MTurk and add new assignments if a worker did not pass
        the catch trials.
        """

        self.__logger.info(f"Loading response for HIT {self._hit_id}.")

        # Approved here means that they were approved by MTurk.
        # Since we automatically accept all assignments, all of them
        # should have the status Approved eventually.
        approved_assignments = []
        paginator = client.get_paginator("list_assignments_for_hit")
        for hit_details_page in paginator.paginate(
            HITId=self._hit_id, AssignmentStatuses=["Approved"]
        ):
            approved_assignments += hit_details_page["Assignments"]

        for assignment in approved_assignments:
            assignment_id = assignment["AssignmentId"]
            if assignment_id in self._processed_assignments:
                continue
            else:
                self._processed_assignments.append(assignment_id)

            self.__logger.info(
                f"Found 1 newly solved assignments. Validating checks..."
            )

            response_data, _ = self._parse_assignment_response(assignment, verify_task)
            passed_checks = response_data["passed_checks"]

            if not passed_checks:
                self.__logger.info(f"Assignment {assignment_id} did not pass checks.")
                self.__logger.info(f"Details: {response_data['check_results']}")

                if self._asked_assignments >= self._max_total_assignments:
                    self.__logger.info(
                        f"Not posting new assignment for HIT {self._hit_id} since max. number "
                        f"of assignments ({self._max_total_assignments}) are reached."
                    )
                else:
                    # this can be done 10 times in total for hits with originally less than 10 assignments
                    if (
                        self._n_approved_assignments_required < 10
                        and self._asked_assignments > 10
                    ):
                        self.__logger.info(
                            "Cannot add new assignment since too many were already started"
                        )
                    else:

                        self.__logger.info("Adding new assignment")
                        client.create_additional_assignments_for_hit(
                            HITId=self._hit_id, NumberOfAdditionalAssignments=1
                        )
                        self._asked_assignments += 1

                        self._assignments_open += 1

        n_previously_accepted_assignments = len(self._assigned_worker_ids)

        if n_previously_accepted_assignments < len(approved_assignments):
            # there was a change, update our list of assignments
            self._assigned_worker_ids = []
            self._response_data = []

            self._assignments_open -= (
                len(approved_assignments) - n_previously_accepted_assignments
            )

            self.__logger.info(
                f"Found {len(approved_assignments) - n_previously_accepted_assignments} newly "
                f"solved, approved and validated assignments."
            )
            for assignment in approved_assignments:
                response_data, worker_id = self._parse_assignment_response(
                    assignment, verify_task
                )

                self._response_data.append(response_data)
                self._assigned_worker_ids.append(worker_id)
        else:
            self.__logger.info("Found no newly approved and validated assignments.")

    def delete(self, client, retries=3):
        """Delete a HIT if there are no finished assignments stored in it"""
        for _ in range(retries):
            try:
                self.__logger.info(
                    f"Checking for pending/completed assignments for HIT {self._hit_id}."
                )
                response = client.get_hit(HITId=self._hit_id)
                n_pending = response["HIT"]["NumberOfAssignmentsPending"]
                n_completed = response["HIT"]["NumberOfAssignmentsCompleted"]
                n_total = n_pending + n_completed

                is_reviewable = response["HIT"]["HITStatus"] == "Reviewable"

                if n_total > 0:
                    self.__logger.info(
                        f"Cannot delete HIT {self._hit_id} due to {n_total}"
                        f" completed/pending assignment(s)."
                    )
                    return False
                else:
                    if is_reviewable:
                        self.__logger.info(f"Deleting HIT {self._hit_id}...")
                        response = client.delete_hit(HITId=self._hit_id)
                        return response["ResponseMetadata"]["HTTPStatusCode"] == 200
                    else:
                        self.__logger.info(
                            f"Cannot delete HIT {self._hit_id} as it is not marked as `Reviewable`."
                        )
                        return False
            except Exception as ex:
                self.__logger.warning("Failed to delete HIT: " + repr(ex))

            time.sleep(30)
