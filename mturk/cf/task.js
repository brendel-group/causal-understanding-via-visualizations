SERVER_TASKS_URL = "https://yourdomain.tld/mturk-data/experiments/";

function initialize() {
  // check whether the HIT has been accepted
  let url = new URL(window.location.href);
  const debug = url.searchParams.get("debug");

  const debug_mode = debug !== null;

  if (!debug_mode && window.opener == null) {
    alert(
      "Something went wrong. Please go back to MTurk and press the 'Go to Experiment'-button again. A pop-up will appear that lets you work on the HIT."
    );
    window.close();
  } else {
    // check whether the user has access to the task
    runExperiment();
  }
}

function runExperiment(type) {
  // get our internal ID to load the correct images
  let url = new URL(window.location.href);
  let taskId = url.searchParams.get("tid");
  let taskNamespace = url.searchParams.get("tns");
  let experimentName = url.searchParams.get("exp");

  let noInstructions = url.searchParams.get("ni");
  noInstructions = noInstructions !== null;

  let noDemo = url.searchParams.get("nd");
  noDemo = noDemo !== null;

  let taskIndexUrl = new URL(
    `${experimentName}/${taskNamespace}/task_${taskId}/index.json`,
    SERVER_TASKS_URL
  );

  let callback = prepareExperiment;
  let demoTaskIndexUrl = new URL(
    `${experimentName}/demo_${taskNamespace}/index.json`,
    SERVER_TASKS_URL
  );

  fetchAllJson(taskIndexUrl, demoTaskIndexUrl)
    .catch((e) => {
      alert("There was an error starting this HIT, please try it again.");
      // alert('You can no longer access this experiment as it has expired. Reminder: after accepting a task you must complete it within 5 minutes.')
    })
    .then((tasks) =>
      callback(...tasks, taskId, experimentName, noInstructions, noDemo)
    );
}

function prepareExperiment(
  main_task_config,
  demo_task_config,
  taskId,
  experimentName,
  noInstructions,
  noDemo
) {
  let timeline = [];

  // set random generator
  Math.seedrandom(main_task_config["task_name"]);

  function addTrials(trials, timeline, is_demo, start_progress, end_progress) {
    let task_timeline = [
      {
        type: "cf-image-confidence-response",
        query_a_stimulus: jsPsych.timelineVariable("min_query"),
        query_b_stimulus: jsPsych.timelineVariable("max_query"),
        query_base_stimulus: jsPsych.timelineVariable("base_query"),
        reference_stimuli: jsPsych.timelineVariable("max_references"),
        choices: ["3", "2", "1", "1", "2", "3"],
        prompt: "",
        correct_text:
          '<p style="margin: 0">That was <span style="color: darkgreen; font-weight: bold;">correct</span>!</p>',
        incorrect_text:
          '<p style="margin: 0">That was <span style="color: darkred; font-weight: bold;">incorrect</span>!</br> The machine actually <span style="color: darkred; font-weight: bold;">favors the other image more</span>.</p>',
        reference_title: "Favorite Images",
        query_base_title:
          main_task_config["n_reference_images"] == 0
            ? "Favorite Image"
            : "Another Favorite Image",
        query_title: "Which image is more favored?",
        feedback_delay_duration: 350,
        feedback_duration: 2000 ? is_demo : 0,
        randomize_queries: true,
        response_ends_trial: false,
        correct_query_choice: "b",
        data: {
          id: jsPsych.timelineVariable("task_id"),
          min_query: jsPsych.timelineVariable("min_query"),
          max_query: jsPsych.timelineVariable("max_query"),
          base_query: jsPsych.timelineVariable("base_query"),
          max_references: jsPsych.timelineVariable("max_references"),
          catch_trial: jsPsych.timelineVariable("catch_trial"),
          is_demo: is_demo,
        },
        on_finish: function () {
          jsPsych.setProgressBar(jsPsych.timelineVariable("progress")());
        },
      },
    ];

    timeline.push({
      timeline: task_timeline,
      timeline_variables: trials.map(function (trial, i) {
        return {
          trial_id: trial.id,
          progress:
            start_progress +
            ((end_progress - start_progress) / trials.length) * (i + 1),
          min_query: trial.min_query,
          max_query: trial.max_query,
          base_query: trial.base_query,
          // max_references: jsPsych.randomization.shuffle(trial.max_references),
          max_references: trial.max_references,
          catch_trial: trial.catch_trial,
        };
      }),
    });
  }

  let instructionImages = [];
  if (!noInstructions) {
    let welcome = {
      type: "instructions",
      pages: ["Welcome to this experiment! </br> It will start soon."],
      show_clickable_nav: true,
      on_finish: function () {
        jsPsych.setProgressBar(0.01);
      },
    };
    timeline.push(welcome);

    let instructions;
    if (main_task_config["n_reference_images"] == 0) {
      instructionImages = Array.from({ length: 10 }, (_, i) => i).map(
        (i) =>
          new URL(
            `${experimentName}/${
              main_task_config["task_name"].split("/")[0]
            }/instructions/${i}.jpg`,
            SERVER_TASKS_URL
          )
      );

      instructions = {
        timeline: [
          {
            type: "instructions",
            pages: [
              "In this experiment, you will be shown images on the screen and <br> asked to make a response by clicking your mouse.",
              "The experiment consists of multiple trials. <br> We will now explain to you how a single trial works.",
              `<br><br><br>In the middle of the screen, you see one image. <br> This is a favorite image of a machine.<br> <img id="jspsych-instructions-image" src="${instructionImages[0]}" />`,
              `<br><br>Below that image, you see two more images. <br> In each image, a square has been placed to cover part of the upper image. <br> This part is hidden from the machine. <br> <img id="jspsych-instructions-image" src="${instructionImages[1]}" />`,
              `<br><br><br>The question you have to answer is always the following: <br> <i> Which image at the bottom is more favored by the machine? </i>  <br><img id="jspsych-instructions-image" src="${instructionImages[2]}" />`,
              `<br><br><br>This is how you answer:<br>Below the two images at the bottom you see two rows of numbers. <br> <img id="jspsych-instructions-image" src="${instructionImages[3]}" />`,
              `<br><br><br> If you think the <b>left</b> image is more favored by the machine, <br> choose a number from the <b>left</b> row of numbers. <br> <img id="jspsych-instructions-image" src="${instructionImages[4]}" />`,
              `<br><br><br> If you think the <b>right</b> image is more favored, <br> choose a number from the <b>right</b> row of numbers. <br> <img id="jspsych-instructions-image" src="${instructionImages[5]}" />`,
              `<br><br>The value of the number indicates how confident you are in your choice: <br> The higher the number, the higher your confidence. <br> If you are not sure, go with your best guess.  <br> <img id="jspsych-instructions-image" src="${instructionImages[6]}" />`,
              `<br><br><br><br>Once you provide your answer, a black frame appears around your chosen image.  <br> <img id="jspsych-instructions-image" src="${instructionImages[7]}" />`,
              `<br><br>This is the end of one trial. <br> To summarize, your task is to choose the image at the bottom <br> that is also a favorite image of the machine. <br> <img id="jspsych-instructions-image" src="${instructionImages[8]}" />`,
              `<br>In each trial, a different favorite image is shown, <br> and each trial is independent of the other ones. <br> The covered image parts may have to do with parts of objects, <br> patterns, color, or even more abstract aspects. <br> <img id="jspsych-instructions-image" src="${instructionImages[8]}" />`,
              `<br><br><br><br>By clicking on the <it>Continue</it> button you continue to the next trial. <br> <img id="jspsych-instructions-image" src="${instructionImages[9]}" />`,
              `<br><br><br><br>This is the last opportunity to go back and re-read the instructions via the <it>Previous</it> button. <br> <img id="jspsych-instructions-image" src="${instructionImages[9]}" />`,
            ],
            images: [null, null].concat(instructionImages),
            show_clickable_nav: true,
            on_finish: function () {
              jsPsych.setProgressBar(0.05);
            },
          },
        ],
      };
    } else {
      instructionImages = Array.from({ length: 11 }, (_, i) => i).map(
        (i) =>
          new URL(
            `${experimentName}/${
              main_task_config["task_name"].split("/")[0]
            }/instructions/${i}.jpg`,
            SERVER_TASKS_URL
          )
      );

      instructions = {
        timeline: [
          {
            type: "instructions",
            pages: [
              "In this experiment, you will be shown images on the screen and <br> asked to make a response by clicking your mouse.",
              "The experiment consists of multiple trials. <br> We will now explain to you how a single trial works.",
              `<br>On the left side of the screen, you see a group of example images. <br> These are the <b>Favorite Images</b> of a machine${
                main_task_config["task_name"].includes("blur")
                  ? ", with the very favorite aspect in high resolution."
                  : "."
              }<br>Usually, these images share at least one common aspect.<br>In this case, all of them are related to birds. <br> <img id="jspsych-instructions-image" src="${
                instructionImages[0]
              }" />`,
              `<br><br>At the top right of the screen, you see one more image. <br> This is yet another favorite image of the machine.<br>And you probably already spotted the bird in it again, didn't you? <br> <img id="jspsych-instructions-image" src="${instructionImages[1]}" />`,
              `<br><br>At the bottom right of the screen, you see two more images. <br> In each image, a square has been placed to cover part of the image. <br> This part is hidden from the machine. <br> <img id="jspsych-instructions-image" src="${instructionImages[2]}" />`,
              `<br><br>The question you have to answer is always the following: <br> <i> Which image at the bottom right is more favored by the machine? </i> <br> In other words, in which image do you still see the common aspect (here: bird) of the favorite images?<br><img id="jspsych-instructions-image" src="${instructionImages[3]}" />`,
              `<br><br><br>This is how you answer:<br>Below the two images at the bottom right you see two rows of numbers. <br> <img id="jspsych-instructions-image" src="${instructionImages[4]}" />`,
              `<br><br><br> If you think the <b>left</b> image is more favored by the machine, <br> choose a number from the <b>left</b> row of numbers. <br> <img id="jspsych-instructions-image" src="${instructionImages[5]}" />`,
              `<br><br><br> If you think the <b>right</b> image is more favored, <br> choose a number from the <b>right</b> row of numbers. <br> <img id="jspsych-instructions-image" src="${instructionImages[6]}" />`,
              `<br><br>The value of the number indicates how confident you are in your choice: <br> The higher the number, the higher your confidence. <br> If you are not sure, go with your best guess.  <br> <img id="jspsych-instructions-image" src="${instructionImages[7]}" />`,
              `<br><br><br><br>Once you provide your answer, a black frame appears around your chosen image.  <br> <img id="jspsych-instructions-image" src="${instructionImages[8]}" />`,
              `<br>This is the end of one trial. <br> To summarize, your task is to understand <br> what the machine likes based on the left images, and to then <br> choose the image on the right that still contains that favorite aspect. <br> <img id="jspsych-instructions-image" src="${instructionImages[9]}" />`,
              `<br>In each trial, a different common aspect might be important. <br>While the common aspect in this trial had to do with birds, <br> common aspects may also have to do with parts of objects, <br> patterns, color, or even more abstract aspects. <br> <img id="jspsych-instructions-image" src="${instructionImages[9]}" />`,
              `<br><br><br><br>By clicking on the <it>Continue</it> button you continue to the next trial. <br> <img id="jspsych-instructions-image" src="${instructionImages[10]}" />`,
              `<br><br>This is the last opportunity to go back and re-read the instructions via the <it>Previous</it> button. <br>Otherwise, we will start with a few demo trials <br>so that you can familiarize yourself with the experiment. <br> <img id="jspsych-instructions-image" src="${instructionImages[10]}" />`,
            ],
            images: [null, null].concat(instructionImages),
            show_clickable_nav: true,
            on_finish: function () {
              jsPsych.setProgressBar(0.05);
            },
          },
        ],
      };
    }

    timeline.push(instructions);
  }

  timeline.push({
    type: "fullscreen",
    fullscreen_mode: true,
    message:
      "<p>To ensure you can see all relevant information, <br> we will switch to fullscreen mode when you press the button below. <br> <br></br> Please do not leave the fullscreen mode until the end of the experiment. Once it is over, you will see a prompt.</p>",
  });

  function getTaskStructure(task_config) {
    const taskName = task_config["task_name"];
    const nTrials = task_config["n_trials"];
    const nReferenceImages = task_config["n_reference_images"];

    let catchTrialIdxs;
    if ("catch_trial_idxs" in task_config) {
      catchTrialIdxs = task_config["catch_trial_idxs"];
    } else {
      catchTrialIdxs = [];
    }

    let maxReferenceImageIdsPerTrial = [];
    for (let i = 8; i > 8 - nReferenceImages; i--) {
      maxReferenceImageIdsPerTrial.push("max_" + i + ".png");
    }

    let trialIds = Array.from(Array(nTrials).keys()).map((x) => x + 1);

    let taskStructure = {
      trials: trialIds.map((trialId) => ({
        max_references: maxReferenceImageIdsPerTrial.map(
          (imageId) =>
            new URL(
              `${experimentName}/${taskName}/trials/trial_${trialId}/references/reference_${imageId}`,
              SERVER_TASKS_URL
            )
        ),
        base_query: new URL(
          `${experimentName}/${taskName}/trials/trial_${trialId}/queries/base.png`,
          SERVER_TASKS_URL
        ),
        max_query: new URL(
          `${experimentName}/${taskName}/trials/trial_${trialId}/queries/max.png`,
          SERVER_TASKS_URL
        ),
        min_query: new URL(
          `${experimentName}/${taskName}/trials/trial_${trialId}/queries/min.png`,
          SERVER_TASKS_URL
        ),
        id: trialId,
        catch_trial: catchTrialIdxs.includes(trialId),
      })),
      length: nTrials,
    };

    return taskStructure;
  }

  const main_task_structure = getTaskStructure(main_task_config);
  const demo_task_structure = getTaskStructure(demo_task_config);

  const main_task_trials = [].concat.apply([], main_task_structure["trials"]);
  const demo_task_trials = [].concat.apply([], demo_task_structure["trials"]);

  const main_task_images = [].concat.apply(
    [],
    main_task_trials.map((trial) =>
      [trial["max_query"], trial["min_query"], trial["base_query"]].concat(
        trial["max_references"]
      )
    )
  );
  const demo_task_images = [].concat.apply(
    [],
    demo_task_trials.map((trial) =>
      [trial["max_query"], trial["min_query"], trial["base_query"]].concat(
        trial["max_references"]
      )
    )
  );
  const images = main_task_images.concat(demo_task_images);

  // show demo only for if there are demo trials available
  if (demo_task_config["n_trials"] == 0) {
    noDemo = true;
  }

  if (!noDemo) {
    let demo_trials_timeline = [];
    addTrials(
      demo_task_structure["trials"],
      demo_trials_timeline,
      true,
      0.1,
      0.2
    );

    // the obvious trials refers to the trials that were hand-picked to be very easy
    // while the non-obvious trials were also hand-picked to be easy but are not as
    // easy as the other ones
    const n_obvious_demo_trials = demo_task_config["n_obvious_trials"];
    const obvious_demo_trials_variables =
      demo_trials_timeline[0].timeline_variables.slice(
        0,
        n_obvious_demo_trials
      );
    const non_obvious_demo_trials_variables =
      demo_trials_timeline[0].timeline_variables.slice(n_obvious_demo_trials);

    const n_obvious_trials_to_show = 1;
    const n_non_obvious_trials_to_show = 3;

    // create every possible permutation of the practice trials
    // but permute only the obvious and the non-obvious trials within their groups
    // but not across these
    let possible_demo_trials_timelines = permutator(
      obvious_demo_trials_variables
    )
      .map((obvs_vars) =>
        permutator(non_obvious_demo_trials_variables).map((non_obvs_vars) => [
          {
            timeline: demo_trials_timeline[0].timeline,
            timeline_variables: obvs_vars
              .slice(0, n_obvious_trials_to_show)
              .concat(non_obvs_vars.slice(0, n_non_obvious_trials_to_show))
              .map((it, idx) => ({
                ...it,
                progress:
                  0.1 +
                  ((0.2 - 0.1) /
                    (n_obvious_trials_to_show + n_non_obvious_trials_to_show)) *
                    (idx + 1),
              })),
          },
        ])
      )
      .flat();

    function createSwitchTimeline(items) {
      return [
        // create switch head that draws a random number
        {
          type: "call-function",
          func: () => getRandomInt(0, items.length - 1),
        },
      ].concat(
        // now test for each item whether the random value matches their index
        items.map((item, idx) => {
          return {
            timeline: item,
            conditional_function: () => {
              const data = jsPsych.data.getLastTimelineData();
              const random_value = data.values().last().value;
              return idx == random_value;
            },
          };
        })
      );
    }

    let demo_timeline = [
      {
        timeline: createSwitchTimeline(possible_demo_trials_timelines).concat([
          {
            timeline: [
              {
                type: "instructions",
                pages: [
                  "As you did not answer all trials correctly, we'd like you to repeat them.",
                ],
                show_clickable_nav: true,
                allow_backward: false,
                on_finish: function () {
                  jsPsych.setProgressBar(0.1);
                },
              },
            ],
            conditional_function: function () {
              let data = jsPsych.data.getLastTimelineData();
              return !data
                .filter({ trial_type: "cf-image-confidence-response" })
                .values()
                .every((i) => i.correct);
            },
          },
          {
            timeline: [
              {
                type: "instructions",
                pages: ["Great! Let's now start with the real trials!"],
                show_clickable_nav: true,
                allow_backward: false,
                key_forward: "Enter",
                button_label_next: "Continue (Enter)",
              },
            ],
            conditional_function: function () {
              let data = jsPsych.data.getLastTimelineData();
              // if the previous trial was the conditional instruction then data
              // will contain only this item
              return (
                data.count() > 1 &&
                data
                  .filter({ trial_type: "cf-image-confidence-response" })
                  .values()
                  .every((i) => i.correct)
              );
            },
            show_clickable_nav: true,
          },
        ]),
        loop_function: function (data) {
          return !data
            .filter({ trial_type: "cf-image-confidence-response" })
            .values()
            .every((i) => i.correct);
        },
      },
    ];
    timeline = timeline.concat(demo_timeline);
  }

  // The progress bar will end at 0.95 with the trials. This is supposed to prevent people from
  // closing the tab before the data is transferred.
  addTrials(main_task_structure["trials"], timeline, false, 0.2, 0.95);

  let feedback_trial = {
    type: "survey-text",
    questions: [
      {
        prompt: "Optional: Is there any feedback you'd like to share with us?",
        name: "feedback",
        rows: 8,
        columns: 80,
        required: false,
      },
    ],
  };
  timeline.push(feedback_trial);

  let sending_warning = {
    type: "instructions",
    pages: [
      "Press Continue to submit your data; this can take a moment. <br> Please do not close the window until we tell you to do so.",
    ],
    show_clickable_nav: true,
    allow_backward: false,
    key_forward: "Enter",
    button_label_next: "Continue (Enter)",
  };
  timeline.push(sending_warning);

  let send_response_payload = {
    type: "call-function",
    async: true,
    func: function (callback) {
      function sendMTurkPayload() {
        // send results back to MTurk
        const rawData = jsPsych.data.getAllData();
        const mainData = jsPsych.data
          .getLastTimelineData()
          .filter({ trial_type: "cf-image-confidence-response" });
        const rawPayload = rawData.json();
        const mainPayload = mainData.json();
        const json_data = JSON.stringify({
          main_data: mainPayload,
          raw_data: rawPayload,
          task_id: taskId,
        });

        window.opener.postMessage(json_data, window.opener.location.href);
      }

      let url = new URL(window.location.href);
      const noBouncerFlag = url.searchParams.get("nb");
      const noBouncer = noBouncerFlag !== null;

      if (noBouncer) {
        sendMTurkPayload();
        callback();
      } else {
        // send request to bouncer to make sure the worker cannot participate again
        let bouncer_url = new URL(
          "https://yourdomain.tld/mturk/bouncer/ban"
        );
        let url = new URL(window.location.href);
        let turk_info = jsPsych.turk.turkInfo();
        bouncer_url.searchParams.append("wid", turk_info.workerId);
        bouncer_url.searchParams.append("eid", url.searchParams.get("exp"));
        bouncer_url.searchParams.append("tns", url.searchParams.get("tns"));
        fetchJson(bouncer_url).finally(() => {
          sendMTurkPayload();
          callback();
        });
      }
    },
    on_finish: function () {
      jsPsych.setProgressBar(0.95);
    },
  };
  timeline.push(send_response_payload);

  let end = {
    type: "html-keyboard-response",
    stimulus:
      '<p style="color: white;">Your responses have been saved and submitted. </br></br>Thanks for your participation!</br></br>This window will automatically be closed in 5 seconds. Feel free to close it now already.</p>',
    choices: jsPsych.NO_KEYS,
    on_start: function (trial) {
      jsPsych.pluginAPI.setTimeout(function () {
        window.close();
      }, 5000);
    },
    on_finish: function () {
      jsPsych.setProgressBar(1.0);
    },
  };
  timeline.push(end);

  jsPsych.init({
    timeline: timeline,
    exclusions: {
      min_width: 940,
      min_height: 600,
    },
    preload_images: images.concat(instructionImages),
    show_preload_progress_bar: true,
    show_progress_bar: true,
    auto_update_progress_bar: false,
  });
}
