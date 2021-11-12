SERVER_TASKS_URL = "https://yourdomain.tld/mturk-data/experiments/";

function initialize() {
  const debug = new URL(window.location.href).searchParams.get("debug");
  const debug_mode = debug !== null;
  if (debug_mode) {
    runExperiment();
  } else {
    if (window.opener == null) {
      alert("Please start tasks only via the Start page.");
      window.location = "start.html";
    } else {
      // check whether the user has access to the task
      runExperiment();
    }
  }
}

function runExperiment(type) {
  // get our internal ID to load the correct images
  let url = new URL(window.location.href);
  let taskId = url.searchParams.get("tid");
  let taskNamespace = url.searchParams.get("tns");
  let experimentName = url.searchParams.get("exp");

  let noInstructions = url.searchParams.get("ni");
  noInstructions = noInstructions !== null ? noInstructions == "true" : false;

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
      callback(...tasks, taskId, experimentName, noInstructions)
    );
}

function prepareExperiment(
  main_task_config,
  demo_task_config,
  taskId,
  experimentName,
  noInstructions
) {
  let timeline = [];

  // set random generator
  Math.seedrandom(main_task_config["task_name"]);

  function addTrials(trials, timeline, is_demo, start_progress, end_progress) {
    let task_timeline = [
      {
        type: "2afc-image-confidence-response",
        query_a_stimulus: jsPsych.timelineVariable("min_query"),
        query_b_stimulus: jsPsych.timelineVariable("max_query"),
        reference_a_stimuli: jsPsych.timelineVariable("min_references"),
        reference_b_stimuli: jsPsych.timelineVariable("max_references"),
        choices: ["1", "2", "3", "1", "2", "3"],
        prompt:
          "<p>Which image at the center matches the <b>Right Examples</b> better?</p>",
        correct_text: "",
        reference_a_title: "Left Examples",
        reference_b_title: "Right Examples",
        incorrect_text: "",
        feedback_delay_duration: 350,
        feedback_duration: 1150,
        randomize_queries: true,
        response_ends_trial: false,
        correct_query_choice: "b",
        data: {
          id: jsPsych.timelineVariable("task_id"),
          min_query: jsPsych.timelineVariable("min_query"),
          max_query: jsPsych.timelineVariable("max_query"),
          min_references: jsPsych.timelineVariable("min_references"),
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
          min_references: jsPsych.randomization.shuffle(trial.min_references),
          max_references: jsPsych.randomization.shuffle(trial.max_references),
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

    instructionImages = Array.from({ length: 12 }, (_, i) => i + 1).map(
      (i) =>
        new URL(
          `${experimentName}/${
            main_task_config["task_name"].split("/")[0]
          }/instructions/${i}.jpg`,
          SERVER_TASKS_URL
        )
    );

    let instructions = {
      timeline: [
        {
          type: "instructions",
          pages: [
            "In this experiment, you will be shown images on the screen and <br> asked to make a response by clicking your mouse.",
            "The experiment consists of multiple responses like this. <br> We will now explain to you how a single trial works.",
            `<br><br>On the left and the right side of the screen, you see two groups of example images. <br> They are labeled <b>Left Examples</b> and <b>Right Examples</b>.  <br> <img id="jspsych-instructions-image" src="${instructionImages[0]}" />`,
            `<br><br>At the center of the screen, you see two more images. <br> While one image belongs to the Left Examples, the other one belongs to the Right Examples. <br> <img id="jspsych-instructions-image" src="${instructionImages[1]}" />`,
            `<br><br>The question you have to answer is always the following: <br> Which image at the center matches the <b>Right Examples</b> better? <br> <img id="jspsych-instructions-image" src="${instructionImages[2]}" />`,
            `<br><br>Here is how you answer:<br>Below the upper center image you see a row of numbers. <br> <img id="jspsych-instructions-image" src="${instructionImages[3]}" />`,
            `<br><br><br> Above the lower center image you also see a row of numbers. <br> <img id="jspsych-instructions-image" src="${instructionImages[4]}" />`,
            `<br><br> If you think the <b>upper</b> image better matches the Right Examples, <br> choose a number from the <b>upper</b> row of numbers. <br> <img id="jspsych-instructions-image" src="${instructionImages[5]}" />`,
            `<br><br> If you think the <b>lower</b> image better matches the Right Examples, <br> choose a number from the <b>lower</b> row of numbers. <br> <img id="jspsych-instructions-image" src="${instructionImages[6]}" />`,
            `<br>The value of the number indicates how confident you are in your choice: <br> The higher the number, the higher your confidence. <br> If you are not sure, go with your best guess.  <br> <img id="jspsych-instructions-image" src="${instructionImages[7]}" />`,
            `<br><br><br>Once you provided your answer, a black frame appears around your chosen image.  <br> <img id="jspsych-instructions-image" src="${instructionImages[8]}" />`,
            `<br><br>Finally, you receive feedback: <br>A green frame appears around the image that truly belongs to the Right Examples.  <br> <img id="jspsych-instructions-image" src="${instructionImages[9]}" />`,
            `<br>This is the end of one trial. <br> Please note that each trial is independent of all other trials. <br> This means that you cannot transfer from one to another trial. <br> <img id="jspsych-instructions-image" src="${instructionImages[9]}" />`,
            `<br><br><br>By clicking on the <it>Continue</it> button you continue to the next trial. <br> <img id="jspsych-instructions-image" src="${instructionImages[10]}" />`, //TODO: change screenshot
            `<br>This is the last opportunity to go back and re-read the instructions via the <it>Previous</it> button. <br>Otherwise, we will start with a couple demo trials <br>so that you can familiarize yourself with the experiment. <br> <img id="jspsych-instructions-image" src="${instructionImages[11]}" />`,
          ],
          images: [null, null].concat(instructionImages),
          show_clickable_nav: true,
          on_finish: function () {
            jsPsych.setProgressBar(0.05);
          },
        },
      ],
    };
    timeline.push(instructions);
  }

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
    let minReferenceImageIdsPerTrial = [];
    for (let i = 8; i > 8 - nReferenceImages; i--) {
      maxReferenceImageIdsPerTrial.push("max_" + i + ".png");
      minReferenceImageIdsPerTrial.push("min_" + (i + 1) + ".png");
    }

    let trialIds = Array.from(Array(nTrials).keys()).map((x) => x + 1);

    let taskStructure = {
      trials: trialIds.map((trialId) => ({
        max_references: maxReferenceImageIdsPerTrial.map(
          (imageId) =>
            new URL(
              `${experimentName}/${taskName}/trials/trial_${trialId}/references/${imageId}`,
              SERVER_TASKS_URL
            )
        ),
        min_references: minReferenceImageIdsPerTrial.map(
          (imageId) =>
            new URL(
              `${experimentName}/${taskName}/trials/trial_${trialId}/references/${imageId}`,
              SERVER_TASKS_URL
            )
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
      [trial["max_query"], trial["min_query"]].concat(
        trial["max_references"],
        trial["min_references"]
      )
    )
  );
  const demo_task_images = [].concat.apply(
    [],
    demo_task_trials.map((trial) =>
      [trial["max_query"], trial["min_query"]].concat(
        trial["max_references"],
        trial["min_references"]
      )
    )
  );
  const images = main_task_images.concat(demo_task_images);

  let demo_trials_timeline = [];
  addTrials(
    demo_task_structure["trials"],
    demo_trials_timeline,
    true,
    0.1,
    0.2
  );

  let demo = {
    timeline: [
      {
        timeline: demo_trials_timeline.concat([
          {
            type: "instructions",
            pages: ["Great! Let's now start with the real trials!"],
            show_clickable_nav: true,
          },
        ]),
      },
    ],
    on_finish: function () {
      jsPsych.setProgressBar(0.2);
    },
  };
  timeline.push(demo);

  timeline.push({
    type: "fullscreen",
    fullscreen_mode: true,
    message:
      "<p>The experiment will switch to full screen mode when you press the button below.</p>",
  });

  addTrials(main_task_structure["trials"], timeline, false, 0.2, 1.0);

  let send_response_payload = {
    type: "call-function",
    async: true,
    func: function (callback) {
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
        // send results back to MTurk
        const rawData = jsPsych.data.getAllData();
        const mainData = jsPsych.data
          .getLastTimelineData()
          .filter({ trial_type: "2afc-image-confidence-response" });
        const rawPayload = rawData.json();
        const mainPayload = mainData.json();
        const json_data = JSON.stringify({
          main_data: mainPayload,
          raw_data: rawPayload,
          task_id: taskId,
        });

        window.opener.postMessage(json_data, window.opener.location.href);
        callback();
      });
    },
    on_finish: function () {
      jsPsych.setProgressBar(1.0);
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
