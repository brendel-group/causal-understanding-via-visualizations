/**
 * jspsych-cf-image-confidence-response
 * Roland Zimmermann
 *
 * plugin for displaying two query stimuli and a set of reference images and getting a confidence button response
 *
 **/

jsPsych.plugins["cf-image-confidence-response"] = (function () {
  var plugin = {};

  jsPsych.pluginAPI.registerPreload(
    "cf-image-confidence-response",
    "query_a_stimulus",
    "image"
  );
  jsPsych.pluginAPI.registerPreload(
    "cf-image-confidence-response",
    "query_b_stimulus",
    "image"
  );
  jsPsych.pluginAPI.registerPreload(
    "cf-image-confidence-response",
    "query_base_stimulus",
    "image"
  );

  jsPsych.pluginAPI.registerPreload(
    "cf-image-confidence-response",
    "reference_stimuli",
    "image"
  );

  plugin.info = {
    name: "cf-image-confidence-response",
    description: "",
    parameters: {
      query_a_stimulus: {
        type: jsPsych.plugins.parameterType.IMAGE,
        pretty_name: "Minimal query stimulus",
        default: undefined,
        description: "The minimal query image to be displayed",
      },
      query_b_stimulus: {
        type: jsPsych.plugins.parameterType.IMAGE,
        pretty_name: "Maximal query stimulus",
        default: undefined,
        description: "The maximal query image to be displayed",
      },
      reference_stimuli: {
        type: jsPsych.plugins.parameterType.IMAGE,
        pretty_name: "Reference stimuli",
        default: undefined,
        array: true,
        description: "The reference images to be displayed on the left side",
      },
      reference_title: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: "Left reference stimuli title",
        default: undefined,
        description: "The title of the reference images shown on the left side",
      },
      query_base_title: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: "Left reference stimuli title",
        default: undefined,
        description: "The title of the reference images shown on the left side",
      },
      query_title: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: "Left reference stimuli title",
        default: undefined,
        description: "The title of the reference images shown on the left side",
      },
      stimulus_height: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: "Image height",
        default: null,
        description: "Set the image height in pixels",
      },
      stimulus_width: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: "Image width",
        default: null,
        description: "Set the image width in pixels",
      },
      maintain_aspect_ratio: {
        type: jsPsych.plugins.parameterType.BOOL,
        pretty_name: "Maintain aspect ratio",
        default: true,
        description: "Maintain the aspect ratio after setting width or height",
      },
      choices: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: "Choices",
        default: undefined,
        array: true,
        description: "The labels for the buttons.",
      },
      randomize_queries: {
        type: jsPsych.plugins.parameterType.BOOL,
        pretty_name: "Randomize Queries",
        default: false,
        description: "Randomly switch query a and query b stimulus",
      },
      correct_query_choice: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: "Is first query a or b correct?",
        default: null,
        description: "",
      },
      button_html: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: "Button HTML",
        default: '<button class="jspsych-btn">%choice%</button>',
        array: true,
        description: "The html of the button. Can create own style.",
      },
      prompt: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: "Prompt",
        default: null,
        description: "Any content here will be displayed under the button.",
      },
      stimulus_duration: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: "Stimulus duration",
        default: null,
        description: "How long to hide the stimulus.",
      },
      trial_duration: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: "Trial duration",
        default: null,
        description: "How long to show the trial.",
      },
      feedback_delay_duration: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: "Feedback delay duration",
        default: null,
        description:
          "How long to wait before showing feedback at the end of the trial.",
      },
      feedback_duration: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: "Feedback duration",
        default: null,
        description: "How long to show feedback at the end of the trial.",
      },
      response_ends_trial: {
        type: jsPsych.plugins.parameterType.BOOL,
        pretty_name: "Response ends trial",
        default: true,
        description: "If true, then trial will end when user responds.",
      },
      render_on_canvas: {
        type: jsPsych.plugins.parameterType.BOOL,
        pretty_name: "Render on canvas",
        default: false,
        description:
          "If true, the image will be drawn onto a canvas element (prevents blank screen between consecutive images in some browsers)." +
          "If false, the image will be shown via an img element.",
      },
      correct_text: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: "Correct text",
        default: "<p class='feedback'>Correct</p>",
        description: "String to show when correct answer is given.",
      },
      incorrect_text: {
        type: jsPsych.plugins.parameterType.STRING,
        pretty_name: "Incorrect text",
        default: "<p class='feedback'>Incorrect</p>",
        description: "String to show when incorrect answer is given.",
      },
    },
  };

  plugin.trial = function (display_element, trial) {
    var height, width;
    var html;

    if (trial.render_on_canvas) {
      // first clear the display element (because the render_on_canvas method appends to display_element instead of overwriting it with .innerHTML)
      if (display_element.hasChildNodes()) {
        // can't loop through child list because the list will be modified by .removeChild()
        while (display_element.firstChild) {
          display_element.removeChild(display_element.firstChild);
        }
      }
      // create canvas element and image
      var canvas = document.createElement("canvas");
      canvas.id = "jspsych-image-button-response-stimulus";
      canvas.style.margin = 0;
      canvas.style.padding = 0;
      var img = new Image();
      img.src = trial.stimulus;
      // determine image height and width
      if (trial.stimulus_height !== null) {
        height = trial.stimulus_height;
        if (trial.stimulus_width == null && trial.maintain_aspect_ratio) {
          width =
            img.naturalWidth * (trial.stimulus_height / img.naturalHeight);
        }
      } else {
        height = img.naturalHeight;
      }
      if (trial.stimulus_width !== null) {
        width = trial.stimulus_width;
        if (trial.stimulus_height == null && trial.maintain_aspect_ratio) {
          height =
            img.naturalHeight * (trial.stimulus_width / img.naturalWidth);
        }
      } else if (
        !((trial.stimulus_height !== null) & trial.maintain_aspect_ratio)
      ) {
        // if stimulus width is null, only use the image's natural width if the width value wasn't set
        // in the if statement above, based on a specified height and maintain_aspect_ratio = true
        width = img.naturalWidth;
      }
      canvas.height = height;
      canvas.width = width;
      // create buttons
      var buttons = [];
      if (Array.isArray(trial.button_html)) {
        if (trial.button_html.length == trial.choices.length) {
          buttons = trial.button_html;
        } else {
          console.error(
            "Error in image-button-response plugin. The length of the button_html array does not equal the length of the choices array"
          );
        }
      } else {
        for (var i = 0; i < trial.choices.length; i++) {
          buttons.push(trial.button_html);
        }
      }
      var btngroup_div = document.createElement("div");
      btngroup_div.id = "jspsych-image-button-response-btngroup";
      html = "";
      for (var i = 0; i < trial.choices.length; i++) {
        var str = buttons[i].replace(/%choice%/g, trial.choices[i]);
        html += `<div class="jspsych-image-button-response-button" style="display: inline-block; margin:'${trial.margin_vertical} ${trial.margin_horizontal}" id="jspsych-image-button-response-button-${i}" data-choice=${i}">'+str+'</div>`;
      }
      btngroup_div.innerHTML = html;
      // add canvas to screen and draw image
      display_element.insertBefore(canvas, null);
      var ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0, width, height);
      // add buttons to screen
      display_element.insertBefore(btngroup_div, canvas.nextElementSibling);
      // add prompt if there is one
      if (trial.prompt !== null) {
        display_element.insertAdjacentHTML("beforeend", trial.prompt);
      }
    } else {
      // before we do anything else, randomly shuffle queries if necessary
      // will contain the correct item in the *shuffled* basis which does not correspond to the true data
      var correct_query;
      var first_query_stimulus;
      var second_query_stimulus;
      var switched_queries = false;

      if (trial.randomize_queries && Math.random() > 0.5) {
        first_query_stimulus = trial.query_b_stimulus;
        second_query_stimulus = trial.query_a_stimulus;
        correct_query = trial.correct_query_choice == "a" ? "b" : "a";
        switched_queries = true;
      } else {
        first_query_stimulus = trial.query_a_stimulus;
        second_query_stimulus = trial.query_b_stimulus;
        correct_query = trial.correct_query_choice;
      }

      let reference_title_html = "";
      let query_base_title_html = "";
      let query_title_html = "";
      if (trial.reference_title) {
        reference_title_html = `<p>${trial.reference_title}</p>`;
      }
      if (trial.query_base_title) {
        query_base_title_html = `<p>${trial.query_base_title}</p>`;
      }
      if (trial.query_title) {
        query_title_html = `<p>${trial.query_title}</p>`;
      }

      const numReferenceAColumns = Math.ceil(
        Math.sqrt(trial.reference_stimuli.length)
      );

      // display stimulus as an image element
      let reference_stimuli_html = `<div class="jspsych-cf-image-confidence-reference-grid" style="grid-template-columns: repeat(${numReferenceAColumns}, auto);">${trial.reference_stimuli
        .map((stim) => `<img class="jspsych-cf-image-stimulus" src="${stim}"/>`)
        .join(" ")}</div>`;
      let query_base_stimulus_html = `<div class="jspsych-cf-image-confidence-reference-grid" style="grid-template-columns: auto;"><img class="jspsych-cf-image-stimulus" src="${trial.query_base_stimulus}"/></div>`;

      reference_stimuli_html = `<div class="jspsych-cf-image-confidence-reference-container">${
        numReferenceAColumns == 0
          ? ""
          : reference_title_html + " " + reference_stimuli_html
      }</div>`;
      query_base_stimulus_html = `<div id="jspsych-cf-image-confidence-base-query-container" class="jspsych-cf-image-confidence-base-query-container">${query_base_title_html} ${query_base_stimulus_html}</div>`;

      let buttons = [];
      if (Array.isArray(trial.button_html)) {
        if (trial.button_html.length == trial.choices.length) {
          buttons = trial.button_html;
        } else {
          console.error(
            "Error in image-button-response plugin. The length of the button_html array does not equal the length of the choices array"
          );
        }
      } else {
        for (let i = 0; i < trial.choices.length; i++) {
          buttons.push(trial.button_html);
        }
      }

      var numChoicesPerQuery = Math.round(trial.choices.length / 2);

      const buttons_a = buttons.slice(numChoicesPerQuery);
      const buttons_b = buttons.slice(numChoicesPerQuery, buttons.length);

      const confidence_hint_a =
        '<div class="jspsych-cf-image-confidence-choice-hint"><div style="width: 100%; max-width: 224px;"><div class="arrow-left"><div></div><div></div></div><p style="font-size: 16px; font-family: \'Open Sans Condensed\', \'Arial\', sans-serif;">More Confident</p></div></div>';
      const confidence_hint_b =
        '<div class="jspsych-cf-image-confidence-choice-hint"><div style="width: 100%; max-width: 224px;"><div class="arrow-right"><div></div><div></div></div><p style="font-size: 16px; font-family: \'Open Sans Condensed\', \'Arial\', sans-serif;">More Confident</p></div></div>';
      const buttons_a_html = buttons_a.map(
        (str, i) =>
          `<div class="jspsych-image-button-response-button" style="display: inline-block;" id="jspsych-image-button-response-button-${i}" data-choice="${i}">${str.replace(
            /%choice%/g,
            trial.choices[i]
          )}</div>`
      );
      const buttons_b_html = buttons_b.map(
        (str, i) =>
          `<div class="jspsych-image-button-response-button" style="display: inline-block;" id="jspsych-image-button-response-button-${
            i + buttons_a.length
          }" data-choice="${i + buttons_a.length}">${str.replace(
            /%choice%/g,
            trial.choices[i + buttons_a.length]
          )}</div>`
      );
      const choice_buttons_a_div_html = `<div id="jspsych-image-button-response-btngroup" class="jspsych-cf-image-confidence-choice-grid" style="grid-template-columns: repeat(${numChoicesPerQuery}, 1fr);">${buttons_a_html.join(
        " "
      )} ${confidence_hint_a}</div>`;
      const choice_buttons_b_div_html = `<div id="jspsych-image-button-response-btngroup" class="jspsych-cf-image-confidence-choice-grid" style="grid-template-columns: repeat(${numChoicesPerQuery}, 1fr);">${buttons_b_html.join(
        " "
      )} ${confidence_hint_b}</div>`;
      const query_choice_a_container_html = `<div><img class="jspsych-cf-image-stimulus jspsych-cf-image-confidence-query" id="jspsych-cf-image-confidence-query-a" src="${first_query_stimulus}"/> ${choice_buttons_a_div_html}</div>`;
      const query_choice_b_container_html = `<div><img class="jspsych-cf-image-stimulus jspsych-cf-image-confidence-query" id="jspsych-cf-image-confidence-query-b" src="${second_query_stimulus}"/> ${choice_buttons_b_div_html}</div>`;
      const query_stimuli_html = `<div class="jspsych-cf-image-confidence-query-choice-container">${query_title_html} <div>${query_choice_a_container_html} ${query_choice_b_container_html}</div> </div>`;

      const right_container_html = `<div class="jspsych-cf-image-confidence-query-container">${query_base_stimulus_html} ${query_stimuli_html} </div>`;

      let confidence_container_style = `grid-template-columns: ${numReferenceAColumns}fr 50px 2fr!important;`;
      confidence_container_style = confidence_container_style.replace(
        "0fr",
        "0"
      );
      html = `<div class="jspsych-cf-image-confidence-container" style="${confidence_container_style}"> ${reference_stimuli_html} ${right_container_html}</div>`;

      if (trial.prompt !== null) {
        html = `<div style="text-align: center">${trial.prompt}</div>` + html;
      }

      // update the page content
      display_element.innerHTML = html;
    }

    // start timing
    var start_time = performance.now();

    function button_pressed(e) {
      const choice_idx =
        e.currentTarget.parentElement.getAttribute("data-choice"); // don't use dataset for jsdom compatibility
      const choice = e.currentTarget.innerText;
      after_response(choice, choice_idx);
      e.currentTarget.removeEventListener("click", button_pressed);
    }

    for (let i = 0; i < trial.choices.length; i++) {
      display_element
        .querySelector(`#jspsych-image-button-response-button-${i} > button`)
        .addEventListener("click", button_pressed);
    }

    // store response
    var response = {
      rt: null,
      button: null,
    };

    // function to handle responses by the subject
    function after_response(choice, choiceIdx) {
      // measure rt
      var end_time = performance.now();
      var rt = end_time - start_time;
      response.button = parseInt(choiceIdx);
      response.rt = rt;
      response.switched_queries = switched_queries;
      if (switched_queries) {
        response.choice = choiceIdx < numChoicesPerQuery ? "b" : "a";
      } else {
        response.choice = choiceIdx < numChoicesPerQuery ? "a" : "b";
      }
      response.confidence = (choiceIdx % (trial.choices.length / 2)) + 1;

      function maybe_show_feedback_and_maybe_end_trial() {
        if (
          (trial.feedback_duration !== undefined) &
          (trial.feedback_duration > 0)
        ) {
          if (correct_query == "a") {
            display_element.querySelector(
              "#jspsych-cf-image-confidence-query-a"
            ).className += " jspsych-cf-image-confidence-query-correct";
          } else if (correct_query == "b") {
            display_element.querySelector(
              "#jspsych-cf-image-confidence-query-b"
            ).className += " jspsych-cf-image-confidence-query-correct";
          }

          let feedback_text = "";
          if (response.choice == trial.correct_query_choice) {
            feedback_text = trial.correct_text;
          } else {
            feedback_text = trial.incorrect_text;
          }
          display_element.querySelector(
            ".jspsych-cf-image-confidence-query-container"
          ).innerHTML += feedback_text;
        }

        if (trial.response_ends_trial) {
          if (
            (trial.feedback_duration !== undefined) &
            (trial.feedback_duration > 0)
          ) {
            jsPsych.pluginAPI.setTimeout(function () {
              end_trial();
            }, trial.feedback_duration);
          } else {
            end_trial();
          }
        } else {
          if (trial.feedback_duration !== undefined) {
            jsPsych.pluginAPI.setTimeout(function () {
              // add continue button
              //display_element.querySelector('.jspsych-cf-image-confidence-choice-hint').style = "display: flex; vertica-align: center;";

              //display_element.querySelector('#jspsych-cf-image-confidence-base-query-container').innerHTML = '<button id="jspsych-image-button-continue" class="jspsych-btn">Continue</button>';
              //display_element.querySelector('#jspsych-cf-image-confidence-base-query-container').addEventListener('click', function(e){
              //    end_trial();
              //});
              const elements = display_element.querySelectorAll(
                ".jspsych-cf-image-confidence-choice-hint"
              );
              // left
              elements[0].innerHTML = "";
              elements[0].style.height = "0px";
              // right
              elements[1].innerHTML = "";
              elements[1].style.height = "0px";
              display_element.querySelector(
                ".jspsych-cf-image-confidence-query-container"
              ).innerHTML +=
                '<div><button style="margin-top: 5px;" id="jspsych-image-button-continue" class="jspsych-btn">Continue</button></div>';
              display_element
                .querySelector("#jspsych-image-button-continue")
                .addEventListener("click", function (e) {
                  end_trial();
                });
            }, trial.feedback_duration);
          }
        }
      }

      // after a valid response, the stimulus will have the CSS class 'responded'
      // which can be used to provide visual feedback that a response was recorded
      if (response.button < numChoicesPerQuery) {
        display_element.querySelector(
          "#jspsych-cf-image-confidence-query-a"
        ).className += " jspsych-cf-image-confidence-query-responded";
      } else {
        display_element.querySelector(
          "#jspsych-cf-image-confidence-query-b"
        ).className += " jspsych-cf-image-confidence-query-responded";
      }

      // disable all the buttons after a response
      var btns = document.querySelectorAll(
        ".jspsych-image-button-response-button button"
      );
      for (var i = 0; i < btns.length; i++) {
        //btns[i].removeEventListener('click');
        btns[i].setAttribute("disabled", "disabled");
      }

      jsPsych.pluginAPI.setTimeout(function () {
        maybe_show_feedback_and_maybe_end_trial();
      }, trial.feedback_delay_duration);
    }

    // function to end trial when it is time
    function end_trial() {
      // kill any remaining setTimeout handlers
      jsPsych.pluginAPI.clearAllTimeouts();

      // gather the data to store for the trial
      const trial_data = {
        rt: response.rt,
        stimulus: trial.stimulus,
        button_pressed: response.button,
        switched_queries: response.switched_queries,
        confidence: response.confidence,
        choice: response.choice,
        correct: response.choice == trial.correct_query_choice,
      };

      // clear the display
      display_element.innerHTML = "";

      // move on to the next trial
      jsPsych.finishTrial(trial_data);
    }

    // hide image if timing is set
    if (trial.stimulus_duration !== null) {
      jsPsych.pluginAPI.setTimeout(function () {
        display_element.querySelector(
          "#jspsych-image-button-response-stimulus"
        ).style.visibility = "hidden";
      }, trial.stimulus_duration);
    }

    // end trial if time limit is set
    if (trial.trial_duration !== null) {
      jsPsych.pluginAPI.setTimeout(function () {
        end_trial();
      }, trial.trial_duration);
    } else if (trial.response_ends_trial === false) {
      console.warn(
        "The experiment may be deadlocked. Try setting a trial duration or set response_ends_trial to true."
      );
    }
  };

  return plugin;
})();
