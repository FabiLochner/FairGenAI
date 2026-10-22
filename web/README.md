## Simple GenAI web service that enforces worst-case fairness

This folder contains a simple webservice for image synthesis where fairness can be enforced. It is similar to that of ChatGPT + DALLE.3 where internally, an LLM generates the description, followed by calling an image synthesis tool.

Note that as the development of this tool is for concept validation purposes, the models being used are not of high quality. Consequently, the generated images can be ugly when comparing with commercial tools. 

### A. Installation

The following has been tested with a new conda environment.

Apart from [pytorch](https://pytorch.org/get-started/locally/) and [ollama](https://ollama.com/download), please install the following with pip: 

`pip install flask flask_cors llama-index transformers einops pillow diffusers llama-index-llms-ollama`

Have VS Code and install Live Server as extension.



### B. Execute the Application


1. Start the Flask Server: Open a terminal, navigate to the directory containing server.py, and run (or run from VS Code)


    $ python server.py

    * With the current configuration, a GPU with 8GB VRAM would be sufficient to run. If you need all computation done with CPU, modify the [config.json](config.json) file. 

    * Note that some of the models will be downloaded (e.g., gemma2:2b, moondream2), so be prepared. 

2. Start the LiveServer: Open VS Code and start the Live Server to serve your index.html.

    * Important: Disable LiveServer automatic reload of the server upon file change [(link)](https://stackoverflow.com/questions/77120592/how-can-i-prevent-page-reloads-upon-saving-changes-with-the-live-server-extensio), as when the image is generated and stored, without disabling, the reloading will simply erase the display. 

3. Access the application: Open your browser and navigate to the URL provided by the Live Server (e.g., http://127.0.0.1:5500/index.html).

4. Test the application: Each time you submit the form, the application should call the LLM, generate an image using Stable Diffusion, and display both the text response and the generated image as a link. Without the story step (see C.3), the text response only notes that the prompt was sent directly to the image model.

    * In the form, enter a prompt related to the "conditioned_value" of the active setup (see C.2), e.g. "office cleaner" for "gender_cleaning_occupations" or "successful business leader" for "successful_business_leader". Only related prompts are monitored and, if needed, enforced; the topic is changed by selecting or adding a setup in config.json.

5. When fairness is enabled, it will explitly notify the user that fairness constraints enforce the generation to be "male" or "female" (for gender fairness), and inform user that they can make the gender explicit, in order not to be enforced. 

### C. Modification & extension

#### C.1 Configuration (config.json)

config.json answers three questions. The first two are answered by a name that selects one entry:

| Question | Selected by | Defined in |
|---|---|---|
| What is enforced? | "active_setup" | "fairness_enforcement" (C.2) |
| Which models run? | "active_pipeline" | "pipelines", which refer to "model_catalog" (C.3, C.4) |
| Which hardware is used? | – | "diffusion_GPU", "image_interpretation_GPU" (C.5) |

Setups and pipelines are independent: any setup can run with any pipeline. The keys "active_setup", "fairness_enforcement", "active_pipeline", "pipelines" and "model_catalog" are required; if one is missing, the server stops at start-up with a message naming it. Each request is logged as described in C.6.

#### C.2 Enforcement setups

Each entry under "fairness_enforcement" is a named setup with:

* "conditioned_value": the topic for which fairness is enforced, e.g. "a person working in a cleaning occupation (e.g., office cleaner, building cleaner)". For every prompt, the gate LLM checks whether it is related to this topic and whether it already asks for a group value; only related prompts without one are monitored.
* "fairness_group": the dimension, "gender" or "demographics".
* Either "beta": the original method of the paper. Every group value must appear at least once in every "beta" monitored images; "beta" must be at least the number of group values.
* Or "target": the target-based extension (targets.py). "mode" selects the target distribution: "parity" (equal shares), "empirical" (a reference distribution such as labour statistics, under "reference": {"source", "distribution"}) or "maximin" (under "maximin_distribution"). Each group value gets its own deadline, 1 / target share rounded down; values whose share is too large for a deadline are left to the image model. The server prints the resulting deadlines at start-up.

#### C.3 Models and pipelines

* "model_catalog": every model is defined once, grouped by role ("image_generation", "llm", "image_interpretation"), with a "description", a "model_id" and optionally a "revision".
* "pipelines": named model combinations. Each pipeline names one catalog entry for "image_generation", "gate_llm" (checks whether a prompt is related and already asks for a group value), "story_llm" (writes the description passed to the image model) and "image_interpretation". "story_llm": null turns the story step off: the user prompt (plus the enforcement clause) goes directly to the image model, while the gate still uses "gate_llm".
* "active_pipeline": the pipeline to run.

Four pipelines are predefined. Together they compare two image models with and without the story step; all other models stay the same:

| Pipeline | Image model | Story step |
|---|---|---|
| "cheng2025_baseline" | Stable Diffusion 1.5 | yes |
| "cheng2025_baseline_no_story" | Stable Diffusion 1.5 | no |
| "cheng2025_with_sd_3_5_medium" | Stable Diffusion 3.5 Medium | yes |
| "cheng2025_with_sd_3_5_medium_no_story" | Stable Diffusion 3.5 Medium | no |

"cheng2025_baseline" is the model combination of the original paper and repo: Stable Diffusion 1.5, Gemma 2 2B via Ollama for gate and story, and moondream2 at revision 2024-07-23.

Stable Diffusion 3.5 Medium needs more resources than the baseline:

* Access: the model is gated. Create a Hugging Face account, accept the Stability AI Community License on the [model page](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium), create an access token of type "Read" and run `huggingface-cli login` once on the machine that runs the server.
* Disk: about 16 GB for the download.
* Memory: about 21 GB peak, mostly for the T5-XXL text encoder. Use at least 24 GB of GPU memory, or at least 32 GB of unified memory on Apple Silicon.

To test another model, add it to the catalog once and add a new pipeline that refers to it; existing pipelines stay unchanged.

#### C.4 Image generation settings

Image generation models in the catalog take two optional fields:

* "torch_dtype": "float16", "bfloat16" or "float32" (default "float16", as hard-coded in the original server.py).
* "generation": settings passed to every image generation call, e.g. "num_inference_steps" and "guidance_scale". Without it, the defaults of the diffusers pipeline class apply. Settings the pipeline does not accept stop the server at start-up.

"settings_source" documents where these values come from; like "description", it is ignored at runtime and not logged. For "sd_1_5", the values are those the original repo used implicitly.

#### C.5 Hardware

* "diffusion_GPU": true runs the image model on the Apple GPU (mps) if available, otherwise on CUDA; false runs it on the CPU.
* "image_interpretation_GPU": true runs the vision model on CUDA (NVIDIA GPUs only; keep it false on a Mac); false runs it on the CPU.

A missing flag counts as false.

#### C.6 Request log

Every request is appended as one JSON line to generated_images/enforcement_log.jsonl, across server runs. Each line contains:

* Run and setup: "run_id" (server start time), "timestamp", "setup", "mode" ("original" for setups with "beta", otherwise the target mode).
* Models: "pipeline" and "models" (per role: name, model_id and, if set, revision, torch_dtype and generation), so runs with different model combinations can be told apart in the same file.
* Gate: "user_prompt", "gate_related", "gate_group_specified", "monitored".
* Enforcement and detection: "enforced", "vision_person_answer", "vision_answer", "counted_as".
* Counters: "counters_before", "counters_after".
* Output: "image_prompt" (the exact prompt sent to the image model) and "image_file" (in generated_images/).

Lines written before a field was introduced do not contain it; lines without "pipeline" were generated with the models of "cheng2025_baseline".