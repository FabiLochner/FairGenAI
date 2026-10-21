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

4. Test the application: Each time you submit the form, the application should call the LLM, generate an image using Stable Diffusion, and display both the text response and the generated image as a link.

    * In the form, please enter "successful business leader" or of similar kind, as this is the only item now being enforced for fairness. One can change it in the config.json to items such as "bad cook".  

5. When fairness is enabled, it will explitly notify the user that fairness constraints enforce the generation to be "male" or "female" (for gender fairness), and inform user that they can make the gender explicit, in order not to be enforced. 

### C. Modification & extension

One shall modify the config.json for the condition where fairness shall be enforced (variable: "conditioned_value"), the dimensions (variable: fairness_group with options "gender" or "demographics"), as well as the worst case bound (variable: "beta").

#### Models

Which models run is chosen independently of the enforcement setup, in three parts of config.json:

* "model_catalog": every model is defined once, grouped by role ("image_generation", "llm", "image_interpretation"), with a "description", a "model_id" and optionally a "revision".
* "pipelines": named model combinations. Each pipeline names one catalog entry for "image_generation", "gate_llm" (checks whether a prompt is related and already asks for a group value), "story_llm" (writes the description passed to the image model) and "image_interpretation".
* "active_pipeline": the pipeline to run.

"cheng2025_baseline" is the model combination of the original paper and repo. To test another model, add it to the catalog once and add a new pipeline that refers to it; existing pipelines stay unchanged. Without "active_pipeline", the server runs "cheng2025_baseline".

Elaborate extensions are left for future work. 