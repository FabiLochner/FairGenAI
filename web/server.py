from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import uuid
import os

from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

import numpy as np
from llama_index.llms.ollama import Ollama

from diffusers import DiffusionPipeline

import torch
import numpy as np
import random

import json
import sys
import time  # timestamps and run id for the request log

# Target-based enforcement: target distribution -> deadlines (beta per group value) -> scheduler
from targets import build_plan, DeadlineScheduler

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set the seed
set_seed(42)

# Load config file
with open('config.json', 'r') as file:
    data = json.load(file)

# Select the enforcement setup to run.
# New format: several named setups under "fairness_enforcement"; "active_setup" names the one to use.
# Old format (original repo): a single setup directly under "fairness_enforcement".
if 'active_setup' in data:
    # Name of the selected setup, e.g. "gender_occupation"
    active_setup = data['active_setup']
    # Stop with a clear message if the name does not exist (e.g. a typo in config.json)
    if active_setup not in data['fairness_enforcement']:
        print(f"Error: active_setup '{active_setup}' not found under fairness_enforcement", file=sys.stderr)
        sys.exit(1)
    # Settings of the selected setup: conditioned_value, fairness_group, beta and/or target
    setup_cfg = data['fairness_enforcement'][active_setup]
    # GPU flags describe the hardware, not a use case, so they sit at the top level
    hardware_cfg = data
else:
    # Old format has no setup name
    active_setup = None
    # The single setup of the old format
    setup_cfg = data.get('fairness_enforcement', {})
    # In the old format the GPU flags are inside the setup
    hardware_cfg = setup_cfg

print(f"Active setup: {active_setup}")

# Select the models to run. Chosen independently of the enforcement setup above:
#  - "model_catalog": every model defined once, grouped by role (image_generation, llm, image_interpretation)
#  - "pipelines": named model combinations that only refer to names in the catalog
#  - "active_pipeline": name of the pipeline to run, e.g. "cheng2025_baseline"
# Note: "pipeline" further below is the diffusers image synthesis object, not a pipeline of config.json.

# Models of the original paper and repo (Cheng et al. 2025).
# Used when config.json has no "active_pipeline" (e.g. the old format of the original repo).
CHENG2025_BASELINE_MODELS = {
    'image_generation': {'name': 'sd_1_5', 'model_id': 'runwayml/stable-diffusion-v1-5'},
    'gate_llm': {'name': 'gemma2_2b', 'model_id': 'gemma2:2b'},
    'story_llm': {'name': 'gemma2_2b', 'model_id': 'gemma2:2b'},
    'image_interpretation': {'name': 'moondream2_2024_07_23', 'model_id': 'vikhyatk/moondream2', 'revision': '2024-07-23'},
}

# Catalog section each pipeline role is looked up in; gate and story share the "llm" section
catalog_sections = {'image_generation': 'image_generation', 'gate_llm': 'llm',
                    'story_llm': 'llm', 'image_interpretation': 'image_interpretation'}

if 'active_pipeline' in data:
    # Name of the selected pipeline, e.g. "cheng2025_baseline"
    active_pipeline = data['active_pipeline']
    # Stop with a clear message if the name does not exist (e.g. a typo in config.json)
    if active_pipeline not in data.get('pipelines', {}):
        print(f"Error: active_pipeline '{active_pipeline}' not found under pipelines", file=sys.stderr)
        sys.exit(1)
    pipeline_cfg = data['pipelines'][active_pipeline]
    # Resolve each role to its catalog entry: role -> {name, model_id, revision (optional)}
    models = {}
    for role, section in catalog_sections.items():
        name = pipeline_cfg.get(role)
        # An explicit "story_llm": null turns the story step off: the user prompt goes directly to the image model
        if role == 'story_llm' and role in pipeline_cfg and name is None:
            models[role] = None
            continue
        catalog = data.get('model_catalog', {}).get(section, {})
        if name not in catalog:
            print(f"Error: {role} '{name}' of pipeline '{active_pipeline}' not found under model_catalog.{section}", file=sys.stderr)
            sys.exit(1)
        # The description is only documentation and not needed at runtime
        models[role] = {'name': name, **{k: v for k, v in catalog[name].items() if k != 'description'}}
else:
    active_pipeline = 'cheng2025_baseline'
    models = CHENG2025_BASELINE_MODELS

print(f"Active pipeline: {active_pipeline}")
for role, model in models.items():
    print(f"  {role}: {model if model is not None else 'None (story step off)'}")

# Image synthesis
pipeline = DiffusionPipeline.from_pretrained(models['image_generation']['model_id'], torch_dtype=torch.float16)
if hardware_cfg.get('diffusion_GPU'):
    print("Use GPU for the image synthesis task")
    pipeline.to("mps" if torch.backends.mps.is_available() else "cuda")
else:
    print("Use CPU for the image synthesis task")    

# Image interpretation
model_id = models['image_interpretation']['model_id']
# None = latest version of the model repository
revision = models['image_interpretation'].get('revision')
img_explain_model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision
)

if hardware_cfg.get('image_interpretation_GPU'):
    print("Use GPU for the image interpretation task")
    img_explain_model.to("cuda")
else:
    print("Use CPU for the image interpretation task")    


tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)


app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes


#llm = Ollama(model="llama3", request_timeout=120.0)
# Gate LLM: checks whether a prompt is related to conditioned_value and already asks for a group value
gate_llm = Ollama(model=models['gate_llm']['model_id'], request_timeout=120.0)
# Story LLM: writes the description that is passed to the image model (None = no story step)
story_llm = Ollama(model=models['story_llm']['model_id'], request_timeout=120.0) if models['story_llm'] is not None else None

header_description = " The person has a strong "
tail_description = " looking."

def check_prompt_topic(instruction, topic):
    # Placeholder function to call your local LLM

    resp = gate_llm.complete("Simply answer yes or no. Is the following instruction about generating textual descriptions related to "+topic+ "?\n" + instruction)
    print("Result of checking prompt topic: "+str(resp))
    return str(resp)




def generate_description_local_llm(instruction):

    resp = story_llm.complete(instruction)
    print(str(resp))
    return str(resp)


def generate_image_from_text(text, output_path):
    image = pipeline(text).images[0]
    image.save(output_path)







cgvs_map = {
    'gender': ["male", "female"],
    'demographics': ["white (Caucasian) ", "black (African American)", "Indian (native American)", "Asian (e.g., Chinese, Japanese)", "Latino"]
}

image_question_map = {
    'gender':  "For the image, describe the gender of the character. If it looks like male, return \"male\". If it looks like female, return \"female\". Otherwise, return \"other\". Only return one of the three words.",
    'demographics': "For the image, describe the demograpgics / ethnicity of the character. Use one of the following options, and find the closest one: {white, black, indian, latino, asian}. If you can't really identify, then use \"other\""

}

# Asked before the group question in the target-based path; the same for every fairness_group.
# "human" excludes robots, mascots and cartoon figures, which the group question would still
# assign to a group value (e.g. "female").
person_question = "Is there a human person in the image? Answer \"yes\" or \"no\" only."

description_annotation_map = {
    'gender': True,
    'demographics': False
}

def check_prompt_gender(instruction):
    resp = gate_llm.complete(" Answer \"yes\" or \"no\" only. Can you check if the instruction explicitly asks for generating a character of a specific gender (e.g., male)?\n "+instruction)
    print("The prompt asks for gender: "+str(resp))
    return str(resp)

def check_prompt_demographics(instruction):
    resp = gate_llm.complete(" Answer \"yes\", or \"no\" only. Can you check if the instruction explicitly asks for generating person of a specific demographics (e.g., white, black)?\n "+instruction)
    print("The prompt asks for demographics: "+str(resp))
    return str(resp)

query_function_map = {
    'gender': check_prompt_gender,
    'demographics': check_prompt_demographics
}



# Change this, if you wish to enforce fairness under some other criteria (e.g., cook).
conditioned_value = setup_cfg.get('conditioned_value')

# Change this, if you wish to enforce fairness to other dimensions
fairness_group = setup_cfg.get('fairness_group')

# If beta = len of fairness_cgvs, then it will lead to round-robin 
beta = setup_cfg.get('beta')

print("conditioned_value | fairness_group | beta:")
print(f"{conditioned_value} | {fairness_group} | {beta}")


fairness_cgvs = cgvs_map[fairness_group]







# Switch between the two code paths:
#  - setup WITHOUT a "target" block (e.g. successful_business_leader): original code of the paper
#  - setup WITH a "target" block (e.g. gender_occupation): target-based enforcement via targets.py
use_targets = 'target' in setup_cfg

if not use_targets:
    # ----- Original code path: one uniform beta for all group values -----

    if beta < len(fairness_cgvs):
        err_msg = "The length of the items should at least be as large as the beta value"
        print(f"Error: {err_msg}", file=sys.stderr)
        sys.exit(1)


    counters = {item: beta for item in fairness_cgvs}

else:
    # ----- Target-based code path -----

    # Turn the target distribution of the setup into one deadline per group value
    try:
        plan = build_plan(setup_cfg, fairness_cgvs)
    except ValueError as err:
        # Inconsistent config, e.g. a group value missing or shares not summing to 1
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)

    # Show what will be enforced, so a wrong config is noticed before any image is generated
    print(f"Target mode: {plan['mode']}")
    # Target share per group value (None in mode "cheng", which has no target distribution)
    print(f"Target distribution: {plan['target']}")
    # Deadline per group value: "at least once in every beta images"
    print(f"Deadlines (beta): {plan['beta']}")
    # Guaranteed minimum share per group value = 1 / beta
    print(f"Guaranteed minimum shares: { {v: round(1 / b, 3) for v, b in plan['beta'].items()} }")
    # Group values without a deadline; their share is left to the generator
    print(f"No deadline: {plan['unconstrained']}")

    # Scheduler that decides before each related image whether to enforce a group value.
    # Ties between equally urgent values are broken by the target shares (uniform in mode "cheng").
    scheduler = DeadlineScheduler(plan['beta'], weights=plan['target'])


# ----- Request log (both code paths) -----
# One JSON line per request, so realised distributions, intervention rates and failed
# enforcements can be computed afterwards. The file is appended to across server runs.
log_path = os.path.join('generated_images', 'enforcement_log.jsonl')
# Start time of this server run; separates the runs within the same log file
run_id = time.strftime('%Y-%m-%d_%H-%M-%S')
# Name of the code path, stored with every request: "original" or the target mode (e.g. "empirical")
path_mode = plan['mode'] if use_targets else 'original'


def log_request(record):
    """Append one request record as a single JSON line to the log file."""
    # Same fields for every request: run, setup, code path, pipeline and its resolved models.
    # The models are stored with name, model_id and revision, so a record stays correct
    # even if the catalog entry in config.json is changed later.
    record = {'run_id': run_id, 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
              'setup': active_setup, 'mode': path_mode,
              'pipeline': active_pipeline, 'models': models, **record}
    # ensure_ascii=False keeps characters like "ä" readable in the file
    with open(log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(json.dumps(record, ensure_ascii=False) + '\n')





@app.route('/process_instruction', methods=['POST'])
def process_instruction():

    if not use_targets:
        print(counters)
    else:
        # Counters of the target-based path: images left until each deadline runs out
        print(scheduler.counters)

    # Copy of the counters before this request, for the log
    counters_before = dict(counters) if not use_targets else dict(scheduler.counters)

    data = request.get_json()
    
    # Print the received POST data
    print("Received POST data:", data)
    
    instruction = "Generate a short text describing an imaginary example of " +data.get('instruction')+". "
    if not data.get('instruction'):
        return jsonify({'error': 'No instruction provided'}), 400

    topic_monitoring = False

    # Defaults for the log; these variables are only set further down if the corresponding step runs
    fairness_dimension_query = None  # gate answer "does the prompt already ask for a group value?"
    person_answer = None             # raw answer of the vision model to person_question (None = not asked)
    attribute = None                 # raw answer of the vision model (None = no vision check)
    concept_group_value = None       # group value used for the counter update

    # Check if the prompt is related to the topic, and if user has special requests on the dimension
    prompt_topic = check_prompt_topic(instruction, conditioned_value).strip().lower()
    print("The instruction sent by user shall be conditioned to fairness: "+ prompt_topic)
    if "yes" in prompt_topic:
        fairness_dimension_query = query_function_map[fairness_group](instruction).strip().lower()
        print("Fairness dimension on ("+fairness_group+") included? " + fairness_dimension_query)
        if "no" in fairness_dimension_query:
            # This means that the user has no special request on the concept group value (e.g., male for gender)
            topic_monitoring = True

    txt_enforcement_prompt = ""
    img_enforcement_prompt = ""
    enforced_cgv = ""

    if topic_monitoring == True:
        # Consider if enforcement should be triggered

        if not use_targets:
            # ----- Original code path -----

            # Check the condition from largest to smallest
            for k in range(beta, 0, -1):
                # Find items with a counter value of exactly k
                matching_items = [item for item, count in counters.items() if count == k]
                # If there are exactly k items with the counter value k
                if len(matching_items) == k:

                    # Randomly select an item from those that match the condition
                    replacement = random.choice(matching_items)
                    enforced_cgv = replacement
                    print("ENFORCEMENT: Enforcing '{}' based on the condition.".format(replacement))

                    if description_annotation_map[fairness_group] == True:
                        txt_enforcement_prompt = header_description+replacement+tail_description

                    img_enforcement_prompt = header_description+replacement+tail_description
                    break

        else:
            # ----- Target-based code path -----

            # Ask the scheduler whether a deadline is about to run out.
            # Returns the group value to enforce (e.g. "female"), or None if there is still time.
            replacement = scheduler.select()

            if replacement is not None:
                # Remember the enforced value, as in the original path
                enforced_cgv = replacement
                print("ENFORCEMENT: Enforcing '{}' based on the condition.".format(replacement))

                # Same clause wording as the original path, so both paths differ only in
                # WHEN and WHICH value is enforced, not in HOW it is phrased
                if description_annotation_map[fairness_group] == True:
                    # For gender, the clause is also added to the text prompt of the LLM
                    txt_enforcement_prompt = header_description+replacement+tail_description

                # The clause is always added to the image prompt
                img_enforcement_prompt = header_description+replacement+tail_description
    else:
        print("The topic shall not be monitored")

    if story_llm is not None:
        # Generate image description
        llm_response = generate_description_local_llm( txt_enforcement_prompt + instruction + txt_enforcement_prompt)
        image_prompt = img_enforcement_prompt + llm_response + img_enforcement_prompt
    else:
        # No story step: the user prompt goes directly to the image model, with the same clause before and after.
        # No template (e.g. "a photo of") is added, so both conditions differ only in the story step.
        llm_response = "(No story step: the prompt was sent directly to the image model.)"
        image_prompt = img_enforcement_prompt + data.get('instruction') + img_enforcement_prompt
    
    # Generate an image using Stable Diffusion
    image_filename = f"{uuid.uuid4()}.png"
    image_path = os.path.join('generated_images', image_filename)
    print(str(image_filename))
    generate_image_from_text(image_prompt, image_path)

    # Perform fainess counter update
    if topic_monitoring == True:

        if not use_targets:
            # ----- Original code path -----

            concept_group_value = ""

            if enforced_cgv != "":
                # Here we are explicitly assuming that the enforcement works so there is no need to double check.
                concept_group_value = enforced_cgv
            else:
                # Use the vision-based interpreter to check the class
                image_pil = Image.open(image_path)
                enc_image = img_explain_model.encode_image(image_pil)
                attribute = img_explain_model.answer_question(enc_image, image_question_map[fairness_group], tokenizer).strip().lower()
                print("The synthesized image has the attribute: "+attribute)
                is_contained = False
                for item in cgvs_map[fairness_group]:
                    # Check if the function's output is a substring of the list item
                    if attribute.lower() in item.lower():
                        concept_group_value = item
                        is_contained = True
                        break
                if is_contained == False and ("other" not in attribute):
                    err_msg = "The image explaination model produces an answer (apart from other): "+ attribute
                    print(f"Error: {err_msg}", file=sys.stderr)
                    sys.exit(1)

            for item in counters:
                if item == concept_group_value:
                    counters[item] = beta
                else:
                    counters[item] -= 1

        else:
            # ----- Target-based code path -----
            # Same detection as the original path, with four changes (marked CHANGE 1-4)

            # None = the image does not count ("removal", Definition 3 of the paper)
            concept_group_value = None

            # CHANGE 1: no shortcut for enforced images. EVERY image is checked with the vision model
            # (Algorithm 1, line 13), so a clause the generator ignored is noticed.
            # Use the vision-based interpreter to check the class
            image_pil = Image.open(image_path)
            # Encoded once, used for both questions
            enc_image = img_explain_model.encode_image(image_pil)

            # CHANGE 4: first ask whether a human person is in the image at all. Without a person
            # the image does not count, and the group question is skipped: otherwise the vision model
            # still assigns a group value, e.g. "female" for a robot or an empty workshop.
            # startswith instead of "yes" in ..., so that e.g. "no, ... eyes ..." is not read as "yes".
            person_answer = img_explain_model.answer_question(enc_image, person_question, tokenizer).strip().lower()
            print("Human person in the synthesized image: "+person_answer)

            if person_answer.startswith("yes"):
                attribute = img_explain_model.answer_question(enc_image, image_question_map[fairness_group], tokenizer).strip().lower()
                print("The synthesized image has the attribute: "+attribute)
                for item in cgvs_map[fairness_group]:
                    # CHANGE 2: skip empty answers. An empty string is a substring of every label
                    # and would otherwise be counted as the first group value (e.g. "male").
                    # Check if the function's output is a substring of the list item
                    if attribute != "" and attribute.lower() in item.lower():
                        concept_group_value = item
                        break
                # CHANGE 3: "other", empty or unexpected answers keep concept_group_value = None, so the image
                # does not count. The original path decrements every counter for "other" and stops the
                # server with sys.exit for unexpected answers.
                if concept_group_value is None:
                    print("The image is not counted (vision model answer: '{}')".format(attribute))
            else:
                print("The image is not counted (no human person; vision model answer: '{}')".format(person_answer))

            # Report when the generator did not follow the enforcement clause
            if enforced_cgv != "" and concept_group_value != enforced_cgv:
                print("ENFORCEMENT FAILED: requested '{}', vision model saw '{}'".format(enforced_cgv, concept_group_value))

            # Update the deadlines with the detected value; None changes nothing
            scheduler.update(concept_group_value)
            print(scheduler.counters)

    # Write one log line for this request (both code paths)
    log_request({
        # What the user typed into the form
        'user_prompt': data.get('instruction'),
        # Gate answer: is the prompt related to conditioned_value? (Algorithm 1, line 4)
        'gate_related': prompt_topic,
        # Gate answer: does the prompt already ask for a group value? (None if not related)
        'gate_group_specified': fairness_dimension_query,
        # True if the image counts towards the deadlines (related and no group value requested)
        'monitored': topic_monitoring,
        # Group value enforced by the appended clause (None = prompt unchanged)
        'enforced': enforced_cgv or None,
        # Raw answer of the vision model to the person question (None = not asked: request not
        # monitored, or original path)
        'vision_person_answer': person_answer,
        # Raw answer of the vision model (None = not checked; the original path skips enforced images,
        # the target-based path skips images without a human person)
        'vision_answer': attribute,
        # Group value used for the counter update (None = not counted).
        # In the original path this is the ENFORCED value for enforced images, not a detected one.
        'counted_as': concept_group_value or None,
        # Counters before and after this request
        'counters_before': counters_before,
        'counters_after': dict(counters) if not use_targets else dict(scheduler.counters),
        # Exact prompt sent to the image model (clause + LLM description + clause,
        # or clause + user prompt + clause without the story step)
        'image_prompt': image_prompt,
        # Generated image in generated_images/
        'image_file': image_filename,
    })

    if enforced_cgv == "":
         return jsonify({
        'response': llm_response,
        'image_url': f"http://localhost:5000/get_image/{image_filename}"
        })
    else:
         return jsonify({
        'response': llm_response + "\n **** Fairness enforcement enabled: "+enforced_cgv +" ****\n",
        'image_url': f"http://localhost:5000/get_image/{image_filename}"
        })


@app.route('/get_image/<filename>', methods=['GET'])
def get_image(filename):
    image_path = os.path.join('generated_images', filename)
    if os.path.exists(image_path):
        return send_file(image_path, mimetype='image/png')
    else:
        return jsonify({'error': 'Image not found'}), 404

if __name__ == '__main__':
    if not os.path.exists('generated_images'):
        os.makedirs('generated_images')
    app.run(debug=True, use_reloader=False,port=5000) #turn off Flask's auto-reloader