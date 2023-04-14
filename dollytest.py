#
# Quick test for the Dolly LLM 
# (c) Guido Appenzeller, 2023
#

import os
import platform
from datetime import datetime

my_os = platform.system()

try:
    from instruct_pipeline import InstructionTextGenerationPipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer
except:
    print("Warning: transformers not installed, using test model")
    mode = 'test'
    model_v = "test/test-v1-0b"
else:
    mode = 'prod'
    if my_os == 'Linux':
        model_v = "databricks/dolly-v2-12b"
    elif my_os == 'Darwin':
        model_v = "databricks/dolly-v2-3b"



# Log messages to console with timestamp

def plog(msg):
    print(datetime.now().strftime("%H:%M:%S") + " " + msg)

plog('starting')
if mode == 'prod':
    plog("loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_v)
    plog("loading model")
    if my_os == 'linux':
        model = AutoModelForCausalLM.from_pretrained(model_v, device_map="auto", load_in_8bit=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_v, device_map="auto")

    plog("creating pipeline")
    generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)
else:
    generate_text = lambda x: "Dolly says:" + x

plog("ready.")

# Get user input and run through generate_text
while True:
    text = input(f'Enter prompt {model_v}: ')
    if len(text) > 0:
        print(generate_text(text))

