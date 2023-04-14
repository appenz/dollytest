#
# Quick test for the Dolly LLM 
# (c) Guido Appenzeller, 2023
#

import os
from datetime import datetime

# Check if we are running in the cloud or on a laptop
if os.environ.get('USER') == 'ubuntu':
    platform = 'aws'
    model_v = "databricks/dolly-v2-12b"
elif False:
    platform = 'macbook'
    model_v = "databricks/dolly-v2-3b"
else:
    platform = 'test'
    model_v = "test/test-v1-0b"   

# Log messages to console with timestamp

def plog(msg):
    print(datetime.now().strftime("%H:%M:%S") + " " + msg)

plog('starting')

# Setup for different platforms

if platform == 'aws':
    from instruct_pipeline import InstructionTextGenerationPipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer

    plog("loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_v)
    plog("loading model")
    model = AutoModelForCausalLM.from_pretrained(model_v, device_map="auto", load_in_8bit=True)

    plog("creating pipeline")
    generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)

elif platform == 'macbook':
    from instruct_pipeline import InstructionTextGenerationPipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer

    plog("loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_v)
    plog("loading model")
    model = AutoModelForCausalLM.from_pretrained(model_v, device_map="auto", load_in_8bit=True)

    plog("creating pipeline")
    generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)

elif platform == 'test':
    generate_text = lambda x: "Dolly says:" + x

plog("ready.")

# Get user input and run through generate_text
while True:
    text = input(f'Enter prompt {model_v}: ')
    if len(text) > 0:
        print(generate_text(text))

