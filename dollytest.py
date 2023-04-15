#
# Quick test for the Dolly LLM 
# (c) Guido Appenzeller, 2023
#

import os
import platform
import torch
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
        ai_device = torch.cuda.current_device()
    elif my_os == 'Darwin':
        model_v = "databricks/dolly-v2-3b"
        ai_device = torch.device("mps")

# Log messages to console with timestamp

def plog(msg):
    print(datetime.now().strftime("%H:%M:%S") + " " + msg)

plog(f'starting with model {model_v} on {my_os}')

if mode == 'prod':
    plog("loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_v)

    plog("loading model")
    if my_os == 'Linux':
        model = AutoModelForCausalLM.from_pretrained(model_v, device_map="auto", load_in_8bit=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_v, device_map="auto", torch_dtype=torch.float16)
        
    # This causes an error in Mac "RuntimeError: Placeholder storage has not been allocated."
    # model.to(ai_device)

    device_map = [{str(value) for key, value in model.hf_device_map.items()}]
    device_str = "".join(str(x) for x in device_map)
    plog(f'creating pipeline on devices: {device_str}')
    generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)
else:
    generate_text = lambda x: "Dolly says:" + x

plog("ready.")

# Get user input and run through generate_text
while True:
    text = input(f'Enter prompt {model_v}: ')
    plog("---")
    if len(text) > 0:
        print(generate_text(text))
    plog("---")
