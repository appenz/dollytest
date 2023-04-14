#
# Quick test for the Dolly LLM 
# (c) Guido Appenzeller, 2023
#

from datetime import datetime

test = True
model = "databricks/dolly-v2-12b"

# Log messages to console with timestamp

def plog(msg):
    print(datetime.now().strftime("%H:%M:%S") + " " + msg)

plog('starting')

if not test:
    from instruct_pipeline import InstructionTextGenerationPipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer

    plog("loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model)
    plog("loading model")
    model = AutoModelForCausalLM.from_pretrained(model, device_map="auto", load_in_8bit=True)

    plog("creating pipeline")
    generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)
else:
    # a lambda function that returns the input prepended by "Dolly says: "
    generate_text = lambda x: "Dolly says: " + x

plog("ready.")

# Get user input and run through generate_text
while True:
    text = input(f'Enter prompt {model}: ')
    if len(text) > 0:
        print(generate_text(text))

