# %%
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)
import torch
import gc
from tqdm.auto import tqdm
import os, glob, json
from activation_steering import SteeringDataset, SteeringVector

# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_str = "Qwen/Qwen2.5-1.5B-Instruct"

# %%
tokenizer = AutoTokenizer.from_pretrained(model_str)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(model_str, dtype=torch.float16).to(device)
generation_config = GenerationConfig.from_pretrained(model_str)

# %%
with open("data/cast/alpaca.json", 'r') as file:
    alpaca_data = json.load(file)

with open("data/cast/behavior_refusal.json", 'r') as file:
    refusal_data = json.load(file)

# %%
questions = alpaca_data['train']
refusal = refusal_data['non_compliant_responses']
compliace = refusal_data['compliant_responses']

# %%
refusal_behavior_dataset = SteeringDataset(
    tokenizer=tokenizer,
    examples=[(item["question"], item["question"]) for item in questions[:100]],
    suffixes=list(zip(refusal, compliace))
)
# %%
refusal_behavior_vector = SteeringVector.train(
    model=model,
    tokenizer=tokenizer,
    steering_dataset=refusal_behavior_dataset,
    method="pca_pairwise",  # Using recommended method
    accumulate_last_x_tokens="suffix-only",
    batch_size=128
)

# %%
refusal_behavior_vector.save('output/cast/refusal_vectors')
# %%
