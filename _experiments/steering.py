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
# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_str = "Qwen/Qwen2.5-1.5B-Instruct"

# %%
tokenizer = AutoTokenizer.from_pretrained(model_str)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(model_str, dtype=torch.bfloat16).to(device)
generation_config = GenerationConfig.from_pretrained(model_str)

# %%
dataset = [
    "Hello, what's your name?",
    "Halo, apa kabar?",
    "How can we reduce air pollution?"
]
# %%
batch_size = 32
num_batches = (len(dataset) + batch_size - 1) // batch_size if dataset else 0
# %%
def steering_hook(vector):
    def _hook(_, input, output):
        # apply this only on initial forward pass and not during generation
        if input[0].shape[1] == 1:
            return None
        return output + 0.45 * vector
    
    return _hook

steering_layers = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20]
# %%
refusal_vectors: torch.Tensor = torch.load('output/refusal-vector-training/refusal_vectors.pt')
handles = []
for i, layer in enumerate(model.model.layers):
    normalized = refusal_vectors[i] / refusal_vectors[i].norm()
    print(normalized)
    handle = layer.register_forward_hook(steering_hook(normalized.type(torch.bfloat16).to(device)))
    handles.append(handle)
# %%
prompt_response_pairs = []

for i in tqdm(
    range(0, len(dataset), batch_size),
    total=num_batches,
    desc="Generating responses",
):
    batch = dataset[i:i+batch_size]
    conversations = [
        [{'role': 'user', 'content': prompt}] for prompt in batch
    ]

    inputs = tokenizer.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        tokenize=True,
        padding=True, 
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        generation_config=generation_config,
        max_new_tokens=128,
        do_sample=False,
    )

    generated_sequences = outputs[:, inputs["input_ids"].shape[1]:].cpu()
    responses = tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)

    for prompt, response_text in zip(batch, responses):
        prompt_response_pairs.append({"prompt": prompt, "response": response_text.strip()})

    del inputs, outputs
    gc.collect()
    torch.cuda.empty_cache()
# %%
for handle in handles:
    handle.remove()
# %%
print(json.dumps(prompt_response_pairs, indent=4))
# %%
