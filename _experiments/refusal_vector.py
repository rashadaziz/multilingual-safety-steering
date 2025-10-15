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
dataset = []

for fn in glob.glob('data/refusal-vector-training/*.json'):
    with open(os.path.join(os.getcwd(), fn), 'r') as f:
        data = json.load(f)
        dataset.extend(item['query'] for item in data)

# %%
def activations_hook(cache, layer):
    def _hook(_, input, output: torch.Tensor):
        last_token_acts = output[:, -1, :].detach().cpu().float()
        cache[layer].extend(last_token_acts)
        return None
    
    return _hook


def parse_guard_output(text: str):
    """Extract safety fields from Qwen Guard completion."""
    fields = {
        "safety": None,
        "categories": None,
        "refusal": None,
        "raw": text.strip(),
    }

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        lowered = line.lower()
        if lowered.startswith("safety:"):
            fields["safety"] = line.split(":", 1)[1].strip()
        elif lowered.startswith("categories:"):
            value = line.split(":", 1)[1].strip()
            if value and value.lower() != "none":
                fields["categories"] = [cat.strip() for cat in value.split(",") if cat.strip()]
        elif lowered.startswith("refusal:"):
            fields["refusal"] = line.split(":", 1)[1].strip()

    return fields
# %%
batch_size = 32
num_batches = (len(dataset) + batch_size - 1) // batch_size if dataset else 0
prompt_response_pairs = []

# activations = [[] for _ in range(len(model.model.layers))]
# handles = []

# for i, layer in enumerate(model.model.layers):
#     handle = layer.register_forward_hook(activations_hook(activations, i))
#     handles.append(handle)

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

# for handle in handles:
#     handle.remove()

# %%
# activations = torch.stack([torch.stack(act) for act in activations])
# %%
del tokenizer, model
gc.collect()
torch.cuda.empty_cache()
# %%
# torch.save(activations, 'output/refusal-vector-training/activations.pt')
# %%
guard_model_str = "Qwen/Qwen3Guard-Gen-4B"
guard_tokenizer = AutoTokenizer.from_pretrained(guard_model_str)
if guard_tokenizer.pad_token is None:
    guard_tokenizer.pad_token = guard_tokenizer.eos_token
guard_tokenizer.padding_side = "left"

guard_model_dtype = torch.float16 if device == 'cuda' else torch.float32
guard_model = AutoModelForCausalLM.from_pretrained(
    guard_model_str,
    torch_dtype=guard_model_dtype,
).to(device)
guard_model.eval()

classification_batch_size = 8 if device == 'cuda' else 2
guard_num_batches = (len(prompt_response_pairs) + classification_batch_size - 1) // classification_batch_size if prompt_response_pairs else 0

for i in tqdm(
    range(0, len(prompt_response_pairs), classification_batch_size),
    total=guard_num_batches,
    desc="Guard classification",
):
    batch_items = prompt_response_pairs[i:i + classification_batch_size]
    conversations = []
    for item in batch_items:
        prompt_text = item.get("prompt") or ""
        response_text = item.get("response") or ""
        conversations.append([
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": response_text},
        ])

    chat_texts = [
        guard_tokenizer.apply_chat_template(conv, tokenize=False)
        for conv in conversations
    ]

    inputs = guard_tokenizer(
        chat_texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids = guard_model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
        )

    prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()

    for offset, (length, input_ids) in enumerate(zip(prompt_lengths, generated_ids)):
        output_ids = input_ids[int(length):].detach().cpu().tolist()
        decoded = guard_tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        parsed = parse_guard_output(decoded)
        safety = parsed["safety"]
        harmfulness = None
        if safety:
            harmfulness = "harmless" if safety.lower().startswith("safe") else "harmful"

        batch_items[offset]["guard_response"] = decoded
        batch_items[offset]["guard_safety"] = safety
        batch_items[offset]["guard_harmfulness"] = harmfulness
        batch_items[offset]["guard_categories"] = parsed["categories"]
        batch_items[offset]["guard_refusal"] = parsed["refusal"]

    del inputs, generated_ids, prompt_lengths
    gc.collect()
    torch.cuda.empty_cache()

del guard_model, guard_tokenizer
gc.collect()
torch.cuda.empty_cache()
# %%
with open('output/refusal-vector-training/prompt_response_pairs.json', 'w', encoding='utf-8') as f:
    json.dump(prompt_response_pairs, f, ensure_ascii=False, indent=4)

# %%
with open('output/refusal-vector-training/activations.pt', 'rb') as f:
    activations = torch.load(f)
# %%
import polars as pl
prompt_response_pairs = pl.DataFrame(prompt_response_pairs)
# %%
reject_idx = (prompt_response_pairs['guard_refusal'] == 'Yes').to_torch()
comply_idx = (prompt_response_pairs['guard_refusal'] == 'No').to_torch()
# %%
reject_acts = activations[:, reject_idx, :]
comply_acts = activations[:, comply_idx, :]
# %%
reject_mean = reject_acts.mean(dim=1)
comply_mean = comply_acts.mean(dim=1)
# %%
refusal_vector = reject_mean - comply_mean
# %%
for l, r in enumerate(refusal_vector):
    print(f"Layer: {l}, Refusal Norm: {r.norm()}")
# %%
torch.save(refusal_vector, 'output/refusal-vector-training/refusal_vectors.pt')
# %%
