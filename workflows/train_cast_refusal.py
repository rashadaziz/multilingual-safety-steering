import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from activation_steering import SteeringDataset, SteeringVector
from pathlib import Path

import argparse

def main(args):
    # 1. Load model
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map='auto', dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # 2. Load data
    with open(args.data_path / "alpaca.json", 'r') as file:
        alpaca_data = json.load(file)

    with open(args.data_path / "behavior_refusal.json", 'r') as file:
        refusal_data = json.load(file)

    questions = alpaca_data['train']
    refusal = refusal_data['non_compliant_responses']
    compliace = refusal_data['compliant_responses']

    # 3. Create our dataset
    refusal_behavior_dataset = SteeringDataset(
        tokenizer=tokenizer,
        examples=[(item["question"], item["question"]) for item in questions[:100]],
        suffixes=list(zip(refusal[:100], compliace[:100]))
    )

    # 4. Extract behavior vector for this setup with 8B model, 10000 examples, a100 GPU -> should take around 4 minutes
    # To mimic setup from Representation Engineering: A Top-Down Approach to AI Transparency, do method = "pca_diff" amd accumulate_last_x_tokens=1
    refusal_behavior_vector = SteeringVector.train(
        model=model,
        tokenizer=tokenizer,
        steering_dataset=refusal_behavior_dataset,
        method="pca_pairwise",  # Using recommended method
        accumulate_last_x_tokens="suffix-only"
    )

    # 5. Let's save this behavior vector for later use
    refusal_behavior_vector.save(str(args.output_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='CAST-Style Refusal Vector Extraction'
    )
    parser.add_argument('--model', default='Qwen/Qwen2.5-1.5B-Instruct')
    parser.add_argument('-dp', '--data-path', type=Path, default=Path('data/json/cast'))
    parser.add_argument('-op', '--output-path', type=Path, default=Path('output/cast/refusal_vector'))

    args = parser.parse_args()
    main(args)