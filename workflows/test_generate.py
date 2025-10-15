import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from core.steerable import SteerableModel

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    steerable = SteerableModel(model, tokenizer, device)
    messages = [
        [
            {'role': 'user', 'content': 'Hello World!'}
        ],
        [
            {'role': 'user', 'content': 'Who are you?'}
        ],
    ]

    output = steerable.generate(messages, max_new_tokens=256)
    print(output)

if __name__ == '__main__':
    main()