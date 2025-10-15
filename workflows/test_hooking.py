import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from core.inspectable import InspectableModel

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inspectable = InspectableModel(model, tokenizer, device)
    messages = [
        [
            {'role': 'user', 'content': 'Hello World!'}
        ],
        [
            {'role': 'user', 'content': 'Who are you?'}
        ],
    ]

    def hook():
        def _hook(module, inputs, outputs):
            print(outputs.shape)

        return _hook
    
    inspectable.register_hook(15, hook())
    inspectable.forward(messages)


if __name__ == '__main__':
    main()