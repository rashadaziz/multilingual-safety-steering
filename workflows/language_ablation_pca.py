import torch
import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer
from core.inspectable import InspectableModel
from data import load_flores_plus, load_polyrefuse, Language, FLORES_LANGUAGE_CODES
from utils.language_subspace import compute_language_means, compute_language_subspace
from hooks.language_ablation import build_language_ablation_hook

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inspectable = InspectableModel(model, tokenizer, device)
    language_means = compute_language_means(inspectable)
    language_subspaces = {
        # get subspace per layer
        i: compute_language_subspace(language_means[i], rank=4)[0].to(device, dtype=torch.float16) for i in range(len(inspectable.layers))
    }

    samples_per_lang = 256
    languages = [
        Language.ENGLISH,
        Language.KOREAN
    ]

    harmless_prompts = {
        lang: load_polyrefuse(lang, kind='harmless', num_samples=samples_per_lang) for lang in languages
    }

    harmful_prompts = {
        lang: load_polyrefuse(lang, kind='harmful', num_samples=samples_per_lang) for lang in languages
    }

    original_activations = {}
    ablated_activations = {}
    batch_size = 32

    # Register hooks
    for layer_idx, subspace in language_subspaces.items():
        hook = build_language_ablation_hook(subspace, ablation_strength=1)
        inspectable.register_hook(layer_idx, hook)

    dataset_configs = [
        ("harmless", harmless_prompts),
        ("harmful", harmful_prompts),
    ]
    markers = {"harmless": "o", "harmful": "X"}

    for dataset_name, prompts in dataset_configs:
        for lang in tqdm.tqdm(languages, desc=f"{dataset_name.title()} Languages"):
            dataset = prompts.get(lang, [])
            if not dataset:
                continue
            key = (dataset_name, lang)
            for i in tqdm.tqdm(range(0, len(dataset), batch_size), desc="Batches"):
                batch = dataset[i : i + batch_size]
                conversations = [
                    [{"role": "user", "content": example.prompt}] for example in batch
                ]

                # Original activations
                orig_acts = inspectable.cache_activations(conversations)
                original_activations.setdefault(key, []).append(orig_acts[:, :, -1, :].contiguous())

                # Ablated activations
                ablated_acts = inspectable.cache_interventions(conversations, independent=True)
                ablated_activations.setdefault(key, []).append(ablated_acts[:, :, -1, :].contiguous())

            if key in original_activations and original_activations[key]:
                original_activations[key] = torch.cat(original_activations[key], dim=1)
            if key in ablated_activations and ablated_activations[key]:
                ablated_activations[key] = torch.cat(ablated_activations[key], dim=1)

    # Perform PCA using sklearn per layer across all languages
    for i in range(len(inspectable.layers)):
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        original_batches = []
        ablated_batches = []
        scatter_spans: list[tuple[str, int, str]] = []

        for dataset_name, prompts in dataset_configs:
            for lang in languages:
                key = (dataset_name, lang)
                if key not in original_activations or key not in ablated_activations:
                    continue
                orig_acts = original_activations[key][i]
                ablated_acts = ablated_activations[key][i]
                if orig_acts.shape[0] != ablated_acts.shape[0]:
                    raise ValueError(f"Mismatched activation counts for language {lang.value} at layer {i}")

                original_batches.append(orig_acts)
                ablated_batches.append(ablated_acts)
                label = f"{lang.value} ({dataset_name})"
                scatter_spans.append((label, orig_acts.shape[0], markers.get(dataset_name, "o")))

        if not original_batches:
            plt.close(fig)
            continue

        stacked_original = torch.cat(original_batches, dim=0)
        stacked_ablated = torch.cat(ablated_batches, dim=0)

        pca = PCA(n_components=2)
        orig_proj = pca.fit_transform(stacked_original.detach().float().cpu().numpy())
        ablated_proj = pca.transform(stacked_ablated.detach().float().cpu().numpy())

        offset = 0
        for label, count, marker in scatter_spans:
            span = slice(offset, offset + count)
            ax[0].scatter(orig_proj[span, 0], orig_proj[span, 1], label=label, alpha=0.5, marker=marker)
            ax[1].scatter(ablated_proj[span, 0], ablated_proj[span, 1], label=label, alpha=0.5, marker=marker)
            offset += count

        ax[0].set_title(f'Layer {i} Original Activations PCA')
        ax[1].set_title(f'Layer {i} Ablated Activations PCA')
        for a in ax:
            a.legend()
        
        # Save the figure
        plt.savefig(f'layer_{i}_pca_comparison.png')
        plt.close()

if __name__ == '__main__':
    main()
