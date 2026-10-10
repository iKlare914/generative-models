"""CLIP text encoding and CIFAR-10 prompt generation."""

import numpy as np
import torch as th


CIFAR10_LABELS = {
    0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
    5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck",
}
CIFAR10_PROMPT_TEMPLATES = (
    "a photo of a {}",
    "a {}",
    "a image of a {}",
    "give me a photo of a {}",
    "give me a image of a {}",
    "{}",
)


class CLIPTextEncoder:
    """Frozen CLIP text encoder with CPU features cached by (prompt, token length).

    The model is loaded on the first cache miss. Pass the same encoder to a
    dataset and sampler to reuse its model and cache. CLIP always runs on CPU;
    only the encoded output tensors may be transferred to another device.
    """

    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.prompt_cache: dict[tuple[str, int], tuple[th.Tensor, th.Tensor]] = {}

    def _load(self):
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer

        class TextOnlyCLIP(CLIPTextModelWithProjection):
            _keys_to_ignore_on_load_unexpected = [
                r"^vision_model\.", r"^visual_projection\.", r"^logit_scale$",
            ]

        self.tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
        model, info = TextOnlyCLIP.from_pretrained(self.model_name, output_loading_info=True)
        if info["missing_keys"]:
            raise ValueError(f"Missing pretrained CLIP text weights: {info['missing_keys']}")
        self.model = model.cpu().eval().requires_grad_(False)

    @th.no_grad()
    def encode_prompts(
        self, prompts: str | list[str], *, max_length: int = 77, device="cpu",
    ) -> tuple[th.Tensor, th.Tensor]:
        """Return context [B, L, D] and bool mask [B, L], where True means valid.

        Missing prompts are encoded together, once per token length. Empty
        strings retain BOS/EOS tokens and therefore have a nonempty mask.
        Cached tensors are detached CPU tensors; returned batches are separate.
        device controls output placement only; the model and its inputs stay on CPU.
        """
        if not 2 <= max_length <= 77:
            raise ValueError("CLIP max_length must be between 2 and 77, including BOS/EOS.")
        prompts = [prompts] if isinstance(prompts, str) else list(prompts)
        missing = list(dict.fromkeys(prompt for prompt in prompts if (prompt, max_length) not in self.prompt_cache))
        if missing:
            if self.model is None:
                self._load()
            inputs = self.tokenizer(
                missing, padding="max_length", truncation=True, max_length=max_length,
                add_special_tokens=True, return_attention_mask=True, return_tensors="pt",
            ).to("cpu")
            hidden = self.model(**inputs).last_hidden_state # [B. L. D]
            mask = inputs["attention_mask"].bool() # [B, L]
            if not mask.any(dim=-1).all():
                raise ValueError("Text conditioning must contain at least one valid token.")
            for index, prompt in enumerate(missing):
                self.prompt_cache[(prompt, max_length)] = (
                    hidden[index].detach().cpu().clone(), mask[index].cpu().clone(),
                )
        context, masks = zip(*(self.prompt_cache[(prompt, max_length)] for prompt in prompts))
        return th.stack(context).to(device), th.stack(masks).to(device)


def get_cifar10_prompt(
    nums: int, text_encoder: CLIPTextEncoder | None = None, *, max_length: int = 77, device="cpu",
) -> tuple[list[str], th.Tensor, th.Tensor]:
    """Return (prompts, context, mask) for independently sampled labels and templates.

    Labels are drawn from 0 through 9 with replacement. Context has shape
    [nums, max_length, 512] for the default CLIP; mask has shape [nums, max_length].
    Pass an existing encoder to reuse its weights and cached features.
    """
    if nums <= 0:
        raise ValueError("nums must be positive.")
    labels = np.random.randint(0, 10, size=nums)
    templates = np.random.choice(CIFAR10_PROMPT_TEMPLATES, size=nums)
    prompts = [template.format(CIFAR10_LABELS[label]) for label, template in zip(labels, templates)]
    encoder = text_encoder if text_encoder is not None else CLIPTextEncoder()
    context, mask = encoder.encode_prompts(prompts, max_length=max_length, device=device)
    return prompts, context, mask
