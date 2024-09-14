import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel

from config.sample_config import get_config


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    LAYERS = [
        "last",
        "pooled",
        "hidden"
    ]

    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77,
                 freeze=True, layer="pooled", layer_idx=2):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.transformer.to(device)
        self.layer = layer
        self.layer_idx = layer_idx
        if layer == "hidden":
            assert layer_idx is not None
            assert 0 <= abs(layer_idx) <= 12

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens, output_hidden_states=self.layer == "hidden")
        if self.layer == "last":
            z = outputs.last_hidden_state[:, -1]  # (1,77,768)  3.0986e-01,  1.9037e-01；  5.9811e-01,  6.6465e-01
        elif self.layer == "pooled":
            z = outputs.pooler_output  # (1,768)
        else:
            z = outputs.hidden_states[self.layer_idx]  # 包括13个，每个都是(1,77,168)
        return z

    def encode(self, text):
        return self(text)


if __name__ == '__main__':
    config = get_config()
    clip_text_model = FrozenCLIPEmbedder(version=config.clip_text_model, device="cuda")
    print(clip_text_model.encode("satelite images of Quick Bird") - clip_text_model.encode(
        "satelite images of GaoFen Two"))
