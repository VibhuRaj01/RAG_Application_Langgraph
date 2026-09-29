from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class TransformerEmbedder:
    """Generate sentence embeddings using a Hugging Face transformer."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str | None = None,
    ):
        self.model_name = model_name

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def mean_pooling(
        model_output,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        token_embeddings = model_output.last_hidden_state

        input_mask_expanded = (
            attention_mask
            .unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )

        return torch.sum(
            token_embeddings * input_mask_expanded,
            dim=1,
        ) / torch.clamp(
            input_mask_expanded.sum(dim=1),
            min=1e-9,
        )

    def encode(
        self,
        texts: List[str],
        batch_size: int = 16,
    ) -> List[List[float]]:
        all_embeddings = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]

            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            encoded = {
                key: value.to(self.device)
                for key, value in encoded.items()
            }

            with torch.no_grad():
                model_output = self.model(**encoded)

            embeddings = self.mean_pooling(
                model_output,
                encoded["attention_mask"],
            )

            embeddings = F.normalize(
                embeddings,
                p=2,
                dim=1,
            )

            all_embeddings.extend(
                embeddings.cpu().tolist()
            )

        return all_embeddings