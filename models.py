import timm
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


from config import cfg


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = timm.create_model(
            "vit_base_patch16_224.augreg2_in21k_ft_in1k", pretrained=True
        )
        self.data_config = timm.data.resolve_model_data_config(self.model)

    def forward(self, x):
        return self.model(x)


class TextEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.model = AutoModel.from_pretrained(
            "sentence-transformers/bert-base-nli-mean-tokens"
        )

        # set params to trainable mode
        for p in self.model.parameters():
            p.requires_grad = True

        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/bert-base-nli-mean-tokens"
        )

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def forward(self, x):
        """
        x : tokenized text
        """

        model_output = self.model(**x)

        # Perform pooling
        sentence_embeddings = self.mean_pooling(model_output, x["attention_mask"])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings

    def get_tokenzier(self):
        return self.tokenizer


class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=cfg.projection.projection_dim,
        dropout=cfg.projection.dropout,
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class ClipLoss(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        labels = torch.arange(num_logits, device=device, dtype=torch.long)
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(
            image_features, text_features, logit_scale
        )
        labels = self.get_ground_truth(device, logits_per_image.shape[0])
        total_loss = (
            F.cross_entropy(logits_per_image, labels)
            + F.cross_entropy(logits_per_text, labels)
        ) / 2
        return total_loss


class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=cfg.clip.temperature,
        image_embedding=cfg.projection.image_embedding,
        text_embedding=cfg.projection.text_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature

        self.clip_loss = ClipLoss()

        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07)))

    def forward(self, batch):
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["image"])

        # Filter the dictionary:
        keys_to_keep = ["input_ids", "token_type_ids", "attention_mask"]
        filtered_dict = {k: batch[k] for k in keys_to_keep if k in batch}
        text_features = self.text_encoder(filtered_dict)

        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        loss = self.clip_loss(image_embeddings, text_embeddings, self.logit_scale)

        return loss
