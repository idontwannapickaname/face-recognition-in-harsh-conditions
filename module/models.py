import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from backbones import get_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaselineModel(nn.Module):
    def __init__(self, model_name: str = "edgeface_xxs", embed_dim: int = 512, num_classes: int = 7):
        super().__init__()
        backbone = get_model(model_name)
        checkpoint_path = f"checkpoints/{model_name}.pt"
        backbone.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def unfreeze_last_layers(self, n: int = 2):
        for param in self.backbone.parameters():
            param.requires_grad = False
        layers = list(self.backbone.children())
        for layer in layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, x):
        embeddings = self.backbone(x)
        return self.head(embeddings)


class ContrastiveBackbone(nn.Module):
    def __init__(self, model_name: str = "edgeface_xxs", embed_dim: int = 512):
        super().__init__()
        backbone = get_model(model_name)
        checkpoint_path = f"checkpoints/{model_name}.pt"
        backbone.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.backbone = backbone

    def forward(self, x1, x2):
        z1 = self.backbone(x1)
        z2 = self.backbone(x2)
        return z1, z2


class ArcFaceHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, s: float = 30.0, m: float = 0.50):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.randn(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        W = F.normalize(self.weight, dim=1)
        x = F.normalize(embeddings, dim=1)
        cosine = F.linear(x, W).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cosine)
        target_theta = theta[torch.arange(len(labels)), labels]
        target_logits = torch.cos(target_theta + self.m)
        logits = cosine.clone()
        logits[torch.arange(len(labels)), labels] = target_logits
        return logits * self.s


class ContrastiveModel(nn.Module):
    def __init__(self, pretrained_backbone: ContrastiveBackbone, embed_dim: int = 512, num_classes: int = 7, num_unfreeze_layers: int = 2):
        super().__init__()
        self.backbone = pretrained_backbone.backbone
        self.unfreeze_last_layers(num_unfreeze_layers)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def unfreeze_last_layers(self, n: int = 2):
        for param in self.backbone.parameters():
            param.requires_grad = False
        layers = list(self.backbone.children())
        for layer in layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, x):
        embedding = self.backbone(x)
        logits = self.head(embedding)
        return logits


class ContrastiveModelWithArcHead(nn.Module):
    def __init__(self, pretrained_backbone: ContrastiveBackbone, embed_dim: int = 512, num_classes: int = 7, num_unfreeze_layers: int = 2):
        super().__init__()
        self.backbone = pretrained_backbone.backbone
        self.unfreeze_last_layers(num_unfreeze_layers)
        self.embedding_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.arcface = ArcFaceHead(embed_dim, num_classes)

    def unfreeze_last_layers(self, n: int = 2):
        for param in self.backbone.parameters():
            param.requires_grad = False
        layers = list(self.backbone.children())
        for layer in layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, x, labels):
        feat = self.backbone(x)
        embedding = self.embedding_head(feat)
        logits = self.arcface(embedding, labels)
        return logits


def save_model(model: nn.Module, path: str) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state_dict_path = path.replace('.pt', '_state_dict.pt')
        torch.save(model.state_dict(), state_dict_path)
        print(f"Saved state_dict to: {state_dict_path}")
    except Exception as e:
        print(f"Error saving model: {e}")


def load_model(model_instance: nn.Module, path: str) -> nn.Module:
    checkpoint = torch.load(path, map_location=device)
    model_instance.load_state_dict(checkpoint)
    model_instance.eval()
    print("Loaded state_dict.")
    return model_instance
