import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from collections import defaultdict
from typing import Dict, List, Tuple

from .augmentations import tta_augment


def build_dataloaders(
    trn_split_list: List[str],
    tst_split_list: List[str],
    root: str = "data",
    pct: float = 0.5,
    batch_size: int = 8,
    shuffle: bool = True,
    should_tta: bool = False,
    seed: int = 43,
):
    from .data_utils import load_image_paths, split_low_light

    normal_paths, normal_labels, low_paths, low_labels, label_map = load_image_paths(trn_split_list, tst_split_list, root)

    (
        low_train_paths, low_train_labels,
        low_val_paths, low_val_labels,
        low_test_paths, low_test_labels,
    ) = split_low_light(low_paths, low_labels, pct, seed=seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((112, 112)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    class FaceDataset(Dataset):
        def __init__(self, paths, labels, transform=None, is_tst=False, should_augment=False):
            self.paths = paths
            self.labels = labels
            self.transform = transform
            self.is_tst = is_tst
            self.should_augment = should_augment

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            import cv2
            import numpy as np
            path = self.paths[idx]
            label = self.labels[idx]
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            if self.is_tst:
                tta_imgs = tta_augment(img) if self.should_augment else [img]
                if self.transform is not None:
                    tta_imgs = [self.transform(im) for im in tta_imgs]
                return tta_imgs, label
            if self.transform is not None:
                img = self.transform(img)
            return img, label

    train_ds = FaceDataset(
        normal_paths + low_train_paths,
        normal_labels + low_train_labels,
        transform=transform,
    )
    val_ds = FaceDataset(low_val_paths, low_val_labels, transform=transform)
    test_ds = FaceDataset(low_test_paths, low_test_labels, transform=transform, is_tst=True, should_augment=should_tta)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader, label_map


def train(
    model,
    train_loader,
    val_loader,
    id_name_map: Dict[int, str],
    lr: float = 1e-3,
    lr_drop_ratio: float = 0.01,
    num_epochs: int = 10,
    unfreeze_epoch: int = 3,
    num_layers_unfreeze: int = 2,
    patience: int = 3,
    device: str = "cuda",
    with_archead: bool = False,
):
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    model.to(device)

    for p in model.backbone.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        if epoch == unfreeze_epoch:
            if num_layers_unfreeze > 0:
                model.unfreeze_last_layers(n=num_layers_unfreeze)
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr * lr_drop_ratio)
            elif num_layers_unfreeze == 0:
                pass
            else:
                for p in model.backbone.parameters():
                    p.requires_grad = True
                optimizer = torch.optim.Adam(model.parameters(), lr=lr * lr_drop_ratio)

        model.train()
        running_loss = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs, labels) if with_archead else model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        from .evaluation import eval_minimal_model
        loss_dict = eval_minimal_model(model, val_loader, id_name_map, device=device, with_archead=with_archead)
        val_loss, overall_acc = loss_dict['val_loss'], loss_dict['overall_acc']
        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {running_loss/len(train_loader):.4f} Val Loss: {val_loss:.4f}")
        print("Overall accuracy:", overall_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            print(f"No improvement. Patience {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping!")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def pretrain(
    model,
    dataloader,
    num_epochs: int = 10,
    lr: float = 1e-4,
    patience: int = 3,
    device: str = "cuda",
    use_pretrain: bool = True,
    use_early_stopping: bool = True,
):
    if not use_pretrain:
        print("Skip pretrain (use_pretrain=False)")
        return model

    model.to(device)
    model.train()

    criterion = nn.CosineEmbeddingLoss(margin=0.11)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    best_model_state = None
    no_improve_epochs = 0

    for epoch in range(num_epochs):
        running_loss = 0.0
        for img1, img2, target in dataloader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            z1, z2 = model(img1, img2)
            loss = criterion(z1, z2, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(dataloader)
        print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}")

        if use_early_stopping:
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_state = copy.deepcopy(model.state_dict())
                no_improve_epochs = 0
                print(f"Improved! Saving best model (loss={best_loss:.4f})")
            else:
                no_improve_epochs += 1
                print(f"No improvement ({no_improve_epochs}/{patience})")
            if no_improve_epochs >= patience:
                print("Early stopping!")
                break
        else:
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_state = copy.deepcopy(model.state_dict())

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    print("Pretrain done!")
    return model
