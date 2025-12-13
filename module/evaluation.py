import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from typing import Dict


def eval_minimal_model(model, data_loader, id_name_map: Dict[int, str], device: str = "cuda", with_archead: bool = False):
    criterion = nn.CrossEntropyLoss()
    model.eval()

    correct = 0
    total = 0
    running_loss = 0.0
    class_correct = {}
    class_total = {}

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.inference_mode():
        for imgs, labels in data_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs, labels) if with_archead else model(imgs)
            probs = torch.softmax(logits, dim=1)
            all_probs.extend(probs.cpu().numpy())
            pred = logits.argmax(dim=1)
            loss = criterion(logits, labels)
            running_loss += loss.item()
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            for l, p in zip(labels, pred):
                class_total[int(l.item())] = class_total.get(int(l.item()), 0) + 1
                if l == p:
                    class_correct[int(l.item())] = class_correct.get(int(l.item()), 0) + 1
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())

    class_acc = {}
    for label in class_total:
        acc = class_correct[label] / class_total[label]
        class_acc[id_name_map[label]] = acc

    overall_acc = correct / total
    val_loss = running_loss / len(data_loader)

    return {
        "val_loss": val_loss,
        "overall_acc": overall_acc,
        "class_acc": class_acc,
    }


def eval_model(model, data_loader, id_name_map: Dict[int, str], device: str = "cuda", with_archead: bool = False):
    criterion = nn.CrossEntropyLoss()
    model.eval()

    correct = 0
    total = 0
    running_loss = 0.0
    class_correct = {}
    class_total = {}
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.inference_mode():
        for imgs, labels in data_loader:
            labels = labels.to(device)
            imgs = imgs[0]
            imgs = [im.to(device) for im in imgs]
            logits_list = []
            for im in imgs:
                logits = model(im.unsqueeze(0), labels) if with_archead else model(im.unsqueeze(0))
                logits_list.append(logits)
            avg_logits = torch.stack(logits_list).mean(dim=0)
            probs = torch.softmax(avg_logits, dim=1)
            pred = avg_logits.argmax(dim=1)
            loss = criterion(avg_logits, labels)
            running_loss += loss.item()
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            for l, p in zip(labels, pred):
                class_total[int(l.item())] = class_total.get(int(l.item()), 0) + 1
                if l == p:
                    class_correct[int(l.item())] = class_correct.get(int(l.item()), 0) + 1
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    class_acc = {}
    for label in class_total:
        acc = class_correct[label] / class_total[label]
        class_acc[id_name_map[label]] = acc

    overall_acc = correct / total
    val_loss = running_loss / len(data_loader)

    num_classes = len(id_name_map)
    eps = 1e-9
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    precision = {}
    recall = {}
    f1 = {}

    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1_i = 2 * prec * rec / (prec + rec + eps)
        name = id_name_map[i]
        precision[name] = prec
        recall[name] = rec
        f1[name] = f1_i

    balanced_acc = np.mean(list(class_acc.values()))
    precision_micro = precision_score(all_labels, all_preds, average='micro')
    recall_micro = recall_score(all_labels, all_preds, average='micro')
    f1_micro = f1_score(all_labels, all_preds, average='micro')
    precision_macro = precision_score(all_labels, all_preds, average='macro')
    recall_macro = recall_score(all_labels, all_preds, average='macro')
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    precision_weighted = precision_score(all_labels, all_preds, average='weighted')
    recall_weighted = recall_score(all_labels, all_preds, average='weighted')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')

    return {
        "val_loss": val_loss,
        "overall_acc": overall_acc,
        "balanced_acc": balanced_acc,
        "class_acc": class_acc,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }
