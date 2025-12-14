import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from typing import Dict
from collections import defaultdict

def eval_minimal_model(model, data_loader, id_name_map, device="cuda", with_archead=False):
    criterion = nn.CrossEntropyLoss()
    model.eval()

    # overall
    correct = 0
    total = 0
    running_loss = 0.0

    # per-class counters
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    # For precision/recall/F1
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.inference_mode():
        for imgs, labels in data_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            if with_archead:
                logits = model(imgs, labels)
            else:
                logits = model(imgs)
            probs = torch.softmax(logits, dim=1)
            all_probs.extend(probs.cpu().numpy())
            pred = logits.argmax(dim=1)

            # Loss
            loss = criterion(logits, labels)
            running_loss += loss.item()

            # overall
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            # per-class accuracy
            for l, p in zip(labels, pred):
                class_total[l.item()] += 1
                if l == p:
                    class_correct[l.item()] += 1

            # For other metrics
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())

    # ----- Accuracy per class -----
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


def eval_model(model, data_loader, id_name_map, device="cuda", with_archead=False):
    criterion = nn.CrossEntropyLoss()
    model.eval()

    from collections import defaultdict

    # overall
    correct = 0
    total = 0
    running_loss = 0.0

    # per-class counters
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    # For precision/recall/F1
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.inference_mode():
        for imgs, labels in data_loader:
            # imgs: list TTA → imgs shape = [batch, T, C, H, W]
            labels = labels.to(device)

            imgs = imgs[0]
            imgs = [im.to(device) for im in imgs]

            # ---- Tính logits trung bình qua TTA ----
            logits_list = []

            for im in imgs:
                if with_archead:
                    logits = model(im.unsqueeze(0), labels)
                else:
                    logits = model(im.unsqueeze(0))
                logits_list.append(logits)

            # average logits over TTA
            avg_logits = torch.stack(logits_list).mean(dim=0)
            probs = torch.softmax(avg_logits, dim=1)

            # lấy pred
            pred = avg_logits.argmax(dim=1)

            # Loss
            loss = criterion(avg_logits, labels)
            running_loss += loss.item()

            # overall
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            # per-class accuracy
            for l, p in zip(labels, pred):
                class_total[l.item()] += 1
                if l == p:
                    class_correct[l.item()] += 1

            # For other metrics
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # ----- Accuracy per class -----
    class_acc = {}
    for label in class_total:
        acc = class_correct[label] / class_total[label]
        class_acc[id_name_map[label]] = acc

    overall_acc = correct / total
    val_loss = running_loss / len(data_loader)

    # ----- Precision / Recall / F1 per class -----
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
        f1[name] = f1_i

    balanced_acc = np.mean(list(class_acc.values()))
    f1_micro        = f1_score(all_labels, all_preds, average='micro')
    f1_macro        = f1_score(all_labels, all_preds, average='macro')


    return {
        "val_loss": val_loss,
        "overall_acc": overall_acc,
        "balanced_acc": balanced_acc,
        "class_acc": class_acc,
        
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }
