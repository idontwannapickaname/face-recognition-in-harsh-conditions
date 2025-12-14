# run.py

import torch
import numpy as np
import os
from collections import defaultdict
from torch.utils.data import DataLoader
from typing import Dict, Any
from module.training import build_dataloaders, pretrain, train
from module.data_utils import load_image_paths
from module.datasets import PairedLightDarkDataset
from module.models import load_model, save_model, ContrastiveBackbone, ContrastiveModel, BaselineModel
from module.evaluation import eval_model

SEED = 200
# Chọn thiết bị: CUDA nếu có, ngược lại là CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Hyperparameters
# Dữ liệu
DATA_ROOT = "data"
TRN_SPLITS = ['normal']
TST_SPLITS = ['low_light']
LOW_LIGHT_SPLIT_PCT = 0.5               # 50% val, 50% tst
BATCH_SIZE = 8

# Pre-training Contrastive
PRETRAIN_EPOCHS = 50 
PRETRAIN_PATIENCE = 15
PRETRAIN_LR = 1e-3
PRETRAIN_BATCH_SIZE = 8192
PRETRAIN_BB_MODEL_PATH = 'models/best_bb_state_dict.pt'

# Fine-tuning
FINETUNE_EPOCHS = 60
FINETUNE_PATIENCE = 5
FINETUNE_LR = 1e-3
LR_DROP_RATIO = 0.1
UNFREEZE_EPOCH = 5
UNFREEZE_LAYERS = 0
FINETUNE_MODEL_PATH = 'models/best_model_state_dict.pt'
EMBED_DIM = 512

def get_model(use_pretrained_bb: bool = True, use_finetuned_model: bool = True):
    # ========== CASE 1: Load full fine-tuned model ==========
    if use_finetuned_model and os.path.exists(FINETUNE_MODEL_PATH):
        print(f"✅ Load fine-tuned model từ {FINETUNE_MODEL_PATH}")
        model = ContrastiveModel(
            pretrained_backbone=ContrastiveBackbone(embed_dim=EMBED_DIM),
            embed_dim=EMBED_DIM,
            num_unfreeze_layers=UNFREEZE_LAYERS,
        )
        return load_model(model, FINETUNE_MODEL_PATH).to(device)

    # ========== CASE 2: Chuẩn bị DataLoaders ==========
    print("\n--- Chuẩn bị DataLoaders ---")
    trn_ld, val_ld, tst_ld, name_id_map = build_dataloaders(
        trn_split_list=TRN_SPLITS,
        tst_split_list=TST_SPLITS,
        root=DATA_ROOT,
        pct=LOW_LIGHT_SPLIT_PCT,
        batch_size=BATCH_SIZE,
        should_tta=False,
        seed=SEED
    )

    num_classes = len(name_id_map)
    id_name_map = {v: k for k, v in name_id_map.items()}
    print(f"✔ Num classes: {num_classes}")

    # ========== CASE 2: Load backbone ==========
    if use_pretrained_bb and os.path.exists(PRETRAIN_BB_MODEL_PATH):
        print(f"✅ Load pretrained backbone từ {PRETRAIN_BB_MODEL_PATH}")
        best_bb = ContrastiveBackbone(embed_dim=EMBED_DIM)
        best_bb = load_model(best_bb, PRETRAIN_BB_MODEL_PATH)
    else:
    # ========== CASE 3: Pretrain Backbone From Scratch ==========
        print("\n--- Pretraining Contrastive Backbone ---")
        light_p, light_ids, dark_p, dark_ids, _ = load_image_paths(
            ['normal'], ['darken_normal'], root=DATA_ROOT
        )

        light_map, dark_map = defaultdict(list), defaultdict(list)
        for i, p in zip(light_ids, light_p): light_map[i].append(p)
        for i, p in zip(dark_ids, dark_p):  dark_map[i].append(p)

        ds = PairedLightDarkDataset(light_map, dark_map, seed=SEED)
        ld = DataLoader(ds, batch_size=PRETRAIN_BATCH_SIZE, shuffle=True)

        bb = ContrastiveBackbone(embed_dim=EMBED_DIM)
        best_bb = pretrain(
            bb, ld,
            num_epochs=PRETRAIN_EPOCHS,
            lr=PRETRAIN_LR,
            patience=PRETRAIN_PATIENCE,
            device=str(device),
            use_early_stopping=True,
        )

    print("\n--- Fine-tuning ---")
    model = ContrastiveModel(
        pretrained_backbone=best_bb,
        embed_dim=EMBED_DIM,
        num_classes=num_classes,
        num_unfreeze_layers=UNFREEZE_LAYERS
    )

    best_model = train(
        model,
        trn_ld,
        val_ld,
        id_name_map,
        lr=FINETUNE_LR,
        lr_drop_ratio=LR_DROP_RATIO,
        num_epochs=FINETUNE_EPOCHS,
        unfreeze_epoch=UNFREEZE_EPOCH,
        num_layers_unfreeze=UNFREEZE_LAYERS,
        patience=FINETUNE_PATIENCE,
        device=device,
        with_archead=False
    )
    return best_model.to(device)

if __name__ == "__main__":
    seeds = [1, 10, 20, 100, 200]
    results_ours = {}

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = load_model(ContrastiveModel(ContrastiveBackbone()), 'eval_models/ours_model_seed_' + str(seed) + '_state_dict.pt').to(device)
        # model = get_model(use_finetuned_model=False, use_pretrained_bb=True)

        _, _, tst_ld, name_id_map = build_dataloaders(
            trn_split_list=TRN_SPLITS,
            tst_split_list=TST_SPLITS,
            root=DATA_ROOT,
            pct=LOW_LIGHT_SPLIT_PCT,
            batch_size=BATCH_SIZE,
            should_tta=False,
            seed=seed
        )

        id_name_map = {v: k for k, v in name_id_map.items()}
        results_ours[seed] = eval_model(
            model, tst_ld, id_name_map, device=device, with_archead=False
        )
        # save_model(model, 'eval_models/ours_model_seed_' + str(seed) + '.pt')
        
    results_baseline = {}

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        # model = BaselineModel().to(device)
        model = load_model(BaselineModel(), 'eval_models/baseline_model_seed_' + str(seed) + '_state_dict.pt').to(device)
        
        trn_ld, val_ld, tst_ld, name_id_map = build_dataloaders(
            trn_split_list=TRN_SPLITS,
            tst_split_list=TST_SPLITS,
            root=DATA_ROOT,
            pct=LOW_LIGHT_SPLIT_PCT,
            batch_size=BATCH_SIZE,
            should_tta=False,
            seed=seed
        )
        # model = train(model, trn_ld, val_ld, id_name_map, num_epochs=60, unfreeze_epoch=5, num_layers_unfreeze=0, patience=5, with_archead=False)

        id_name_map = {v: k for k, v in name_id_map.items()}
        results_baseline[seed] = eval_model(
            model, tst_ld, id_name_map, device=device, with_archead=False
        )
        # save_model(model, 'eval_models/baseline_model_seed_' + str(seed) + '.pt')
        
    print("\n===== BASELINE RESULTS =====")
    for seed, eval_results in results_baseline.items():
        print(f"\n--- Seed {seed} ---")
        for k in ['val_loss', 'balanced_acc', 'f1_micro', 'f1_macro']:
            print(f"{k}: {eval_results[k]}")

    print("\n===== OUR MODEL RESULTS =====")
    for seed, eval_results in results_ours.items():
        print(f"\n--- Seed {seed} ---")
        for k in ['val_loss', 'balanced_acc', 'f1_micro', 'f1_macro']:
            print(f"{k}: {eval_results[k]}")