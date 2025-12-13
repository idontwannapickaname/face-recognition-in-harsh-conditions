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
from module.models import load_model, save_model, ContrastiveBackbone, ContrastiveModel
from module.evaluation import eval_model

SEED = 200
# Chọn thiết bị: CUDA nếu có, ngược lại là CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
torch.manual_seed(SEED)
np.random.seed(SEED)

# Hyperparameters
# Dữ liệu
DATA_ROOT = "data"
TRN_SPLITS = ['normal', 'darken_normal'] # Tập normal và ảnh tối giả (cho finetune)
TST_SPLITS = ['low_light']              # Tập low_light để chia Val/Test
LOW_LIGHT_SPLIT_PCT = 0.5               # 50% low_light cho train, 25% val, 25% test
BATCH_SIZE = 8

# Pre-training Contrastive
PRETRAIN_EPOCHS = 50 
PRETRAIN_PATIENCE = 15
PRETRAIN_LR = 1e-3
PRETRAIN_BATCH_SIZE = 256
PRETRAIN_BB_MODEL_PATH = 'models/best_bb_state_dict.pt'

# Fine-tuning
FINETUNE_EPOCHS = 60
FINETUNE_PATIENCE = 5
FINETUNE_LR = 1e-3
LR_DROP_RATIO = 0.1
UNFREEZE_EPOCH = 5
UNFREEZE_LAYERS = 2 # Mở khóa 2 layer cuối của backbone
FINETUNE_MODEL_PATH = 'models/best_finetune_model.pt'
EMBED_DIM = 512

print("\n--- 0. Chuẩn bị DataLoaders cho Finetune (Classification) ---")
try:
    trn_ld, val_ld, tst_ld, name_id_map = build_dataloaders(
        trn_split_list=TRN_SPLITS, 
        tst_split_list=TST_SPLITS, 
        root=DATA_ROOT, 
        pct=LOW_LIGHT_SPLIT_PCT, 
        batch_size=BATCH_SIZE, 
        should_tta=True, # Bật TTA cho tập Test
        seed=SEED
    )
    num_classes = len(name_id_map)
    id_name_map = {v: k for k, v in name_id_map.items()}
    print(f"✅ DataLoaders đã sẵn sàng.")
    print(f"Số lượng lớp (người): {num_classes}")
    print(f"Kích thước tập Train/Val/Test: {len(trn_ld.dataset)}/{len(val_ld.dataset)}/{len(tst_ld.dataset)}")

except Exception as e:
    print(f"❌ Lỗi khi tải dữ liệu. Đảm bảo cấu trúc thư mục '{DATA_ROOT}/...' là chính xác.")
    print(f"Lỗi: {e}")
    exit()

print("\n--- 1. Bắt đầu Pre-training Contrastive (Light-Dark Pairs) ---")

try:
    light_paths, light_ids, dark_paths, dark_ids, _ = load_image_paths(['normal'], ['darken_normal_for_finetune'], root=DATA_ROOT)
    light_map = defaultdict(list)
    dark_map = defaultdict(list)
    for id, path in zip(light_ids, light_paths):
        light_map[id].append(path)
    for id, path in zip(dark_ids, dark_paths):
        dark_map[id].append(path)

    pretrn_ds = PairedLightDarkDataset(light_map, dark_map, seed=SEED)
    pretrn_ld = DataLoader(pretrn_ds, batch_size=PRETRAIN_BATCH_SIZE, shuffle=True) 

    # Khởi tạo Contrastive Backbone
    pretrained_bb = ContrastiveBackbone(embed_dim=EMBED_DIM)

    # Huấn luyện
    best_bb = pretrain(
        pretrained_bb,
        pretrn_ld,
        num_epochs=PRETRAIN_EPOCHS,
        lr=PRETRAIN_LR,
        patience=PRETRAIN_PATIENCE,
        device=str(device),
        use_pretrain=True,
        use_early_stopping=True,
    )

    # Lưu checkpoint của backbone đã pre-train tốt nhất
    save_model(best_bb, PRETRAIN_BB_MODEL_PATH)
    print("--- Kết thúc Pre-training ---")

except ValueError as e:
    print(f"⚠️ Bỏ qua Pre-training do lỗi dữ liệu: {e}")
    # Nếu không thể pretrain, tải lại checkpoint nếu có
    try:
        if os.path.exists(PRETRAIN_BB_MODEL_PATH.replace('.pt', '_state_dict.pt')):
            print(f"Đang tải backbone đã lưu từ {PRETRAIN_BB_MODEL_PATH}...")
            best_bb = ContrastiveBackbone(embed_dim=EMBED_DIM)
            best_bb = load_model(best_bb, PRETRAIN_BB_MODEL_PATH)
        else:
            print("Không có checkpoint pre-train nào được tìm thấy. Sử dụng backbone khởi tạo ngẫu nhiên.")
            best_bb = ContrastiveBackbone(embed_dim=EMBED_DIM) # Tải lại model ban đầu
            
    except Exception as e:
        print(f"Lỗi khi tải checkpoint: {e}")
        exit()


print("\n--- 2. Bắt đầu Fine-tuning (Simple Head) ---")

# Khởi tạo mô hình Finetune với backbone đã pre-train (hoặc đã tải)
finetune_model = ContrastiveModel(
    pretrained_backbone=best_bb, 
    embed_dim=EMBED_DIM, 
    num_classes=num_classes, 
    num_unfreeze_layers=UNFREEZE_LAYERS
)

# Huấn luyện Finetune
best_finetune_model = train(
    finetune_model, 
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

# Lưu checkpoint của mô hình finetune tốt nhất
save_model(best_finetune_model, FINETUNE_MODEL_PATH)
print("--- Kết thúc Fine-tuning ---")

print("\n--- 3. Đánh giá cuối cùng trên tập Test ---")

# Đánh giá với TTA (Test-Time Augmentation)
test_results = eval_model(
    best_finetune_model, 
    tst_ld, 
    id_name_map, 
    device=device,
    with_archead=False # Giả định không dùng ArcFace cho mô hình này
)

print("\n=============================================")
print("             KẾT QUẢ TEST CUỐI CÙNG           ")
print("=============================================")
print(f"🔥 Overall Accuracy (Có TTA): {test_results['overall_acc']:.4f}")
print(f"🔥 Balanced Accuracy: {test_results['balanced_acc']:.4f}")
print(f"Loss: {test_results['val_loss']:.4f}")
print("-" * 45)
print("Metrics chi tiết:")
print(f"F1-Score (Weighted): {test_results['f1_weighted']:.4f}")
print(f"Precision (Weighted): {test_results['precision_weighted']:.4f}")
print(f"Recall (Weighted): {test_results['recall_weighted']:.4f}")
print("-" * 45)
print("Accuracy theo lớp:")
for person, acc in test_results['class_acc'].items():
     print(f"  {person:<10}: {acc:.4f}")
print("-" * 45)
print("Confusion Matrix:\n", test_results['confusion_matrix'])
print("=============================================")