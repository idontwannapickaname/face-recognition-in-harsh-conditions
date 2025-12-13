from module.models import BaselineModel
from module.training import build_dataloaders, train
from module.evaluation import eval_model


def main():
    seed = 200
    trn_ld, val_ld, tst_ld, name_id_map = build_dataloaders(['normal'], ['low_light'], root="data", pct=0.5, batch_size=4, seed=seed)
    model = BaselineModel(embed_dim=512, num_classes=len(name_id_map))
    model = train(model, trn_ld, val_ld, name_id_map, num_epochs=60, unfreeze_epoch=5, num_layers_unfreeze=0, patience=5, with_archead=False)
    metrics = eval_model(model, tst_ld, {v: k for k, v in name_id_map.items()}, with_archead=False)
    print(metrics)


if __name__ == "__main__":
    main()
