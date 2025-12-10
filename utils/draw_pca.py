import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from datetime import datetime
from utils.get_embedding import get_embedding

def draw_pca(class_to_paths: dict, save_path="pca_plot.png"):
    """
    class_to_paths: dict[str, list[str]]
        Ví dụ:
        {
            "low_light": [...],
            "normal": [...],
            "classA": [...],
        }
    """

    # 1. Lấy embedding cho từng class
    class_embeddings = {}
    for cls, paths in class_to_paths.items():
        embs = []
        for f in paths:
            embs.append(get_embedding(f))
        class_embeddings[cls] = np.vstack(embs)

    # 2. Ghép tất cả embedding
    all_embs = np.vstack(list(class_embeddings.values()))

    # 3. PCA -> 2D
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(all_embs)

    # 4. Tách lại theo class
    splitted = {}
    idx = 0
    for cls, embs in class_embeddings.items():
        n = len(embs)
        splitted[cls] = emb_2d[idx:idx+n]
        idx += n

    # 5. Vẽ scatter
    plt.figure(figsize=(8, 6))
    for cls, pts in splitted.items():
        plt.scatter(pts[:, 0], pts[:, 1], label=cls, alpha=0.7)

    plt.legend()
    plt.title("PCA Projection of Face Embeddings")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    # 6. Lưu file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = save_path.replace(".png", f"_{timestamp}.png")
    plt.savefig(final_path, dpi=300)
    plt.close()

    return final_path
