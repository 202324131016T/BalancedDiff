import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def get_TSNE(save_path="../figures/1.pdf", data=None):
    # 初始化t-SNE对象，设置降维后的空间维度为2
    tsne = TSNE(n_components=2, init='pca', random_state=42)
    # 将Tensor转换为numpy数组，因为t-SNE需要numpy数组作为输入
    tensor_data_np = data.cpu().numpy()
    # 执行t-SNE降维
    tsne_result = tsne.fit_transform(tensor_data_np)

    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], cmap='viridis', alpha=1, s=3)
    # plt.title('t-SNE visualization')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')

    plt.subplots_adjust(left=0.07, bottom=0.07, right=0.99, top=0.99)

    # 保存图片，设置DPI为300
    # plt.savefig(save_path, dpi=300)
    plt.savefig(save_path)

    # plt.show()
