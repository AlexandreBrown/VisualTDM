import torch
import matplotlib
import matplotlib.pyplot as plt


def get_images_grid(images: torch.Tensor, num_rows=6, num_cols=6) -> matplotlib.figure.Figure:
    fig, ax = plt.subplots(num_rows, num_cols, figsize=(15, 15))
    ax = ax.ravel()
    for i in range(num_rows * num_cols):
        img = images[i].permute(1, 2, 0)
        ax[i].imshow(img.numpy())
        ax[i].axis('off')
    fig.tight_layout()
    return fig
