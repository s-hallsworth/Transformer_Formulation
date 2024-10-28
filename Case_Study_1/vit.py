
# Code adapted from: https://github.com/shub-garg/Vision-Transformer-VIT-for-MNIST/blob/main/Vision_Transformer_for_MNIST.ipynb
# # Transformers in Computer Vision
# 
# 
# 
# Transformer architectures owe their origins in natural language processing (NLP), and indeed form the core of the current state of the art models for most NLP applications.
# 
# We will now see how to develop transformers for processing image data (and in fact, this line of deep learning research has been gaining a lot of attention in 2021). The *Vision Transformer* (ViT) introduced in [this paper](https://arxiv.org/pdf/2010.11929.pdf) shows how standard transformer architectures can perform very well on image. The high level idea is to extract patches from images, treat them as tokens, and pass them through a sequence of transformer blocks before throwing on a couple of dense classification layers at the very end.
# 
# 
# Some caveats to keep in mind:
# - ViT models are very cumbersome to train (since they involve a ton of parameters) so budget accordingly.
# - ViT models are a bit hard to interpret (even more so than regular convnets).
# - Finally, while in this notebook we will train a transformer from scratch, ViT models in practice are almost always *pre-trained* on some large dataset (such as ImageNet) before being transferred onto specific training datasets.
# 
# 
# 
# 

# # Setup
# 
# As usual, we start with basic data loading and preprocessing.

# !pip install einops

# # %%
# !pip install torchinfo

# # %%
# !pip install torchvision

import torch
from torch import nn
from torch import nn, einsum
import torch.nn.functional as F
from torch import optim

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
import torchvision
import time
from torchinfo import summary

torch.manual_seed(42)

DOWNLOAD_PATH = '/data/mnist'
BATCH_SIZE_TRAIN = 100
BATCH_SIZE_TEST = 1000

transform_mnist = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
                               #torchvision.transforms.Normalize((0.1307,), (0.3081,))])

train_set = torchvision.datasets.MNIST(DOWNLOAD_PATH, train=True, download=True,
                                       transform=transform_mnist)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True)

test_set = torchvision.datasets.MNIST(DOWNLOAD_PATH, train=False, download=True,
                                      transform=transform_mnist)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE_TEST, shuffle=True)


import matplotlib.pyplot as plt

# Function to show a batch of images
def show_batch(data_loader):
    batch = next(iter(data_loader))
    images, labels = batch
    print(torch.max(images), torch.min(images))
    grid = torchvision.utils.make_grid(images, nrow=10)
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.title("Batch of Training Images")
    plt.axis('off')
    plt.show()

# #Print images for validation
# show_batch(train_loader)

# 
# 
# 
# 
# # The ViT Model
from vit_TNN import *

# %% [markdown]
# We will now set up the ViT model. There will be 3 parts to this model:
# 
# * A ``patch embedding'' layer that takes an image and tokenizes it. There is some amount of tensor algebra involved here (since we have to slice and dice the input appropriately), and the `einops` package is helpful. We will also add learnable positional encodings as parameters.
# * A sequence of transformer blocks. This will be a smaller scale replica of the original proposed ViT, except that we will only use 4 blocks in our model (instead of 32 in the actual ViT).
# * A (dense) classification layer at the end.
# 
# Further, each transformer block consists of the following components:
# 
# * A *self-attention* layer with $H$ heads,
# * A one-hidden-layer (dense) network to collapse the various heads. For the hidden neurons, the original ViT used something called a [GeLU](https://arxiv.org/pdf/1606.08415.pdf) activation function, which is a smooth approximation to the ReLU. For our example, regular ReLUs seem to be working just fine. The original ViT also used Dropout but we won't need it here.
# * *layer normalization* preceeding each of the above operations.
# 
# Some care needs to be taken in making sure the various dimensions of the tensors are matched.

# %%


# %%
# original:
#model = ViT(image_size=28, patch_size=4, num_classes=10, channels=1, dim=64, depth=6, heads=4, mlp_dim=128)
#optimizer = optim.Adam(model.parameters(), lr=0.003)
import csv
import os

image_size=28
patch_size=4 
num_classes=10
channels=1


# dim=18
# depth=2
heads=6
# mlp_dim=32
path = r".\trained_transformer\verification"
csv_file_path = path+"\\vit_results.csv"
columns = ['name', 'dim', 'depth', 'heads', 'mlp_dim', 'avg_test_loss', 'test_accuracy']


for dim in [6, 12, 18, 24]:
    for depth in [1, 2, 4]:
        for mlp_dim in [12, 24, 32, 64]:
            dim_head = int(dim/heads)
            model = ViT(image_size=image_size, patch_size=patch_size, num_classes=num_classes, channels=channels, dim=dim, dim_head=dim_head, depth=depth, heads=heads, mlp_dim=mlp_dim)
            optimizer = optim.Adam(model.parameters(), lr=0.003)

            summary(model)


            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)

            print(count_parameters(model))


            def train_epoch(model, optimizer, data_loader, loss_history):
                total_samples = len(data_loader.dataset)
                model.train()

                for i, (data, target) in enumerate(data_loader):
                    optimizer.zero_grad()
                    output = F.log_softmax(model(data), dim=1)
                    loss = F.nll_loss(output, target)
                    loss.backward()
                    optimizer.step()

                    if i % 100 == 0:
                        print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                            ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                            '{:6.4f}'.format(loss.item()))
                        loss_history.append(loss.item())


            def evaluate(model, data_loader, loss_history):
                model.eval()

                total_samples = len(data_loader.dataset)
                correct_samples = 0
                total_loss = 0

                with torch.no_grad():
                    for data, target in data_loader:
                        output = F.log_softmax(model(data), dim=1)
                        loss = F.nll_loss(output, target, reduction='sum')
                        _, pred = torch.max(output, dim=1)

                        total_loss += loss.item()
                        correct_samples += pred.eq(target).sum()

                avg_loss = total_loss / total_samples
                loss_history.append(avg_loss)
                print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
                    '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
                    '{:5}'.format(total_samples) + ' (' +
                    '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)\n')
                return '{:.4f}'.format(avg_loss) , '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%'
            N_EPOCHS = 10

            start_time = time.time()


            train_loss_history, test_loss_history = [], []
            for epoch in range(1, N_EPOCHS + 1):
                print('Epoch:', epoch)
                train_epoch(model, optimizer, train_loader, train_loss_history)
                evaluate(model, test_loader, test_loss_history)

            print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')

            name= f"vit_{dim}_{depth}_{heads}_{mlp_dim}"
            torch.save(model, path+f"\{name}.pt")

            print(name)
            avg_test_loss, test_accuracy = evaluate(model, test_loader, test_loss_history)

            # save: 
            file_exists = os.path.isfile(csv_file_path)
            with open(csv_file_path, mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=columns)

                # Write the header only if the file does not exist
                if not file_exists:
                    writer.writeheader()

                # Write the data row
                writer.writerow({
                    'name': name,
                    'dim': dim,
                    'depth': depth,
                    'heads': heads,
                    'mlp_dim': mlp_dim,
                    'avg_test_loss': avg_test_loss,
                    'test_accuracy': test_accuracy
                })

# # %%
# import seaborn as sns
# from sklearn.metrics import confusion_matrix
# import numpy as np

# # Confusion matrix
# def plot_confusion_matrix(model, data_loader):
#     model.eval()
#     all_preds = []
#     all_labels = []

#     with torch.no_grad():
#         for data, target in data_loader:
#             output = model(data)
#             _, preds = torch.max(output, dim=1)
#             all_preds.extend(preds.numpy())
#             all_labels.extend(target.numpy())

#     cm = confusion_matrix(all_labels, all_preds)
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.title('Confusion Matrix')
#     plt.show()

# plot_confusion_matrix(model, test_loader)

# # %%
# # Visualize correct and wrong predictions
# def plot_predictions(model, data_loader):
#     model.eval()
#     all_preds = []
#     all_labels = []
#     all_images = []

#     with torch.no_grad():
#         for data, target in data_loader:
#             output = model(data)
#             _, preds = torch.max(output, dim=1)
#             all_preds.extend(preds.numpy())
#             all_labels.extend(target.numpy())
#             all_images.extend(data.numpy())

#     all_preds = np.array(all_preds)
#     all_labels = np.array(all_labels)
#     all_images = np.array(all_images)

#     # Plot example images with predictions
#     fig, axes = plt.subplots(2, 5, figsize=(15, 6))
#     axes = axes.flatten()
#     for i, ax in enumerate(axes):
#         ax.imshow(all_images[i][0], cmap='gray')
#         ax.set_title(f'Label: {all_labels[i]}\nPred: {all_preds[i]}')
#         ax.axis('off')
#     plt.tight_layout()
#     plt.show()

# plot_predictions(model, test_loader)


