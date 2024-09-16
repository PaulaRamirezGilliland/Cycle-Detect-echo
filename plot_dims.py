import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

path = "C:\\Users\\prg20local\\OneDrive - King's College London\\Research Project\\PhD\\US_recon\\Data\\2D_echo\\current_training_data\\cropped-files-heart\\iFIND00226_10Mar2017\\"
filename = "crop-heart-IM_0084-res"
embedding = ["PCA", "SE", "LLE"]

n_comp = [2, 3, 5, 10, 15, 20]


comp=3
plt.figure(figsize=(15, 5))
for i, method in enumerate(embedding):
    df = pd.read_csv(os.path.join(path, filename + '-' + method + '.csv'))

    plt.subplot(1, 3, i+1)
    plt.title(f"{method}, N_components = {comp}")

    # Extract data
    x = df[f'N_comp_{comp}_dim_0']
    y = df[f'N_comp_{comp}_dim_1']
    frame_number = np.arange(len(x))

    # Create a scatter plot
    scatter = plt.scatter(x, y, c=frame_number, cmap='viridis', edgecolor='k', marker='o')  # or 'x' for crosses
    plt.xlabel('dim 0')
    plt.ylabel('dim 1')

    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Frame Number')

plt.tight_layout()
plt.savefig(os.path.join(path, "dim0_1.svg"),format='svg', transparent=True)
plt.show()



comp=3
plt.figure(figsize=(15, 5))
for i, method in enumerate(embedding):
    df = pd.read_csv(os.path.join(path, filename + '-' + method + '.csv'))

    ax = plt.subplot(1, 3, i+1, projection='3d')
    ax.set_title(f"{method}, N_components = {comp}")

    # Extract data
    x = df[f'N_comp_{comp}_dim_0']
    y = df[f'N_comp_{comp}_dim_1']
    z = df[f'N_comp_{comp}_dim_2']

    frame_number = np.arange(len(x))

    # Create a scatter plot
    scatter = ax.scatter(x, y, z, c=frame_number, cmap='viridis', edgecolor='k')

    # Set labels
    ax.set_xlabel('Dim 0')
    ax.set_ylabel('Dim 1')
    ax.set_zlabel('Dim 2')

    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Frame Number')

plt.tight_layout()
plt.savefig(os.path.join(path, "3d_3comp.svg"),format='svg', transparent=True)
plt.show()

