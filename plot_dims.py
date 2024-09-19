import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

path = "C:\\Users\\prg20local\\OneDrive - King's College London\\Research Project\\PhD\\US_recon\\Data\\2D_echo\\current_training_data\\cropped-files-heart\\iFIND00226_10Mar2017\\"
filename = "embedding_scaling\\crop-heart-IM_0084-res"
embedding = ["PCA", "SE", "LLE", "DMD", "DMD_recon", "img"]

n_comp = [2, 3, 5, 10, 15, 20]

ED = [0, 22, 45, 70, 95, 116, 142]
ES = [10, 34, 58, 84, 106, 130, 154]



plt.figure(figsize=(20, 20)) # (15, 20)
ind_plot=1
for i, comp in enumerate(n_comp):

    for ind, method in enumerate(embedding):

        df = pd.read_csv(os.path.join(path, filename + '-' + method + '_dist.csv'))
        print("SUBPLOT IND ",  ind_plot)
        plt.subplot(6, 6, ind_plot)
        plt.title(f"{method}, N_components = {comp}")

        # Extract data
        x = df[f'N_comp_{comp}']
        frame_number = np.arange(len(x))

        # Create a scatter plot
        plt.plot(frame_number, x, 'g-', linewidth=0.4)
        plt.plot(ED, np.zeros(len(ED)), 'b*', label='ED')
        plt.plot(ES, np.zeros(len(ES)), 'r*', label='ES', linewidth=0.05)

        for x in ED:
            plt.axvline(x=x, color='blue', linestyle='--', alpha=0.5, linewidth=0.5)

        # Plot vertical lines for ES points
        for x in ES:
            plt.axvline(x=x, color='red', linestyle='--', alpha=0.5, linewidth=0.5)

        plt.xlabel('Frame number')
        plt.ylabel('ED')
        plt.legend()
        ind_plot += 1

plt.tight_layout()
plt.savefig(os.path.join(path, "plots",  "ED_all_embed_seq.svg"),format='svg', transparent=True)
plt.show()
"""

comp=3

method = "DMD"
df = pd.read_csv(os.path.join(path, filename + '-' + method + '.csv'))

plt.figure(figsize=(15, 5))
plt.subplot(1,3,1)
plt.title("DMD Comp. 0")
plt.plot(df[f'N_comp_{comp}_dim_0'])
plt.subplot(1,3,2)
plt.title("DMD Comp. 1")
plt.plot(df[f'N_comp_{comp}_dim_1'])
plt.subplot(1,3,3)
plt.title("DMD Comp. 2")
plt.plot(df[f'N_comp_{comp}_dim_2'])
plt.show()
#plt.savefig(os.path.join(path, filename + 'comp_DMD.svg'), format='svg')



plt.figure(figsize=(10, 10))
for i, method in enumerate(embedding):
    df = pd.read_csv(os.path.join(path, filename + '-' + method + '.csv'))

    ax = plt.subplot(2, 2, i+1, projection='3d')
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
plt.savefig(os.path.join(path, f"3d_3comp.svg"),format='svg', transparent=True)
plt.show()

comp=3
plt.figure(figsize=(10, 10))
for i, method in enumerate(embedding):
    df = pd.read_csv(os.path.join(path, filename + '-' + method + '.csv'))

    ax = plt.subplot(2, 2, i+1, projection='3d')
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
plt.savefig(os.path.join(path, f"3d_{comp}comp.svg"),format='svg', transparent=True)
plt.show()
"""