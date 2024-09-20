import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from monai.transforms import ResizeWithPadOrCrop
from scipy.ndimage import binary_erosion


import matplotlib
matplotlib.use('TkAgg')

path = "C:\\Users\\prg20local\\OneDrive - King's College London\\Research Project\\PhD\\US_recon\\Data\\2D_echo\\current_training_data\\cropped-files-heart\\iFIND00226_10Mar2017\\ED_ES_reg\\"
filename = "crop-heart-IM_0084-res"


def read_img(filename):
    img = sitk.ReadImage(filename)
    arr = sitk.GetArrayFromImage(img)
    return arr


# Read in average images
avg_odd = read_img(os.path.join(path, filename + '-ev-avg.nii.gz'))
avg_ev = read_img(os.path.join(path, filename + '-odd-avg.nii.gz'))

# Read in registered images
reg_targ_odd = read_img(os.path.join(path, filename + '-ev-avg-reg-ds6-mask.nii.gz'))
reg_targ_ev = read_img(os.path.join(path, filename + '-odd-avg-reg-ds6-mask.nii.gz'))

# Read in Jacobian
jac_targ_ev = read_img(os.path.join(path, filename + '-odd-avg-jacobian-ds6-mask.nii.gz'))
jac_targ_odd = read_img(os.path.join(path, filename + '-ev-avg-jacobian-ds6-mask.nii.gz'))

# Read in predicted mask
mask_heart = read_img(os.path.join(path, 'seg-IM_0084-res.nii.gz'))[0, ...]
# Crop to image size
tf = ResizeWithPadOrCrop(avg_odd.shape)
mask_heart = tf(np.expand_dims(mask_heart, 0))[0, ...]
# Erode

# Define the structuring element for erosion (e.g., 3x3 square)
structuring_element = np.ones((3, 3))

# Perform the erosion
eroded_segmentation = binary_erosion(mask_heart, structure=structuring_element, iterations=30)

# Convert boolean output to integers if needed (0s and 1s)
eroded_segmentation = eroded_segmentation.astype(int)

colors = ['blue', 'red']  # Define colors for the two ranges
custom_cmap = ListedColormap(colors)
masked_jacobian_ev = np.where(jac_targ_ev < 100, 0, 1)  # Mask values < 1 as 0, others as 1
masked_jacobian_odd = np.where(jac_targ_odd < 100, 0, 1)  # Mask values < 1 as 0, others as 1


# Plot
plt.figure(figsize=(20,10))
plt.subplot(3, 2, 1)
plt.title("Average Odd Valleys")
plt.imshow(avg_odd, cmap='gray')
plt.imshow(mask_heart, alpha=0.5)
plt.xticks([])
plt.yticks([])
plt.subplot(3, 2, 2)
plt.title("Average Even Valleys")
plt.imshow(avg_ev, cmap='gray')
plt.imshow(eroded_segmentation, alpha=0.5)
plt.xticks([])
plt.yticks([])
plt.subplot(3, 2, 3)
plt.title("Target Odd Valleys")
plt.imshow(reg_targ_odd[0,...], cmap='gray')
plt.xticks([])
plt.yticks([])
plt.subplot(3, 2, 4)
plt.title("Target Even Valleys")
plt.imshow(reg_targ_ev[0,...], cmap='gray')
plt.xticks([])
plt.yticks([])
plt.subplot(3, 2, 5)
plt.title("Jacobian (target odd)")
#plt.imshow(jac_targ_odd[0,...])
plt.imshow(masked_jacobian_odd[0,...]*eroded_segmentation)
plt.colorbar(ticks=[0, 1], label='Jacobian Value Category')
plt.colorbar()
plt.xticks([])
plt.yticks([])
plt.subplot(3, 2, 6)
plt.title("Jacobian (target even)")
plt.imshow(masked_jacobian_ev[0,...]*eroded_segmentation)
plt.colorbar()
plt.xticks([])
plt.yticks([])
plt.show()

plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
plt.title("Target Odd Valleys")
plt.imshow(reg_targ_odd[0,...], cmap='gray')
plt.imshow(jac_targ_odd[0,...], alpha=0.5)
plt.colorbar()
plt.xticks([])
plt.yticks([])
plt.subplot(1,2,2)
plt.title("Target Even Valleys")
plt.imshow(reg_targ_ev[0,...], cmap='gray')
plt.imshow(jac_targ_ev[0,...], alpha=0.5)
plt.colorbar()
plt.xticks([])
plt.yticks([])
plt.show()
