import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import matplotlib
matplotlib.use('TkAgg')

path = "C:\\Users\\prg20local\\OneDrive - King's College London\\Research Project\\PhD\\US_recon\\Data\\2D_echo\\current_training_data\\cropped-files-heart\\iFIND00226_10Mar2017\\"
filename = "crop-heart-IM_0084-res"


def read_img(filename):
    img = sitk.ReadImage(filename)
    arr = sitk.GetArrayFromImage(img)
    return arr


# Read in average images
avg_odd = read_img(os.path.join(path, filename + '-ev-avg.nii.gz'))
avg_ev = read_img(os.path.join(path, filename + '-odd-avg.nii.gz'))

# Read in registered images
reg_targ_odd = read_img(os.path.join(path, filename + '-ev-avg-reg-ds6.nii.gz'))
reg_targ_ev = read_img(os.path.join(path, filename + '-odd-avg-reg-ds6.nii.gz'))

# Read in Jacobian
jac_targ_ev = read_img(os.path.join(path, filename + '-odd-avg-jacobian-ds6.nii.gz'))
jac_targ_odd = read_img(os.path.join(path, filename + '-ev-avg-jacobian-ds6.nii.gz'))


colors = ['blue', 'red']  # Define colors for the two ranges
custom_cmap = ListedColormap(colors)
masked_jacobian_ev = np.where(jac_targ_ev < 100, 0, 1)  # Mask values < 1 as 0, others as 1
masked_jacobian_odd = np.where(jac_targ_odd < 100, 0, 1)  # Mask values < 1 as 0, others as 1


# Plot
plt.figure(figsize=(20,10))
plt.subplot(3, 2, 1)
plt.title("Average Odd Valleys")
plt.imshow(avg_odd, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.subplot(3, 2, 2)
plt.title("Average Even Valleys")
plt.imshow(avg_ev, cmap='gray')
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
plt.imshow(masked_jacobian_odd[0,...])
plt.colorbar(ticks=[0, 1], label='Jacobian Value Category')
plt.colorbar()
plt.xticks([])
plt.yticks([])
plt.subplot(3, 2, 6)
plt.title("Jacobian (target even)")
plt.imshow(masked_jacobian_ev[0,...])
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
