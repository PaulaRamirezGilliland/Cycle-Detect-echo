import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


# Define file path and names
path = "C:\\Users\\prg20local\\OneDrive - King's College London\\Research Project\\PhD\\US_recon\\Data\\2D_echo\\current_training_data\\cropped-files-heart\\iFIND00226_10Mar2017\\"
filename = "crop-heart-IM_0084-res"

# Read even and odd images
img_ev = sitk.ReadImage(os.path.join(path, filename + '-ev.nii.gz'))
img_odd = sitk.ReadImage(os.path.join(path, filename + '-odd.nii.gz'))

# Voxelwise average
arr_ev = sitk.GetArrayFromImage(img_ev)
arr_odd = sitk.GetArrayFromImage(img_odd)

arr_ev_mean = np.mean(arr_ev, axis=0)
arr_odd_mean = np.mean(arr_odd, axis=0)

# Convert means back to SimpleITK images
mean_img_ev = sitk.GetImageFromArray(arr_ev_mean)
mean_img_odd = sitk.GetImageFromArray(arr_odd_mean)


# Set the origin and spacing
mean_img_ev.SetOrigin(img_ev.GetOrigin())
mean_img_ev.SetSpacing(img_ev.GetSpacing())
mean_img_odd.SetOrigin(img_odd.GetOrigin())
mean_img_odd.SetSpacing(img_odd.GetSpacing())

sitk.WriteImage(mean_img_ev, os.path.join(path, filename + '-ev-avg.nii.gz'))
sitk.WriteImage(mean_img_odd, os.path.join(path, filename + '-odd-avg.nii.gz'))




