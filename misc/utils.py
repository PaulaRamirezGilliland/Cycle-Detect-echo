import SimpleITK as sitk
import numpy as np


def read_img(filename):
    img = sitk.ReadImage(filename)
    arr = sitk.GetArrayFromImage(img)
    return arr


def get_bound_box(mask):
    non_zero_pixels = np.argwhere(mask == 1)
    x_min, y_min = non_zero_pixels.min(axis=0)
    x_max, y_max = non_zero_pixels.max(axis=0)
    bbox = [x_min, y_min, x_max, y_max]

    return bbox


# Extract bounding box coordinates for MedSAM
