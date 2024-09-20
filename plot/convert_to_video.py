import SimpleITK as sitk
import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


path = "C:\\Users\\prg20local\\OneDrive - King's College London\\Research Project\\PhD\\US_recon\\Data\\2D_echo\\current_training_data\\cropped-files-heart\\iFIND00226_10Mar2017\\"
filename = "crop-heart-IM_0084-res"


def nifti_to_video(input_nifti_path, output_video_path, fps=30):
    # Load the NIfTI file using SimpleITK
    nifti_image = sitk.ReadImage(input_nifti_path)

    # Get the 3D numpy array from the NIfTI file (assumed to be 2D+t ultrasound images)
    image_array = sitk.GetArrayFromImage(nifti_image)

    # Get the dimensions of the image
    num_frames, height, width = image_array.shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=False)

    for frame_idx in range(num_frames):
        # Extract the 2D frame from the 3D+t NIfTI file
        frame = image_array[frame_idx]

        # Normalize the frame to 0-255 for display (assuming grayscale)
        frame_normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)

        # Convert the frame to uint8 (which is required by OpenCV)
        frame_uint8 = np.uint8(frame_normalized)

        # Write the frame to the video
        video_writer.write(frame_uint8)

    # Release the video writer object
    video_writer.release()

    print(f"Video saved at {output_video_path}")


PATH = "C:\\Users\\prg20local\\OneDrive - King's College London\\Research Project\\PhD\\US_recon\\Data\\2D_echo\\current_training_data\\cropped-files-heart\\iFIND00270_30Jun2017\\"
input_nifti_path = os.path.join(PATH, 'crop-heart-IM_0315-res.nii.gz')  # Path to your input NIfTI file
output_video_path = os.path.join(PATH, 'vid_IM_0315.mp4')  # Output video path
nifti_to_video(input_nifti_path, output_video_path, fps=15)