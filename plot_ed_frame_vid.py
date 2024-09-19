import SimpleITK as sitk
import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


path = "C:\\Users\\prg20local\\OneDrive - King's College London\\Research Project\\PhD\\US_recon\\Data\\2D_echo\\current_training_data\\cropped-files-heart\\iFIND00226_10Mar2017\\"
filename = "crop-heart-IM_0084-res"
embedding = "DMD"
comp = 20

# Load the data
df = pd.read_csv(os.path.join(path, 'single_cycle1', filename + '-' + embedding + '_dist.csv'))
x = df[f'N_comp_{comp}']

df_embed = pd.read_csv(os.path.join(path, 'single_cycle1', filename + '-' + embedding + '.csv'))
x_mode0 = df_embed[f'N_comp_{comp}_dim_0']
x_mode1 = df_embed[f'N_comp_{comp}_dim_1']
x_mode2 = df_embed[f'N_comp_{comp}_dim_2']


frame_number = np.arange(len(x))

# Ensure all arrays have the same length
min_length = min(len(frame_number), len(x_mode0), len(x_mode1), len(x_mode2))

# Truncate arrays to the minimum length
x = x[:min_length]
frame_number = frame_number[:min_length]
x_mode0 = x_mode0[:min_length]
x_mode1 = x_mode1[:min_length]
x_mode2 = x_mode2[:min_length]

# Load the NIfTI file for ultrasound frames
nifti_path = os.path.join(path, 'crop-heart-IM_0084-res.nii.gz')
nifti_image = sitk.ReadImage(nifti_path)
image_array = sitk.GetArrayFromImage(nifti_image)
num_frames, height, width = image_array.shape

# Load in reconstructed data
recon = np.load(os.path.join(path, 'embedding_scaling', f"recon_dmd_{comp}.npy"))
# Reshape
recon = recon.reshape(num_frames, height, width)


# Normalize ultrasound image frames for display
image_array = np.array([cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) for img in image_array])
recon_array = cv2.normalize(recon, None, 0, 1, cv2.NORM_MINMAX).astype(np.uint8)


# Create the figure and axis objects for the plot and image
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Ultrasound image
ultrasound_display = ax1.imshow(image_array[0], cmap='gray', aspect='auto')
ax1.axis('off')  # Hide the axes for the ultrasound image
ax1.set_title('Ultrasound Seq.')



# Plot the x array (ED)
line, = ax2.plot([], [], 'g-', linewidth=0.4)
ax2.set_xlim(0, len(x))
ax2.set_ylim(np.min(x), np.max(x))
ax2.set_xlabel("Frame Number")
ax2.set_ylabel(f"ED (DMD) N_comp_{comp}")

# Plot embedded dims
line_mode0, = ax3.plot([], [], 'r-', linewidth=0.6, label='Mode 0')
line_mode1, = ax3.plot([], [], 'b-', linewidth=0.6, label='Mode 1')
line_mode2, = ax3.plot([], [], 'k-', linewidth=0.6, label='Mode 2')
ax3.set_xlim(0, len(x_mode0))
ax3.set_ylim(min(np.min(x_mode0), np.min(x_mode1), np.min(x_mode2)),
             max(np.max(x_mode0), np.max(x_mode1), np.max(x_mode2)))
ax3.set_xlabel("Frame Number")
ax3.set_ylabel(f"DMD Modes N_comp_{comp}")
ax3.legend()


# Initialization function for the animation
def init():
    line.set_data([], [])
    line_mode0.set_data([], [])
    line_mode1.set_data([], [])
    line_mode2.set_data([], [])
    ultrasound_display.set_data(image_array[0])
    return line, line_mode0, line_mode1, line_mode2, ultrasound_display

# Update function for the animation
def update(frame):
    # Update the ultrasound image
    ultrasound_display.set_data(image_array[frame])

    # Update the x array plot
    line.set_data(frame_number[:frame+1], x[:frame+1])

    # Update the embedded modes plot
    line_mode0.set_data(frame_number[:frame+1], x_mode0[:frame+1])
    line_mode1.set_data(frame_number[:frame+1], x_mode1[:frame+1])
    line_mode2.set_data(frame_number[:frame+1], x_mode2[:frame+1])

    return line, line_mode0, line_mode1, line_mode2, ultrasound_display
# Create the animation
ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True, interval=1000/15)

# Define the path for saving the video
output_video_path = os.path.join(path, 'plots', f"synchronized_us_{embedding}_{comp}.mp4")

# Save the animation to an MP4 file using FFMpegWriter
writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
ani.save(output_video_path, writer=writer)

plt.tight_layout()
plt.show()

print(f"Video saved at {output_video_path}")
