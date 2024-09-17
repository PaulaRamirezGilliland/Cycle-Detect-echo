
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import os

# Read in Data
path = "C:\\Users\\prg20local\\OneDrive - King's College London\\Research Project\\PhD\\US_recon\\Data\\2D_echo\\current_training_data\\cropped-files-heart\\"
folder = "iFIND00226_10Mar2017"
filename = "crop-heart-IM_0084-res.nii.gz"

image = sitk.ReadImage(os.path.join(path, folder, filename))
image_array = sitk.GetArrayFromImage(image)
image_array = (image_array - np.min(image_array)) / (
        np.max(image_array) - np.min(image_array))

time, height, width = image_array.shape

# Flatten the images
flattened_data = image_array.reshape(image_array.shape[0], -1)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def calc_freq(nFrame, dt):
    iFdc = (nFrame // 2) + 1
    fMax = 1 / dt
    f = fMax * (np.arange(1, nFrame + 1) - iFdc) / nFrame
    return f


def estimate_heartrate_2d_time(imSeq, frameDuration, hrRange=(110, 180)):
    nFrames, height, width = imSeq.shape

    # Flatten the image sequence for FFT processing
    xtRoi = imSeq.reshape(nFrames, -1)  # Shape: (nFrames, height * width)

    # Perform FFT on the entire sequence
    xfRoi = np.abs(np.fft.fft(xtRoi, axis=0))  # FFT along the time axis
    xfMeanSig = np.mean(xfRoi, axis=1)  # Mean signal across spatial dimensions

    # Frequency vector
    f = calc_freq(nFrames, frameDuration)

    # Define frequency range based on heart rate
    minHR = hrRange[0]
    maxHR = hrRange[1]
    minFreq = minHR / 60  # Convert BPM to Hz
    maxFreq = maxHR / 60  # Convert BPM to Hz

    # Identify peaks in the frequency spectrum within the HR range
    indf0 = np.where((f >= minFreq) & (f <= maxFreq))[0]
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(xfMeanSig, 'ro')
    plt.subplot(1,2,2)
    plt.plot(xfMeanSig[indf0], 'ro')
    plt.show()

    peaks, properties = find_peaks(xfMeanSig[indf0], height=[0, 100])

    if len(peaks) == 0:
        return None, None  # No peaks found

    # Get the fundamental frequency corresponding to the highest peak
    peak_idx = peaks[np.argmax(properties['peak_heights'])]
    fundamental_frequency = f[indf0[peak_idx]]

    # Calculate R-R interval and heart rate
    rrInterval = 1 / fundamental_frequency  # R-R interval in seconds
    heart_rate = 60 / rrInterval  # Convert to BPM

    return rrInterval, heart_rate

rrInterval, heart_rate = estimate_heartrate_2d_time(image_array, frameDuration=17.266*10**-3)

print('Done')
"""
# Perform Fourier Transform along the time axis
fft_result = np.abs(np.fft.fft(flattened_data, axis=0))
xfMeanSig = np.mean(fft_result, axis=1)  # Mean signal across spatial dimensions
# Get the frequencies
frequencies = np.fft.fftfreq(time, 17.266*10**-3)

minHR = 110
maxHR = 180
minFreq = minHR / 60  # Convert BPM to Hz
maxFreq = maxHR / 60  # Convert BPM to Hz

# Identify peaks in the frequency spectrum within the HR range
indf0 = np.where((frequencies >= minFreq) & (frequencies <= maxFreq))[0]

plt.figure()
plt.title("Mean signal")
plt.plot(xfMeanSig, 'ro')
plt.show()

plt.figure()
plt.title("Mean signal filter")
plt.plot(xfMeanSig[indf0], 'ro')
plt.show()


# Compute amplitude spectrum
amplitude = np.abs(fft_result)

# Plotting the results
plt.figure(figsize=(12, 6))
plt.imshow(amplitude.T, aspect='auto', extent=[frequencies.min(), frequencies.max(), 0, height * width], origin='lower')
plt.colorbar(label='Amplitude')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Spatial Component Index')
plt.title('Fourier Transform Amplitude Spectrum of Fetal Heart Ultrasound Data')
plt.show()
