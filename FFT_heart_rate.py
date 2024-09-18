import os
import matplotlib
matplotlib.use('TkAgg')
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

# Read in Data
path = "C:\\Users\\prg20local\\OneDrive - King's College London\\Research Project\\PhD\\US_recon\\Data\\2D_echo\\current_training_data\\cropped-files-heart\\"
folder = "iFIND00226_10Mar2017"
filename = "crop-heart-IM_0084-res.nii.gz"

image = sitk.ReadImage(os.path.join(path, folder, filename))
image_array = sitk.GetArrayFromImage(image)
image_array = (image_array - np.min(image_array)) / (
        np.max(image_array) - np.min(image_array))


def calc_freq(nFrame, dt):
    iFdc = (nFrame // 2) + 1
    fMax = 1 / dt
    f = fMax * (np.arange(1, nFrame + 1) - iFdc) / nFrame
    return f


def estimate_heartrate_2d_time(imSeq, frameDuration, hrRange=(110, 180)):
    nFrames, height, width = imSeq.shape

    #flattened_data = image_array.reshape(image_array.shape[0], -1)

    # Step 1: Perform the Fourier transform along the first axis (N_frames/time axis)
    fft_result = np.fft.fftshift(np.fft.fft(imSeq, axis=0), axes=0)

    # Step 2: Take the magnitude of the Fourier transform
    fft_magnitude = np.abs(fft_result)

    # Step 3: Compute the mean over the spatial dimensions (Height, Width)
    xfMeanSig = np.mean(fft_magnitude, axis=(1, 2))  # Mean over Height and Width

    # Frequency vector
    f = calc_freq(nFrames, frameDuration)

    plt.figure()
    plt.plot(f, xfMeanSig)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Spatial Mean FFT Magnitude')
    plt.title('Frequency Spectrum')
    plt.show()

    # Define frequency range based on heart rate
    minHR = hrRange[0]
    maxHR = hrRange[1]
    minFreq = minHR / 60  # Convert BPM to Hz
    maxFreq = maxHR / 60  # Convert BPM to Hz

    # Identify peaks in the frequency spectrum within the HR range
    indf0 = np.where((f >= minFreq) & (f <= maxFreq))[0]

    plt.figure()
    plt.subplot(1,2,1)
    plt.title("Mean FFT mag")
    plt.plot(f, xfMeanSig)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Spatial Mean FFT Magnitude')
    plt.subplot(1,2,2)
    plt.title("Mean FFT mag in expected frequency range")
    plt.plot(f[indf0], xfMeanSig[indf0])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Spatial Mean FFT Magnitude')
    plt.show()

    peaks, properties = find_peaks(xfMeanSig[indf0], height=[0, np.max(xfMeanSig)+1])

    if len(peaks) == 0:
        return None, None  # No peaks found

    # Get the fundamental frequency corresponding to the highest peak
    peak_idx = peaks[np.argmax(properties['peak_heights'])]
    fundamental_frequency = f[indf0[peak_idx]]
    print("Fundamental frequency ", fundamental_frequency)
    # Calculate R-R interval and heart rate
    rrInterval = 1 / fundamental_frequency  # R-R interval in seconds
    heart_rate = 60 / rrInterval  # Convert to BPM

    return rrInterval, heart_rate

rrInterval, heart_rate = estimate_heartrate_2d_time(image_array, frameDuration=0.017266)
print(rrInterval)
print(heart_rate)
print('Done')