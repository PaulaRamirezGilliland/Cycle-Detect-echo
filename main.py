import os

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding, SpectralEmbedding
import pandas as pd


class CycleDetect:
    def __init__(self, path, folders_list, n_components_list = [2, 3, 5, 10, 15, 20], image_name=None):
        """
        Initializes the CycleDetect class with a path and folders.
        :param path: global path where folders are present
        :param folders_list: list of iFIND folders (cases) with 2D images
        :param n_components_list: list of integers containing number of components to test for
        """
        self.path = path
        self.folders_list = folders_list
        self.n_components_list = n_components_list
        self.image_name = image_name

        self.out_data_list, self.image_array_list, self.case_list, self.filename_list, self.itk_image_list = self.read_images_prep()

    def read_images_prep(self):
        """
        Function to loop through different folders, read images (.nii.gz) and flatten
        :return: list containing data for each case (kept in different lists)
        """
        out_data_list, case_list, case_data_list, filename_list, image_array_list, itk_image_list = [], [], [], [], [], []

        for folder in self.folders_list:
            if isinstance(self.image_name, list):
                image_names = self.image_name
            else:
                image_names = os.listdir(os.path.join(self.path, folder))
            for filename in image_names:
                if filename.find('-ev')==-1 and filename.find('-odd')==-1:
                    image = sitk.ReadImage(os.path.join(self.path, folder, filename))
                    image_array = sitk.GetArrayFromImage(image)
                    filename_list.append(filename)
                    # Flatten the images
                    flattened_images = image_array.reshape(image_array.shape[0], -1)
                    case_data_list.append(flattened_images)
                    out_data_list.append(case_data_list)
                    case_list.append(folder)
                    image_array_list.append(image_array)
                    itk_image_list.append(image)
        return out_data_list, image_array_list, case_list, filename_list, itk_image_list


    def pca_ed(self, flattened_data, plot=False):

       distances_list = []
       for n_components in self.n_components_list:

           # TODO - ADD in technique to use
           # TODO - to github
           # TODO - save to separate csv
           # TODO - modify returns
           # PCA
           pca = PCA(n_components=n_components)
           pca_result = pca.fit_transform(flattened_data)

           # LLE
           lle = LocallyLinearEmbedding(n_neighbors=10, n_components=2)
           lle_result = lle.fit_transform(flattened_data)

           # Spectral embedding

           # FFT


           distances = np.linalg.norm(np.diff(pca_result, axis=0), axis=1)
           distances_list.append(distances)

       if plot:
            fig, axs = plt.subplots(3, 2, figsize=(14, 10))
            axs = axs.flatten()

            for i, distances in enumerate(distances_list):
                # Plot Euclidean Distance
                axs[i].plot(distances, label=f'Components={self.n_components_list[i]}')
                axs[i].set_title(f'Euclidean Distance (Components={self.n_components_list[i]})')
                axs[i].set_xlabel('Frame Number')
                axs[i].set_ylabel('Euclidean Distance')
                axs[i].legend()
                axs[i].grid(True)

            plt.tight_layout()
            plt.show()

       return distances_list

    def find_best_component(self, distances_list):

        min_valley_interval_std_dev = float('inf')

        # Loop through the number of components, and the list of distances for each component
        for n_components, distances in zip(self.n_components_list, distances_list):
            # Find valleys for that particular component
            valleys, _ = find_peaks(-distances, prominence=2, distance=5)
            if len(valleys)<2:
                print("Only found {} valleys!".format(len(valleys)))
                best_start = None
                best_end = None
                best_window_valley = None
                best_component = None
                best_distances = None
                best_valleys = None
                best_combined_std_dev = None
                continue

            # Only check valid windows that span exactly 6 valleys
            for i in range(len(valleys) - 5):
                window_valleys = valleys[i:i + 6]  # Get a set of exactly 6 valleys
                start, end = window_valleys[0], window_valleys[-1]

                # Calculate the valley intervals and standard deviation within the window
                valley_intervals = np.diff(window_valleys)
                valley_interval_std_dev = np.std(valley_intervals)
                window_std_dev = np.std(distances[start:end])

                combined_std_dev = (valley_interval_std_dev + window_std_dev)/2

                # Find the minimal standard deviation
                if combined_std_dev < min_valley_interval_std_dev:
                    min_valley_interval_std_dev = combined_std_dev
                    best_start = start
                    best_end = end
                    best_window_valley = window_valleys
                    best_component = n_components
                    best_distances = distances
                    best_valleys = valleys
                    best_combined_std_dev = combined_std_dev

        # Append the results to the dictionary
        best_dict = {}
        best_dict["best_window_start"] = best_start
        best_dict["best_window_end"] = best_end
        best_dict["best_window_valleys"] = best_window_valley
        best_dict["best_component"] = best_component
        best_dict["best_distances"] = best_distances
        best_dict["best_valleys"] = best_valleys
        best_dict["best_combined_std_dev"] = best_combined_std_dev

        return best_dict


    def find_best_valleys(self, best_dict, plot=False):

        # find the best 4 valleys
        best_four_valleys = None
        min_window_std_dev = float('inf')

        for i in range(len(best_dict["best_window_valleys"]) - 3):
            # Select the next four consecutive valleys starting from index i
            selected_valleys = best_dict["best_window_valleys"][i:i + 4]

            # Calculate the standard deviation only if exactly 4 valleys are selected
            window_std_dev = np.std([best_dict["best_distances"][v] for v in selected_valleys])

            # Check if this window has the smallest standard deviation so far
            if window_std_dev < min_window_std_dev:
                min_window_std_dev = window_std_dev
                best_four_valleys = selected_valleys

        print(f"Auto Best Frames: {best_four_valleys}")

        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(best_dict["best_distances"], label=f'Components={best_dict["best_component"]}')
            plt.plot(best_dict["best_valleys"], best_dict["best_distances"][best_dict["best_valleys"]], "o", label='Valleys')
            plt.plot(best_four_valleys, best_dict["best_distances"][best_four_valleys], "x", label='Best Four Valleys', markersize=10)
            plt.axvspan(best_dict["best_window_start"], best_dict["best_window_end"], color='y', alpha=0.3, label='Best Window')
            plt.title(f'Euclidean Distance (Components={best_dict["best_component"]})')
            plt.xlabel('Frame Number')
            plt.ylabel('Euclidean Distance')
            plt.legend()
            plt.grid(True)
            plt.show()

        return best_four_valleys


    def display_frames(self, frames, image_data, title, fontsize=14):
        fig, axes = plt.subplots(1, len(frames), figsize=(10, 8))
        for ax, frame in zip(axes, frames):
            ax.imshow(image_data[frame], cmap='gray')
            ax.set_title(f'Frame {frame}', fontsize=fontsize)
            ax.axis('off')
        plt.suptitle(title, fontsize=fontsize + 2)
        plt.show()

    def append_to_csv(self, df, best_valleys, case, img_name):

        # Create the 'Even' and 'Odd' lists
        even_vals = best_valleys[::2]
        odd_vals = best_valleys[1::2]

        # Calculate the maximum length
        max_len = max(len(even_vals), len(odd_vals))

        # Convert to float to accommodate NaN values and pad the shorter list
        even_vals = list(map(float, even_vals)) + [np.nan] * (max_len - len(even_vals))
        odd_vals = list(map(float, odd_vals)) + [np.nan] * (max_len - len(odd_vals))

        # Create a temporary dataframe with the new data
        temp_df = pd.DataFrame({
            'Even': even_vals,
            'Odd': odd_vals,
            'Case': [case] * max_len,
            'Image': [img_name] * max_len
        })

        # Append the temporary dataframe to the main dataframe
        df = pd.concat([df, temp_df], ignore_index=True)
        return df

    def convert_img_itk(self, image, itk_image):
        image = sitk.GetImageFromArray(image)
        image.SetSpacing(itk_image.GetSpacing())
        image.SetOrigin(itk_image.GetOrigin())
        return image

    def save_frames(self, frames, image_data, path, filename, itk_img):

        # Separate into ED/ES frames (even/uneven)
        frames_even = frames[::2]
        frames_odd = frames[1::2]

        images_ev, images_odd = [], []
        for frame_ev, frame_odd in zip(frames_even, frames_odd):
            image_ev = image_data[frame_ev]
            image_odd = image_data[frame_odd]
            images_ev.append(image_ev)
            images_odd.append(image_odd)

        images_ev = np.stack(images_ev, axis=0)
        images_odd = np.stack(images_odd, axis=0)
        image_ev = self.convert_img_itk(images_ev, itk_img)
        image_odd = self.convert_img_itk(images_odd, itk_img)
        sitk.WriteImage(image_ev, os.path.join(path, filename.split('.')[0]+'-ev.nii.gz'))
        sitk.WriteImage(image_odd, os.path.join(path, filename.split('.')[0]+'-odd.nii.gz'))

    def run_all(self):
        df = pd.DataFrame()
        for ind_case, case in enumerate(self.out_data_list):
            image = case[ind_case]
            print("---------------------------------------------------------------")
            print(" Case  = {}, filename = {}".format(self.case_list[ind_case], self.filename_list[ind_case]))
            print("---------------------------------------------------------------")
            distances_list = self.pca_ed(image, plot=True) # List of len n-components, length of sublists=n_frames
            best_dict = self.find_best_component(distances_list)
            # Get rid of Nones in dict
            best_dict_filtered = {}
            for key, val in best_dict.items():
                if isinstance(val, list):
                    best_dict_filtered[key] = [i for i in val if i is not None]
                else:
                    if val is not None:
                        best_dict_filtered[key] = val

            if bool(best_dict_filtered)==True:
                best_four_valleys = self.find_best_valleys(best_dict_filtered, plot=True)

                if best_four_valleys is not None and len(best_four_valleys) > 0:
                    self.display_frames(best_four_valleys, self.image_array_list[ind_case],
                                        self.filename_list[ind_case] + ' Image ' + self.case_list[ind_case].split('.')[0], fontsize=20)
                    self.save_frames(best_four_valleys, self.image_array_list[ind_case],
                                     os.path.join(self.path, self.case_list[ind_case]),
                                     self.filename_list[ind_case], self.itk_image_list[ind_case])
                    df = self.append_to_csv(df, best_four_valleys, self.case_list[ind_case].split('.')[0],
                                       self.filename_list[ind_case].split('.')[0])

        df.to_csv(os.path.join(path, 'predicted_frames.csv'))


if __name__ == '__main__':
    path = "C:\\Users\\prg20local\\OneDrive - King's College London\\Research Project\\PhD\\US_recon\\Data\\2D_echo\\current_training_data\\cropped-files-heart\\"
    folders_list = ["iFIND00226_10Mar2017"]#["iFIND00207_13Jan2017", "iFIND00209_20Jan2017", "iFIND00270_30Jun2017"]
    image_list = ["crop-heart-IM_0084-res.nii.gz"]
    cycle_detect = CycleDetect(path, folders_list, n_components_list=[2, 3, 5, 10, 15, 20], image_name=image_list)
    cycle_detect.run_all()





