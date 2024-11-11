import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

class HRTFContainer:
    def __init__(self):
        self.hrir_r = None  # Right ear HRTF
        self.hrir_l = None  # Left ear HRTF
        self.azimuths = []  # List to store azimuth angles (if available)

    def load_hrtf(self, file_path):
        """Load HRTF data from a .mat file."""
        data = loadmat(file_path)
        print("Keys in the loaded .mat file:", data.keys())  # Inspect keys
        self.hrir_r = data['hrir_r']  # Right ear HRIR
        self.hrir_l = data['hrir_l']  # Left ear HRIR

        # If you have azimuth data, you can load it as well
        # self.azimuths = data['azimuths'].flatten()  # Adjust if you have azimuth data

    def check_hrtf(self, azimuth_index):
        """Check HRTF values for specific azimuth index."""
        if self.hrir_r is not None and self.hrir_l is not None:
            hrir_r = self.hrir_r[azimuth_index]  # Right ear HRIR
            hrir_l = self.hrir_l[azimuth_index]  # Left ear HRIR
            print(f'HRTF values for Azimuth index {azimuth_index}:')
            print('Right Ear HRIR:', hrir_r)
            print('Left Ear HRIR:', hrir_l)
            return hrir_r, hrir_l
        else:
            print('HRTF data not loaded.')
            return None, None

    def visualize_hrtf(self, azimuth_index):
        """Visualize HRTF values for a specific azimuth index."""
        hrir_r, hrir_l = self.check_hrtf(azimuth_index)
        if hrir_r is not None and hrir_l is not None:
            time = np.arange(len(hrir_r))  # Time axis for the HRIR
            plt.figure(figsize=(12, 6))
            plt.plot(time, hrir_r, label='Right Ear HRIR', color='blue')
            plt.plot(time, hrir_l, label='Left Ear HRIR', color='red')
            plt.title(f'HRTF Visualization for Azimuth Index: {azimuth_index}')
            plt.xlabel('Sample Index')
            plt.ylabel('Amplitude')
            plt.grid()
            plt.legend()
            plt.show()

# Example usage
hrtf_container = HRTFContainer()
hrtf_container.load_hrtf(r'classify_speech\cipic-hrtf-database-master\standard_hrir_database\subject_021\hrir_final.mat')  # Load your HRTF .mat file

# Visualize HRTF values for a specific azimuth index (adjust index based on your data)
hrtf_container.visualize_hrtf(0)  # Visualize for the first azimuth index