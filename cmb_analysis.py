"""
This script visualizes CMB data from FITS files, performs initial data checks,
and displays foreground templates. It calculates the total variance in Q and U
maps and compares noise levels against the noise map.

Author: AI Assistant
Date: October 26, 2023
Version: 1.0
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

class CMBDataAnalyzer:
    """
    A class to analyze CMB data stored in FITS files.
    """

    def __init__(self):
        """
        Initializes the CMBDataAnalyzer.
        """
        self.confidence = 1.0  # Initial confidence level

    def load_fits_file(self, filename):
        """
        Loads a FITS file and returns the data and header.

        Args:
            filename (str): The path to the FITS file.

        Returns:
            tuple: A tuple containing the data (numpy array) and the header (astropy.io.fits.Header).
                   Returns (None, None) if an error occurs.
        """
        try:
            with fits.open(filename) as hdul:
                data = hdul[0].data
                header = hdul[0].header
            print(f"Successfully loaded FITS file: {filename}")
            return data, header
        except FileNotFoundError:
            print(f"Error: File not found: {filename}")
            self.confidence -= 0.1
            return None, None
        except Exception as e:
            print(f"Error loading FITS file {filename}: {e}")
            self.confidence -= 0.2
            return None, None

    def display_map(self, data, title="CMB Map", cmap='jet'):
        """
        Displays a 2D map using matplotlib.

        Args:
            data (numpy.ndarray): The 2D data to display.
            title (str): The title of the plot.
            cmap (str): The colormap to use.
        """
        if data is None:
            print("Error: No data to display.")
            self.confidence -= 0.05
            return

        try:
            plt.figure()
            plt.imshow(data, origin='lower', cmap=cmap)  # Ensure correct orientation
            plt.colorbar()
            plt.title(title)
            plt.xlabel("X Pixel")
            plt.ylabel("Y Pixel")
            plt.show()
        except Exception as e:
            print(f"Error displaying map: {e}")
            self.confidence -= 0.1
            
    def check_header(self, header):
        """
        Prints the header information.

        Args:
            header (astropy.io.fits.Header): The FITS header to print.
        """
        if header is None:
            print("Error: No header to display.")
            self.confidence -= 0.05
            return

        try:
            print("Header Information:")
            print(header)
        except Exception as e:
            print(f"Error displaying header: {e}")
            self.confidence -= 0.05

    def calculate_variance(self, data):
        """
        Calculates the total variance of the data.

        Args:
            data (numpy.ndarray): The data to calculate the variance from.

        Returns:
            float: The total variance of the data.  Returns None if an error occurs or data is None.
        """
        if data is None:
            print("Error: No data to calculate variance.")
            self.confidence -= 0.05
            return None

        try:
            variance = np.var(data)
            print(f"Total Variance: {variance}")
            return variance
        except Exception as e:
            print(f"Error calculating variance: {e}")
            self.confidence -= 0.1
            return None

    def compare_noise_to_noise_map(self, data_variance, noise_map_file):
        """
        Compares the calculated data variance to the values in a noise map FITS file.

        Args:
            data_variance (float): The calculated variance from the Q or U map.
            noise_map_file (str):  Path to the noise map FITS file.

        Returns:
            bool: True if the data variance is within a reasonable range of the noise map values, False otherwise.
        """

        if data_variance is None:
            print("Error: Data variance is None, cannot compare to noise map.")
            self.confidence -= 0.05
            return False

        noise_data, noise_header = self.load_fits_file(noise_map_file)

        if noise_data is None:
            print("Error: Could not load noise map file.")
            return False

        try:
            # Calculate the mean and standard deviation of the noise map
            noise_mean = np.mean(noise_data)
            noise_std = np.std(noise_data)

            # Define a reasonable range (e.g., within 3 standard deviations of the mean)
            lower_bound = noise_mean - 3 * noise_std
            upper_bound = noise_mean + 3 * noise_std

            if lower_bound <= data_variance <= upper_bound:
                print("Data variance is within the expected range based on the noise map.")
                return True
            else:
                print("Data variance is outside the expected range based on the noise map.")
                return False
        except Exception as e:
            print(f"Error comparing noise to noise map: {e}")
            self.confidence -= 0.1
            return False

    def run_analysis(self, q_file, u_file, foreground_file, noise_map_file):
        """
        Runs the complete analysis pipeline.

        Args:
            q_file (str): Path to the Q map FITS file.
            u_file (str): Path to the U map FITS file.
            foreground_file (str): Path to the foreground template FITS file.
            noise_map_file (str): Path to the noise map FITS file.
        """
        print("Starting CMB data analysis...")

        # Load and display Q map
        q_data, q_header = self.load_fits_file(q_file)
        if q_data is not None:
            self.display_map(q_data, title="Q Map")
            self.check_header(q_header)
            q_variance = self.calculate_variance(q_data)
            self.compare_noise_to_noise_map(q_variance, noise_map_file)


        # Load and display U map
        u_data, u_header = self.load_fits_file(u_file)
        if u_data is not None:
            self.display_map(u_data, title="U Map")
            self.check_header(u_header)
            u_variance = self.calculate_variance(u_data)
            self.compare_noise_to_noise_map(u_variance, noise_map_file)

        # Load and display foreground template
        foreground_data, foreground_header = self.load_fits_file(foreground_file)
        if foreground_data is not None:
            self.display_map(foreground_data, title="Foreground Template")
            self.check_header(foreground_header)

        print("CMB data analysis complete.")
        print(f"Confidence level: {self.confidence}")


if __name__ == '__main__':
    # Example usage:  Replace with actual file paths.  Create dummy files if needed for testing.
    # Note: The files MUST exist for the script to run without errors.
    q_file = 'q_map.fits'
    u_file = 'u_map.fits'
    foreground_file = 'foreground_template.fits'
    noise_map_file = 'noise_map.fits'

    # Create dummy FITS files for testing if they don't exist.
    if not os.path.exists(q_file):
        empty_data = np.zeros((100, 100))
        fits.writeto(q_file, empty_data, overwrite=True)
        print(f"Created dummy file: {q_file}")
    if not os.path.exists(u_file):
        empty_data = np.zeros((100, 100))
        fits.writeto(u_file, empty_data, overwrite=True)
        print(f"Created dummy file: {u_file}")
    if not os.path.exists(foreground_file):
        empty_data = np.zeros((100, 100))
        fits.writeto(foreground_file, empty_data, overwrite=True)
        print(f"Created dummy file: {foreground_file}")
    if not os.path.exists(noise_map_file):
        empty_data = np.random.rand(100, 100)  # Noise map needs some data for meaningful comparison
        fits.writeto(noise_map_file, empty_data, overwrite=True)
        print(f"Created dummy file: {noise_map_file}")


    analyzer = CMBDataAnalyzer()
    analyzer.run_analysis(q_file, u_file, foreground_file, noise_map_file)