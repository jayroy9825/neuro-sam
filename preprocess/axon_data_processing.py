import os
import cv2
import numpy as np
import pandas as pd

class AxonDetector:
    def __init__(self, mask_folder):
        self.mask_folder = mask_folder

    def find_axons(self, mask_image):
        # Read the mask image
        img = cv2.imread(mask_image, cv2.IMREAD_GRAYSCALE)

        # Find contours of axons
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize list to store axon information
        axon_data = []

        # Process each contour
        for idx, contour in enumerate(contours):
            # Calculate centroid and area
            M = cv2.moments(contour)
            area = M['m00']

            # Skip contour if area is too small
            if area < 10:
                continue

            # Calculate centroid
            cx = int(M['m10'] / area)
            cy = int(M['m01'] / area)

            # Convert contour to list of coordinates
            contour_coords = contour.reshape(-1, 2).tolist()

            # Append axon information to list
            axon_data.append({
                'image_name': os.path.basename(mask_image),
                'axon_id': idx,
                'axon_centroid': (cx, cy),
                'axon_coordinates': contour_coords
            })

        return axon_data

    def process_images_in_folder(self):
        # Initialize list to store axon data for all images
        all_axon_data = []

        # Iterate over all image files in the folder
        for filename in os.listdir(self.mask_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Process each image to find axons
                image_path = os.path.join(self.mask_folder, filename)
                axon_data = self.find_axons(image_path)
                all_axon_data.extend(axon_data)

        return all_axon_data

    def write_to_csv(self, axon_data, output_csv):
        # Create DataFrame from axon data
        df = pd.DataFrame(axon_data)

        # Write DataFrame to CSV
        df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    # Path to folder containing mask images
    mask_folder = "dataset/axon/"

    # Output CSV file path
    output_csv_path = "axon_data_combined.csv"

    # Initialize AxonDetector instance
    axon_detector = AxonDetector(mask_folder)

    # Process all images in the folder
    all_axon_data = axon_detector.process_images_in_folder()

    # Write axon data to a single CSV file
    axon_detector.write_to_csv(all_axon_data, output_csv_path)
