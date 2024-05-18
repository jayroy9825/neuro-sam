import os
import cv2
import numpy as np
import pandas as pd

class MyelinDetector:
    def __init__(self, mask_folder):
        self.mask_folder = mask_folder

    def find_myelin(self, myelin_mask):
        try:
            # Read the myelin mask image
            img = cv2.imread(myelin_mask, cv2.IMREAD_GRAYSCALE)

            # Invert the image so that the myelin regions are white
            inverted_img = cv2.bitwise_not(img)

            # Find contours of myelin
            contours, _ = cv2.findContours(inverted_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Initialize list to store myelin information
            myelin_data = []

            # Process each contour
            for idx, contour in enumerate(contours):
                # Calculate moments
                M = cv2.moments(contour)

                # Skip contour if area is too small
                if M['m00'] == 0:
                    continue

                # Calculate centroid
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                # Convert contour to list of coordinates
                contour_coords = contour.reshape(-1, 2).tolist()

                # Append myelin information to list
                myelin_data.append({
                    'image_name': os.path.basename(myelin_mask),
                    'myelin_id': idx,
                    'myelin_centroid': (cx, cy),
                    'myelin_coordinates': contour_coords
                })

            return myelin_data

        except Exception as e:
            print(f"Error processing {myelin_mask}: {e}")
            return []

    def process_images_in_folder(self):
        # Initialize list to store myelin data for all images
        all_myelin_data = []

        # Iterate over all image files in the folder
        for filename in os.listdir(self.mask_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Process each image to find myelin
                image_path = os.path.join(self.mask_folder, filename)
                myelin_data = self.find_myelin(image_path)
                all_myelin_data.extend(myelin_data)

        return all_myelin_data

    def write_to_csv(self, myelin_data, output_csv):
        try:
            # Create DataFrame from myelin data
            df = pd.DataFrame(myelin_data)

            # Write DataFrame to CSV
            df.to_csv(output_csv, index=False)
            print(f"Myelin data saved to {output_csv}")

        except Exception as e:
            print(f"Error writing to {output_csv}: {e}")

if __name__ == "__main__":
    # Path to folder containing myelin mask images
    folder_path = "/Neuroinformatics/dataset/myelin/"

    # Output CSV file path
    output_csv_path = "myelin_data_combined.csv"

    # Initialize MyelinDetector instance
    myelin_detector = MyelinDetector(folder_path)

    # Process all images in the folder
    all_myelin_data = myelin_detector.process_images_in_folder()

    # Write myelin data to a single CSV file
    myelin_detector.write_to_csv(all_myelin_data, output_csv_path)
