import os
import numpy as np
from PIL import Image

def separate_white_and_gray(image):
    '''
    Function to separate white and gray shades from an image.
    '''
    # Convert the image to grayscale
    gray_image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])

    # Threshold the image to separate white and gray shades
    white_shades = (gray_image > 240).astype(np.uint8) * 255
    
    # Extract gray shades by subtracting white shades from the grayscale image
    gray_shades = gray_image - white_shades
    
    # Apply thresholding to further refine the gray shades
    gray_shades = (gray_shades > 20).astype(np.uint8) * 255

    return white_shades, gray_shades

def process_images(input_folder, output_folder_white, output_folder_gray):
    '''
    Function to process images in a folder and save separated white and gray shades.
    '''
    for filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, filename)
        # Load image using PIL
        image = np.array(Image.open(image_path))
        
        # Separate white and gray shades
        white_shades, gray_shades = separate_white_and_gray(image)
        
        # Save the white shades
        w_output_path = os.path.join(output_folder_white, filename)
        Image.fromarray(white_shades).save(w_output_path)
        
        # Save the gray shades
        g_output_path = os.path.join(output_folder_gray, filename)
        Image.fromarray(gray_shades).save(g_output_path)
        
        print(f"Image {filename} processed and saved.")

if __name__ == "__main__":
    input_folder = "Neuroinformatics/dataset/masks/"
    output_folder_white = "Neuroinformatics/dataset/axon/"
    output_folder_gray = "Neuroinformatics/dataset/myelin/"
    
    # Ensure output folders exist, if not, create them
    os.makedirs(output_folder_white, exist_ok=True)
    os.makedirs(output_folder_gray, exist_ok=True)
    
    # Process images
    process_images(input_folder, output_folder_white, output_folder_gray)
