import os
from PIL import Image

def replace_colors(image_path, replacement_colors):
    # Open the image
    image = Image.open(image_path)

    # Convert the image to RGB mode (in case it's in a different mode)
    image = image.convert('RGB')

    # Get the width and height of the image
    width, height = image.size

    # Iterate through each pixel in the image
    for y in range(height):
        for x in range(width):
            # Get the RGB values of the pixel
            r, g, b = image.getpixel((x, y))
            hex_code = '#{0:02x}{1:02x}{2:02x}'.format(r, g, b)
            if hex_code=="#ff0000":
            # Replace the original color with one of the replacement colors
                new_color = replacement_colors[0]
            elif hex_code=="#ffff00":
            # Replace the original color with one of the replacement colors
                new_color = replacement_colors[1]
            elif hex_code=="#008080":
            # Replace the original color with one of the replacement colors
                new_color = replacement_colors[2]
            else:
                print("")

            # Set the new color for the pixel
            image.putpixel((x, y), new_color)

    return image

# Define the three replacement colors (in RGB format)
replacement_colors = [
    (0, 0, 0),         # Black
    (255, 255, 255),   # white
    (127, 127, 127)    # Blue
]


for filename in os.listdir("/Users/jayroy/Study/NYU/Sem-2/dataset-6/"):
        # Check if the file is an image
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # Get the full path of the image
            image_path = os.path.join("/Users/jayroy/Study/NYU/Sem-2/dataset-6/", filename)
            
            # Process the image
            modified_image = replace_colors(image_path, replacement_colors)
            modified_image.save('/Users/jayroy/Study/NYU/Sem-2/dataset-6/'+filename)
