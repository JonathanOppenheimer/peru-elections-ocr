import cv2

import numpy as np
from PIL import Image

def preprocess_image_for_consistent_background(image_np):
    """
    Preprocess the image to ensure consistent black background with white lines/signatures.
    This now includes median filtering to reduce noise.
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY) if len(image_np.shape) == 3 else image_np
    
    # Apply median blur to reduce noise
    blurred = cv2.medianBlur(gray, 5)  # Median filtering to remove noise
    
    # Apply Otsu's thresholding to create a binary image
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Ensure black background with white lines
    if np.mean(thresh) > 127:  # If it's mostly white (background), invert the colors
        thresh = cv2.bitwise_not(thresh)
    
    return thresh

def generate_empty_template(image_path, save_path="./processed_empty_template.png"):
    """
    Generate a preprocessed empty template image and save it.
    The template will have a black background with white lines/signatures.
    """
    # Load the empty signature box image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Preprocess the image for consistent background
    processed_template = preprocess_image_for_consistent_background(image)

    # Save the processed template
    Image.fromarray(processed_template).save(save_path)
    print(f"Processed empty template saved at {save_path}")

# Example usage
empty_image_path = './data/empty_signature_box.png'  # Path to the empty signature box image
processed_template_path = './data/processed_empty_template.png'  # Path to save the processed template

# Generate the empty template with updated preprocessing
generate_empty_template(empty_image_path, processed_template_path)
