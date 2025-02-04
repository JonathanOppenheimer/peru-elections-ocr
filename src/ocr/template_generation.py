import cv2
import json
import numpy as np
from PIL import Image
import pdf2image
import os

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

def split_image_into_grid(image_np, rows, cols):
    """
    Splits an image into a grid of specified rows and columns.

    Parameters:
        image_np (numpy.ndarray): The image to split.
        rows (int): Number of rows.
        cols (int): Number of columns.

    Returns:
        list: A list of numpy.ndarray images, each representing a cell in the grid.
    """
    img_height, img_width = image_np.shape[:2]
    cell_height = img_height // rows
    cell_width = img_width // cols
    boxes = []
    for i in range(rows):
        for j in range(cols):
            top = i * cell_height
            bottom = (i + 1) * cell_height if i != rows - 1 else img_height
            left = j * cell_width
            right = (j + 1) * cell_width if j != cols - 1 else img_width
            box = image_np[top:bottom, left:right]
            boxes.append(box)
    return boxes

def extract_box_from_pdf(pdf_path, bbox_name, bounding_boxes, page_num=0):
    """
    Extract a specific box from a PDF using bounding box coordinates.
    
    Parameters:
        pdf_path (str): Path to the PDF file
        bbox_name (str): Name of the bounding box to extract
        bounding_boxes (dict): Dictionary of bounding box configurations
        page_num (int): Page number to extract from (0-based)
    
    Returns:
        numpy.ndarray: Cropped image of the specified box
    """
    if bbox_name not in bounding_boxes:
        raise ValueError(f"Bounding box '{bbox_name}' not found")
    
    # Convert PDF page to image
    images = pdf2image.convert_from_path(pdf_path, dpi=300)
    if page_num >= len(images):
        raise ValueError(f"Page {page_num} not found in PDF")
    
    page_image = images[page_num]
    img_width, img_height = page_image.size
    
    # Get bounding box coordinates
    bbox = bounding_boxes[bbox_name]
    left = int(bbox["left_pct"] * img_width)
    top = int(bbox["top_pct"] * img_height)
    right = int(bbox["right_pct"] * img_width)
    bottom = int(bbox["bottom_pct"] * img_height)
    
    # Crop the image
    cropped = page_image.crop((left, top, right, bottom))
    return np.array(cropped)

def generate_empty_templates(empty_pdf_path, bounding_boxes, output_dir="./templates/r2"):
    """
    Generate empty templates for all signature boxes defined in the bounding_boxes.json
    
    Parameters:
        empty_pdf_path (str): Path to a PDF with empty signature boxes
        bounding_boxes (dict): Dictionary of bounding box configurations
        output_dir (str): Directory to save the generated templates
    """
    # Dictionary mapping box names to their page numbers
    box_pages = {
        "numobs1": 0,  # First page
        "numobs2": 1,  # Second page
        "numobs3": 1   # Second page
    }
    
    for box_name, page_num in box_pages.items():
        try:
            # Extract the full signature area
            box_image = extract_box_from_pdf(empty_pdf_path, box_name, bounding_boxes, page_num)
            
            # Get grid configuration
            grid_config = bounding_boxes[box_name]["grid"]
            rows = grid_config["rows"]
            cols = grid_config["columns"]
            
            # Split into individual signature boxes
            signature_boxes = split_image_into_grid(box_image, rows, cols)
            
            # Process and save just the first box as template
            processed_template = preprocess_image_for_consistent_background(signature_boxes[0])
            output_path = f"{output_dir}/empty_{box_name}.png"
            Image.fromarray(processed_template).save(output_path)
            print(f"Generated template for {box_name} at {output_path}")
            
        except Exception as e:
            print(f"Error generating template for {box_name}: {str(e)}")

if __name__ == "__main__":
    # Update configuration paths to use project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    # Configuration paths
    empty_pdf_path = os.path.join(project_root, "data", "testing", "000001.pdf")
    bounding_boxes_path = os.path.join(project_root, "templates", "r2", "bounding_boxes.json")
    output_dir = os.path.join(project_root, "templates", "r2")
    
    # Load bounding box configurations
    with open(bounding_boxes_path, 'r') as f:
        bounding_boxes = json.load(f)
    
    # Generate all templates
    generate_empty_templates(empty_pdf_path, bounding_boxes, output_dir)
