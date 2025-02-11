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

def extract_box_from_pdf(pdf_path, bbox_name, bounding_boxes, page_num=0, document_type="default"):
    """
    Extract a specific box from a PDF using bounding box coordinates.
    
    Parameters:
        pdf_path (str): Path to the PDF file
        bbox_name (str): Name of the bounding box to extract
        bounding_boxes (dict): Dictionary of bounding box configurations
        page_num (int): Page number to extract from (0-based)
        document_type (str): Type of document being processed. Defaults to "default".
    
    Returns:
        numpy.ndarray: Cropped image of the specified box
    """
    # Check if document type exists in bounding boxes
    if document_type not in bounding_boxes:
        document_type = "default"
    
    # Check if the specific box exists in the document type
    if bbox_name in bounding_boxes[document_type]:
        bbox = bounding_boxes[document_type][bbox_name]
    else:
        # Fallback to default if the box is not defined for this document type
        if bbox_name not in bounding_boxes["default"]:
            raise ValueError(f"Bounding box '{bbox_name}' not found in default configuration.")
        bbox = bounding_boxes["default"][bbox_name]
    
    # Convert PDF page to image
    images = pdf2image.convert_from_path(pdf_path, dpi=300)
    if page_num >= len(images):
        raise ValueError(f"Page {page_num} not found in PDF")
    
    page_image = images[page_num]
    img_width, img_height = page_image.size
    
    # Get bounding box coordinates
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
        "numobs3": 1,  # Second page
        "document_type": 0  # First page
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # First, detect the document type of the empty PDF
    images = pdf2image.convert_from_path(empty_pdf_path, dpi=300)
    document_type = "default"  # Start with default
    
    try:
        # Extract document type box using default configuration
        doc_type_box = extract_box_from_pdf(empty_pdf_path, "document_type", bounding_boxes, 0, "default")
        # Convert to grayscale if needed
        if len(doc_type_box.shape) == 3:
            doc_type_box = cv2.cvtColor(doc_type_box, cv2.COLOR_RGB2GRAY)
        # Apply thresholding
        _, thresh = cv2.threshold(doc_type_box, 150, 255, cv2.THRESH_BINARY)
        # Perform OCR
        import pytesseract
        text = pytesseract.image_to_string(thresh, config='--oem 3 --psm 6', lang='spa')
        detected_type = ' '.join(text.strip().split())
        if detected_type in bounding_boxes:
            document_type = detected_type
            print(f"Detected document type: {document_type}")
    except Exception as e:
        print(f"Warning: Could not detect document type, using default. Error: {str(e)}")
    
    for box_name, page_num in box_pages.items():
        try:
            # Extract the full signature area using detected document type
            box_image = extract_box_from_pdf(empty_pdf_path, box_name, bounding_boxes, page_num, document_type)
            
            # Save the full signature area
            full_area_processed = preprocess_image_for_consistent_background(box_image)
            full_area_output_path = f"{output_dir}/full_area_{box_name}.png"
            Image.fromarray(full_area_processed).save(full_area_output_path)
            print(f"Generated full area template for {box_name} at {full_area_output_path}")
            
            # Only process grid for signature boxes (not document_type)
            if box_name != "document_type":
                # Get grid configuration from the same source as the bounding box
                if document_type in bounding_boxes and box_name in bounding_boxes[document_type]:
                    grid_config = bounding_boxes[document_type][box_name]["grid"]
                else:
                    grid_config = bounding_boxes["default"][box_name]["grid"]
                
                rows = grid_config["rows"]
                cols = grid_config["columns"]
                
                # Split into individual signature boxes
                signature_boxes = split_image_into_grid(box_image, rows, cols)
                
                # Process and save just the first box as template
                processed_template = preprocess_image_for_consistent_background(signature_boxes[0])
                output_path = f"{output_dir}/empty_{box_name}.png"
                Image.fromarray(processed_template).save(output_path)
                print(f"Generated template for {box_name} at {output_path}")
            else:
                # For document_type, save the full area as is
                output_path = f"{output_dir}/empty_{box_name}.png"
                Image.fromarray(full_area_processed).save(output_path)
                print(f"Generated template for {box_name} at {output_path}")
            
        except Exception as e:
            print(f"Error generating template for {box_name}: {str(e)}")

if __name__ == "__main__":
    # Update configuration paths to use project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    # Configuration paths
    bounding_boxes_path = os.path.join(project_root, "templates", "erm", "bounding_boxes.json")
    
    # Load bounding box configurations
    with open(bounding_boxes_path, 'r') as f:
        bounding_boxes = json.load(f)
    
    # Define empty PDFs for each type
    type_pdfs = {
        "REGIONAL": os.path.join(project_root, "data", "input", "testing", "ACERMC3709207405405.PDF"),  # A REGIONAL form
        "MUNICIPAL PROVINCIAL": os.path.join(project_root, "data", "input", "testing", "ACERMC3709307844823.PDF")  # A MUNICIPAL PROVINCIAL form
    }
    
    # Generate templates for each type
    for doc_type, pdf_path in type_pdfs.items():
        output_dir = os.path.join(project_root, "templates", doc_type.lower().replace(" ", "_"))
        print(f"\nGenerating templates for {doc_type} using {pdf_path}")
        print(f"Output directory: {output_dir}")
        generate_empty_templates(pdf_path, bounding_boxes, output_dir)
