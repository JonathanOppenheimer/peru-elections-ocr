# Standard library imports
import json
import locale
import multiprocessing as mp
import os
import re

# Third-party imports
import cv2
import numpy as np
import pandas as pd
import pdf2image
import pytesseract
from concurrent.futures import ProcessPoolExecutor
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# Local imports
from src.ocr.time_extraction import get_ocr_results

# Set the locale to Spanish for proper date parsing
try:
    locale.setlocale(locale.LC_TIME, 'es_ES')
except locale.Error:
    print("Spanish locale 'es_ES' not installed. Using default locale.")

# Update paths to use project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
bounding_boxes_path = os.path.join(project_root, 'templates', 'r2', 'bounding_boxes.json')

# Load bounding box configurations from JSON file
with open(bounding_boxes_path, 'r') as f:   
    bounding_boxes = json.load(f)

def get_bounding_box_coordinates(bbox_name, img_width, img_height):
    """
    Retrieve bounding box coordinates and ROI for a given bounding box name.
    ROI coordinates are relative to the bounding box.

    Parameters:
        bbox_name (str): The name of the bounding box.
        img_width (int): The width of the image in pixels.
        img_height (int): The height of the image in pixels.

    Returns:
        dict: A dictionary containing:
              - "left": Left coordinate of the bounding box (int).
              - "top": Top coordinate of the bounding box (int).
              - "right": Right coordinate of the bounding box (int).
              - "bottom": Bottom coordinate of the bounding box (int).
              - "roi": Tuple (x, y, w, h) relative to the bounding box, or None if ROI is not defined.
    """
    if bbox_name not in bounding_boxes:
        raise ValueError(f"Bounding box '{bbox_name}' not found.")

    bbox = bounding_boxes[bbox_name]

    # Calculate absolute bounding box pixel coordinates
    left = int(bbox["left_pct"] * img_width)
    top = int(bbox["top_pct"] * img_height)
    right = int(bbox["right_pct"] * img_width)
    bottom = int(bbox["bottom_pct"] * img_height)

    result = {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom
    }

    # Check if ROI exists and calculate its pixel coordinates relative to the bounding box
    if "roi" in bbox:
        roi = bbox["roi"]
        bbox_width = right - left
        bbox_height = bottom - top

        # Calculate ROI coordinates relative to the bounding box
        roi_x = int(roi["x"] * bbox_width)
        roi_y = int(roi["y"] * bbox_height)
        roi_w = int(roi["w"] * bbox_width)
        roi_h = int(roi["h"] * bbox_height)

        result["roi"] = (roi_x, roi_y, roi_w, roi_h)
    else:
        result["roi"] = None

    return result

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

def preprocess_image_for_consistent_background(image_np):
    """
    Preprocess the image to obtain a consistent background for template matching.

    Parameters:
        image_np (numpy.ndarray): The image to preprocess.

    Returns:
        numpy.ndarray: The preprocessed binary image.
    """
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np

    # Apply median blur to reduce noise
    blurred = cv2.medianBlur(gray, 5)

    # Apply Otsu's thresholding to get a binary image
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Ensure that the result has a black background and white lines/signatures
    if np.mean(thresh) > 127:
        thresh = cv2.bitwise_not(thresh)

    return thresh

def template_match_signature_area(cropped_image_np, empty_template, threshold=0.75):
    """
    Compare a signature area with an empty template using Structural Similarity Index (SSIM).

    Parameters:
        cropped_image_np (numpy.ndarray): The image of the signature area to analyze.
        empty_template (numpy.ndarray): The image of the empty template.
        threshold (float): The threshold below which the area is considered signed.

    Returns:
        bool: True if the area is signed, False otherwise.
    """
    # Preprocess the signature region
    processed_image = preprocess_image_for_consistent_background(cropped_image_np)
    processed_template = preprocess_image_for_consistent_background(empty_template)
    resized_template = cv2.resize(processed_template, (processed_image.shape[1], processed_image.shape[0]))

    # Calculate SSIM
    score, _ = ssim(processed_image, resized_template, full=True)
    return score < threshold

def analyze_single_signature_box(pdf_image, empty_template_path, box_name, debug=False):
    """
    Analyze a signature box area and count the number of signed boxes.
    
    Parameters:
        pdf_image (PIL.Image.Image): The image of the PDF page.
        empty_template_path (str): Path to the empty template image.
        box_name (str): The name of the bounding box area to analyze.
        debug (bool): If True, saves debug images showing the bounding box.

    Returns:
        int: The number of signed boxes in the area.
    """
    # Load the empty template image from the path
    empty_template = cv2.imread(empty_template_path, cv2.IMREAD_GRAYSCALE)
    if empty_template is None:
        raise FileNotFoundError(f"Could not load template image from {empty_template_path}")

    img_width, img_height = pdf_image.size

    # Get bounding box coordinates
    coordinates = get_bounding_box_coordinates(box_name, img_width, img_height)
    left = coordinates["left"]
    top = coordinates["top"]
    right = coordinates["right"]
    bottom = coordinates["bottom"]

    if debug:
        # Convert PIL Image to numpy array for OpenCV
        full_image_np = np.array(pdf_image)
        debug_image = full_image_np.copy()
        
        # Draw rectangle around the bounding box
        cv2.rectangle(debug_image, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Save debug image
        os.makedirs('debug', exist_ok=True)
        cv2.imwrite(f'debug/{box_name}_full_page.png', cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR))
        
        # Save cropped region
        cropped_image = pdf_image.crop((left, top, right, bottom))
        cropped_np = np.array(cropped_image)
        cv2.imwrite(f'debug/{box_name}_cropped.png', cv2.cvtColor(cropped_np, cv2.COLOR_RGB2BGR))

    cropped_image = pdf_image.crop((left, top, right, bottom))
    cropped_image_np = np.array(cropped_image)

    # Get grid configuration from bounding boxes
    grid_config = bounding_boxes[box_name]["grid"]
    rows = grid_config["rows"]
    cols = grid_config["columns"]

    # Split the image into grids
    signature_boxes = split_image_into_grid(cropped_image_np, rows, cols)
    
    # Perform signature analysis
    signature_count = 0
    for box in signature_boxes:
        if template_match_signature_area(box, empty_template):
            signature_count += 1

    return signature_count

def extract_table_number(pdf_image, pdf_file_path):
    """
    Extract the table number from the PDF image.
    Fallbacks:
        1. If OCR fails or the number is longer than 6 digits, use the PDF file name if it's a 6-digit number.
        2. If the above fails, use '999999'.
    
    Parameters:
        pdf_image (PIL.Image.Image): The image of the PDF page.
        pdf_file_path (str): The full file path of the PDF file.
    
    Returns:
        str: The extracted table number or fallback value.
    """
    img_width, img_height = pdf_image.size

    # Get bounding box coordinates
    coordinates = get_bounding_box_coordinates('mesa_sufragio', img_width, img_height)
    left = coordinates["left"]
    top = coordinates["top"]
    right = coordinates["right"]
    bottom = coordinates["bottom"]

    # Crop the image to the bounding box
    cropped_image = pdf_image.crop((left, top, right, bottom))
    cropped_image_np = np.array(cropped_image)

    # Convert to grayscale
    gray = cv2.cvtColor(cropped_image_np, cv2.COLOR_RGB2GRAY)

    # Apply binary inverse thresholding
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Configure pytesseract to recognize digits only
    custom_config = r'--oem 3 --psm 6 digits'

    # Perform OCR
    text = pytesseract.image_to_string(thresh, config=custom_config)
    
    # Extract digits from the OCR result
    ocr_number = ''.join(filter(str.isdigit, text)).strip()

    # Check if OCR extracted a valid number (6 digits or fewer)
    if ocr_number and len(ocr_number) <= 6:
        return ocr_number

    # If OCR number is longer than 6 digits or empty, attempt to use the file name
    base_name = os.path.basename(pdf_file_path)          # e.g., '324.pdf' from '/path/to/324.pdf'
    file_name_without_ext = os.path.splitext(base_name)[0]  # e.g., '324'

    # Use regex to check if the file name is exactly six digits
    if re.fullmatch(r'\d{6}', file_name_without_ext):
        return file_name_without_ext

    # If both OCR and file name methods fail, return '999999'
    return '999999'

def extract_time_and_date(pdf_image, box_name):
    """
    Extract and parse time and date from the specified area in the PDF image.

    Parameters:
        pdf_image (PIL.Image.Image): The image of the PDF page.
        box_name (str): The name of the bounding box area to analyze.

    Returns:
        dict: OCR results containing time and date information.
    """
    img_width, img_height = pdf_image.size

    # Get bounding box coordinates
    coordinates = get_bounding_box_coordinates(box_name, img_width, img_height)
    left = coordinates["left"]
    top = coordinates["top"]
    right = coordinates["right"]
    bottom = coordinates["bottom"]

    # Access ROI coordinates if they exist
    roi_coords = coordinates.get("roi")

    cropped_image = pdf_image.crop((left, top, right, bottom))
    cropped_image_np = np.array(cropped_image)

    # Preprocess the image for better OCR accuracy
    ocr_results = get_ocr_results(cropped_image_np, roi_coords)
    return ocr_results

def get_all_features(pdf_path, empty_template_paths):
    """
    Analyze all features (signature counts, table number, open/close times) from the PDF.

    Parameters:
        pdf_path (str): The path to the PDF file.
        empty_template_paths (dict): Dictionary mapping box names to template image paths.

    Returns:
        tuple: Extracted features including table number, signature counts, open and close times.
    """
    pdf_images = pdf2image.convert_from_path(pdf_path, dpi=300)
    table_number = extract_table_number(pdf_images[0], pdf_path)

    # Analyze numobs1 (on page 1)
    numobs1_count = analyze_single_signature_box(pdf_images[0], empty_template_paths["numobs1"], "numobs1")
    print(numobs1_count)
    # Analyze numobs2 and numobs3 (on page 2)
    numobs2_count = analyze_single_signature_box(pdf_images[1], empty_template_paths["numobs2"], "numobs2")
    numobs3_count = analyze_single_signature_box(pdf_images[1], empty_template_paths["numobs3"], "numobs3")

    # Analyze open and close times
    # open_time = extract_time_and_date(pdf_images[0], "open_time")
    # close_time = extract_time_and_date(pdf_images[0], "close_time")

    return table_number, numobs1_count, numobs2_count, numobs3_count#, open_time, close_time

def process_single_pdf(args):
    """
    Process a single PDF file and return its results.
    
    Parameters:
        args (tuple): (pdf_path, empty_template_paths, debug)
    
    Returns:
        tuple: (filename, results) or (filename, None) if error
    """
    pdf_path, empty_template_paths, debug = args
    filename = os.path.basename(pdf_path)
    
    try:
        # Convert PDF to images
        pdf_images = pdf2image.convert_from_path(pdf_path, dpi=300)
        
        # Extract features
        table_number = extract_table_number(pdf_images[0], pdf_path)
        numobs1 = analyze_single_signature_box(pdf_images[0], empty_template_paths["numobs1"], "numobs1", debug)
        numobs2 = analyze_single_signature_box(pdf_images[1], empty_template_paths["numobs2"], "numobs2", debug)
        numobs3 = analyze_single_signature_box(pdf_images[1], empty_template_paths["numobs3"], "numobs3", debug)
        
        return filename, [table_number, numobs1, numobs2, numobs3]
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return filename, None

def process_folder(input_folder, empty_template_paths, csv_output_path, debug=False, first_pdf_only=False, batch_size=100):
    """
    Process PDFs in parallel with checkpointing.
    """
    # Create output directory if needed
    os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
    
    # Get list of PDF files
    pdf_files = [f for f in os.listdir(input_folder) if f.endswith('.pdf')]
    if first_pdf_only and pdf_files:
        pdf_files = [pdf_files[0]]
    
    # Load progress file if it exists
    progress_file = csv_output_path + '.progress'
    processed_files = set()
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            processed_files = set(f.read().splitlines())
    
    # Filter out already processed files
    remaining_files = [f for f in pdf_files if f not in processed_files]
    
    # Prepare arguments for parallel processing
    process_args = [
        (os.path.join(input_folder, filename), empty_template_paths, debug)
        for filename in remaining_files
    ]
    
    columns = ["acta_number", "numobs1", "numobs2", "numobs3"]
    
    # Initialize CSV file if it doesn't exist
    if not os.path.exists(csv_output_path):
        pd.DataFrame(columns=columns).to_csv(csv_output_path, index=False)
    
    # Process in batches
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        for i in range(0, len(process_args), batch_size):
            batch_args = process_args[i:i + batch_size]
            batch_results = []
            
            # Process batch in parallel with progress bar
            with tqdm(total=len(batch_args), desc=f"Processing batch {i//batch_size + 1}") as pbar:
                for filename, results in executor.map(process_single_pdf, batch_args):
                    if results is not None:
                        batch_results.append(results)
                    
                    # Update progress file
                    with open(progress_file, 'a') as f:
                        f.write(f"{filename}\n")
                    
                    pbar.update(1)
            
            # Write batch results to CSV
            if batch_results:
                df_batch = pd.DataFrame(batch_results, columns=columns)
                df_batch.to_csv(csv_output_path, mode='a', header=False, index=False)
    
    # Final sorting of the CSV file
    if os.path.exists(csv_output_path) and os.path.getsize(csv_output_path) > 0:
        df = pd.read_csv(csv_output_path)
        df.sort_values(by="acta_number", inplace=True)
        df.to_csv(csv_output_path, index=False)
    
    # Clean up progress file
    if os.path.exists(progress_file):
        os.remove(progress_file)
    
    print(f"All PDFs processed. Results saved to {csv_output_path}")

if __name__ == "__main__":
    # Update paths to use project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_directory = os.path.join(project_root, 'data')
    
    empty_template_paths_r1 = {
        "numobs1": os.path.join(project_root, 'templates/r1/empty_numobs1.png'),
        "numobs2": os.path.join(project_root, 'templates/r1/empty_numobs2.png'),
        "numobs3": os.path.join(project_root, 'templates/r1/empty_numobs3.png')
    }
    
    empty_template_paths_r2 = {
        "numobs1": os.path.join(project_root, 'templates/r2/empty_numobs1.png'),
        "numobs2": os.path.join(project_root, 'templates/r2/empty_numobs2.png'),
        "numobs3": os.path.join(project_root, 'templates/r2/empty_numobs3.png')
    }

    # Update output path
    output_dir = os.path.join(project_root, 'data', 'output', 'csv')
    os.makedirs(output_dir, exist_ok=True)

    # List all subdirectories in the data directory
    input_folders = [os.path.join(data_directory, name) for name in os.listdir(data_directory)
                     if os.path.isdir(os.path.join(data_directory, name))]

    # Process each input folder
    for input_folder in input_folders:
        input_folder_name = os.path.basename(input_folder)
        csv_output_path = os.path.join(output_dir, f'{input_folder_name}.csv')
        print(f"Processing input folder: {input_folder}")
        process_folder(input_folder, empty_template_paths_r2, csv_output_path, debug=True, first_pdf_only=True)
