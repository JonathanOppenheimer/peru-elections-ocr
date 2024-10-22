import csv
import cv2
import locale
import numpy as np
import os
import pdf2image
import pytesseract

from skimage.metrics import structural_similarity as ssim
from time_extraction import get_ocr_results

# Set the locale to Spanish for proper date parsing
try:
    locale.setlocale(locale.LC_TIME, 'es_ES')
except locale.Error:
    print("Spanish locale 'es_ES' not installed. Using default locale.")

# Define the bounding boxes for different signature areas
bounding_boxes = {
    "mesa_sufragio": {
        "left_pct": 0.05,
        "top_pct": 0.12,
        "right_pct": 0.19,
        "bottom_pct": 0.165
    },
    "numobs1": {
        "left_pct": 0.84,
        "top_pct": 0.205,
        "right_pct": 0.97,
        "bottom_pct": 0.98
    },
    "numobs2": {
        "left_pct": 0.717,
        "top_pct": 0.045,
        "right_pct": 0.965,
        "bottom_pct": 0.495
    },
    "numobs3": {
        "left_pct": 0.717,
        "top_pct": 0.505,
        "right_pct": 0.965,
        "bottom_pct": 0.955
    },
    "open_time": {
        "left_pct": 0.05,
        "top_pct": 0.225,
        "right_pct": 0.8,
        "bottom_pct": 0.255,
        "roi": {
            "x": 0.124,
            "y": 0.13,
            "w": 0.212,
            "h": 0.82
        }
    },
    "close_time": {
        "left_pct": 0.05,
        "top_pct": 0.86,
        "right_pct": 0.75,
        "bottom_pct": 0.89,
        "roi": {
            "x": 0.124,
            "y": 0.13,
            "w": 0.212,
            "h": 0.82
        }
    }
}

signature_box_splits = {
    "numobs1": (10, 1),  # 10 rows, 1 column
    "numobs2": (5, 2),   # 5 rows, 2 columns
    "numobs3": (5, 2)
}

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

def analyze_single_signature_box(pdf_image, empty_template_path, box_name):
    """
    Analyze a signature box area and count the number of signed boxes.

    Parameters:
        pdf_image (PIL.Image.Image): The image of the PDF page.
        empty_template_path (str): Path to the empty template image.
        box_name (str): The name of the bounding box area to analyze.

    Returns:
        int: The number of signed boxes in the area.
    """
    # Load the empty template image from the path
    empty_template = cv2.imread(empty_template_path, cv2.IMREAD_GRAYSCALE)

    img_width, img_height = pdf_image.size

    # Get bounding box coordinates
    coordinates = get_bounding_box_coordinates(box_name, img_width, img_height)
    left = coordinates["left"]
    top = coordinates["top"]
    right = coordinates["right"]
    bottom = coordinates["bottom"]

    cropped_image = pdf_image.crop((left, top, right, bottom))
    cropped_image_np = np.array(cropped_image)

    # Determine the number of rows and columns for splitting
    rows, cols = signature_box_splits.get(box_name, (10, 1))

    # Split the image into grids
    signature_boxes = split_image_into_grid(cropped_image_np, rows, cols)

    # Perform signature analysis
    signature_count = 0
    for box in signature_boxes:
        if template_match_signature_area(box, empty_template):
            signature_count += 1

    return signature_count

def extract_table_number(pdf_image, pdf_file_name):
    """
    Extract the table number from the PDF image.
    If OCR fails, use the PDF file name (without extension).

    Parameters:
        pdf_image (PIL.Image.Image): The image of the PDF page.
        pdf_file_name (str): The name or path of the PDF file.

    Returns:
        str: The extracted table number.
    """
    img_width, img_height = pdf_image.size

    # Get bounding box coordinates
    coordinates = get_bounding_box_coordinates('mesa_sufragio', img_width, img_height)
    left = coordinates["left"]
    top = coordinates["top"]
    right = coordinates["right"]
    bottom = coordinates["bottom"]

    cropped_image = pdf_image.crop((left, top, right, bottom))
    cropped_image_np = np.array(cropped_image)
    gray = cv2.cvtColor(cropped_image_np, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    custom_config = r'--oem 3 --psm 6 digits'
    text = pytesseract.image_to_string(thresh, config=custom_config)
    number = ''.join(filter(str.isdigit, text))

    if not number:
        # If OCR failed to extract a number, use the PDF file name without extension
        number = os.path.splitext(os.path.basename(pdf_file_name))[0]

    return number

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

    # Analyze numobs2 and numobs3 (on page 2)
    numobs2_count = analyze_single_signature_box(pdf_images[1], empty_template_paths["numobs2"], "numobs2")
    numobs3_count = analyze_single_signature_box(pdf_images[1], empty_template_paths["numobs3"], "numobs3")

    # Analyze open and close times
    # open_time = extract_time_and_date(pdf_images[0], "open_time")
    # close_time = extract_time_and_date(pdf_images[0], "close_time")

    return table_number, numobs1_count, numobs2_count, numobs3_count#, open_time, close_time

def append_result_to_csv(row, csv_path):
    """
    Save the result to CSV file row by row.

    Parameters:
        row (list): The data row to append.
        csv_path (str): The path to the CSV file.
    """
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["acta_number", "numobs1", "numobs2", "numobs3"]) #, "open_time", "close_time"])
        writer.writerow(row)

def process_folder(input_folder, empty_template_paths, csv_output_path):
    """
    Process all PDFs in the input folder and save the results to a CSV file.

    Parameters:
        input_folder (str): Path to the folder containing PDF files.
        empty_template_paths (dict): Dictionary mapping box names to template image paths.
        csv_output_path (str): Path to the output CSV file.
    """
    # Iterate over all PDF files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(input_folder, filename)
            print(f"Processing {filename}...")
            try:
                # Analyze the signature boxes
                table_number, numobs1, numobs2, numobs3 = get_all_features(pdf_path, empty_template_paths)

                # Save the results to CSV
                append_result_to_csv([table_number, numobs1, numobs2, numobs3], csv_output_path)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print(f"All PDFs processed. Results saved to {csv_output_path}")

if __name__ == "__main__":
    input_folder = '/Users/oppenheimerj/Desktop/Senior Year/Semester 1/POL429/Acta Images/'  # Folder containing multiple PDFs
    empty_template_paths = {
        "numobs1": './templates/empty_numobs1.png',
        "numobs2": './templates/empty_numobs2.png',
        "numobs3": './templates/empty_numobs3.png'
    }
    csv_output_path = './signature_counts.csv'

    # Process all PDFs in the input folder and write results to the CSV
    process_folder(input_folder, empty_template_paths, csv_output_path)
