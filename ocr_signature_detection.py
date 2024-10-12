import csv
import cv2
import numpy as np
import os
import pdf2image
import pytesseract

from PIL import Image
from skimage.metrics import structural_similarity as ssim

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
    }
}

# Function to extract the table number from page 1
def extract_table_number(pdf_image):
    img_width, img_height = pdf_image.size
    left, top, right, bottom = get_bounding_box_coordinates("mesa_sufragio", img_width, img_height)
    cropped_image = pdf_image.crop((left, top, right, bottom))
    cropped_image_np = np.array(cropped_image)
    gray = cv2.cvtColor(cropped_image_np, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    custom_config = r'--oem 3 --psm 6 digits'
    text = pytesseract.image_to_string(thresh, config=custom_config)
    number = ''.join(filter(str.isdigit, text))
    print(f"Extracted Table Number: {number}")
    return number

def get_bounding_box_coordinates(bbox_name, img_width, img_height):
    bbox = bounding_boxes[bbox_name]
    left = int(bbox["left_pct"] * img_width)
    top = int(bbox["top_pct"] * img_height)
    right = int(bbox["right_pct"] * img_width)
    bottom = int(bbox["bottom_pct"] * img_height)
    return left, top, right, bottom

def preprocess_image_for_consistent_background(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY) if len(image_np.shape) == 3 else image_np

    # Apply median blur to reduce noise
    blurred = cv2.medianBlur(gray, 5)  # Median filtering to reduce salt-and-pepper noise
    
    # Apply Otsu's thresholding to get a binary image
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Ensure that the result has a black background and white lines/signatures
    if np.mean(thresh) > 127:  # Invert to ensure black background with white lines
        thresh = cv2.bitwise_not(thresh)
    return thresh

def template_match_signature_area(cropped_image_np, empty_template, threshold=0.75):
    # Preprocess the signature region
    processed_image = preprocess_image_for_consistent_background(cropped_image_np)
    processed_template = preprocess_image_for_consistent_background(empty_template)
    resized_template = cv2.resize(processed_template, (processed_image.shape[1], processed_image.shape[0]))

    # Calculate SSIM
    score, _ = ssim(processed_image, resized_template, full=True)
    print(f"SSIM Score: {score}")
    return score < threshold

# Function to analyze all signature boxes across pages
def analyze_signature_boxes(pdf_path, empty_template_paths):
    pdf_images = pdf2image.convert_from_path(pdf_path, dpi=300)
    table_number = extract_table_number(pdf_images[0])

    # Analyze numobs1 (on page 1)
    numobs1_count = analyze_single_signature_box(pdf_images[0], empty_template_paths["numobs1"], "numobs1")
    
    # Analyze numobs2 and numobs3 (on page 2)
    numobs2_count = analyze_single_signature_box(pdf_images[1], empty_template_paths["numobs2"], "numobs2")
    numobs3_count = analyze_single_signature_box(pdf_images[1], empty_template_paths["numobs3"], "numobs3")
    
    return table_number, numobs1_count, numobs2_count, numobs3_count

# Function to analyze single signature box and save the cropped region for debugging
def analyze_single_signature_box(pdf_image, empty_template_path, box_name):
    # Load the empty template image from the path
    empty_template = cv2.imread(empty_template_path, cv2.IMREAD_GRAYSCALE)

    img_width, img_height = pdf_image.size
    left, top, right, bottom = get_bounding_box_coordinates(box_name, img_width, img_height)
    cropped_image = pdf_image.crop((left, top, right, bottom))
    cropped_image_np = np.array(cropped_image)

    # Now split into signature boxes depending on the box (2x5 for numobs2 and numobs3)
    if box_name in ["numobs2", "numobs3"]:
        signature_boxes = split_signature_area_2x5(cropped_image_np)
    else:
        signature_boxes = split_signature_area_vertically(cropped_image_np)

    # Perform signature analysis
    signature_count = 0
    for i, box in enumerate(signature_boxes):
        if template_match_signature_area(box, empty_template):
            signature_count += 1
            print(f"Signature detected in {box_name}, box {i + 1}")
        else:
            print(f"No signature in {box_name}, box {i + 1}")
    
    return signature_count

def split_signature_area_vertically(cropped_image_np, num_boxes=10):
    img_height, img_width = cropped_image_np.shape[:2]
    box_height = img_height // num_boxes
    boxes = []
    for i in range(num_boxes):
        top = i * box_height
        bottom = (i + 1) * box_height if i != num_boxes - 1 else img_height
        boxes.append(cropped_image_np[top:bottom, :])
    return boxes

def split_signature_area_2x5(cropped_image_np):
    """
    Split the signature area into a 2x5 grid (two columns and five rows).
    """
    img_height, img_width = cropped_image_np.shape[:2]
    row_height = img_height // 5  # 5 rows
    col_width = img_width // 2    # 2 columns

    boxes = []
    for i in range(5):  # For each row
        for j in range(2):  # For each column
            top = i * row_height
            bottom = (i + 1) * row_height if i != 4 else img_height  # Handle the last row
            left = j * col_width
            right = (j + 1) * col_width if j != 1 else img_width  # Handle the last column
            box = cropped_image_np[top:bottom, left:right]
            boxes.append(box)
    return boxes

# Function to save results to CSV row by row
def append_result_to_csv(row, csv_path):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["acta_number", "numobs1", "numobs2", "numobs3"])  # Write header if file doesn't exist
        writer.writerow(row)

# Function to process all PDFs in the input folder
def process_folder(input_folder, empty_template_paths, csv_output_path):
    # Iterate over all PDF files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(input_folder, filename)
            print(f"Processing {filename}...")
            
            # Analyze the signature boxes
            table_number, numobs1, numobs2, numobs3 = analyze_signature_boxes(pdf_path, empty_template_paths)
            
            # Save the results to CSV
            append_result_to_csv([table_number, numobs1, numobs2, numobs3], csv_output_path)
    
    print(f"All PDFs processed. Results saved to {csv_output_path}")


############################## PROCESSING ##############################

input_folder = './data/'  # Folder containing multiple PDFs
empty_template_paths = {
    "numobs1": './templates/empty_numobs1.png',
    "numobs2": './templates/empty_numobs2.png',
    "numobs3": './templates/empty_numobs3.png'
}
csv_output_path = './signature_counts.csv'

# Process all PDFs in the input folder and write results to the CSV
process_folder(input_folder, empty_template_paths, csv_output_path)
