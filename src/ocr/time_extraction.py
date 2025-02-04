import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
import re


def preprocess_roi_for_time(image_rgb, roi_coords):
    """
    Preprocess the Region of Interest (ROI) in the image for OCR.

    Parameters:
        image_rgb (numpy.ndarray): The input RGB image.
        roi_coords (tuple): A tuple (roi_x, roi_y, roi_w, roi_h) specifying the ROI.

    Returns:
        numpy.ndarray: The preprocessed ROI image ready for OCR.
    """
    roi_x, roi_y, roi_w, roi_h = roi_coords

    # Isolate the ROI from the image
    roi = image_rgb[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    # Convert ROI to grayscale
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced_roi = clahe.apply(gray_roi)

    # Apply Gaussian blur to reduce noise
    blurred_roi = cv2.GaussianBlur(contrast_enhanced_roi, (3, 3), 0)

    # Apply Otsu's thresholding to binarize the image
    _, thresh_roi = cv2.threshold(
        blurred_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Perform morphological operations to enhance text regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_roi = cv2.morphologyEx(thresh_roi, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Further enhance the ROI
    # Increase contrast using Histogram Equalization
    enhanced_roi = cv2.equalizeHist(morph_roi)

    # Apply sharpening to make characters more defined
    kernel_sharpen = np.array(
        [[-1, -1, -1],
         [-1,  9, -1],
         [-1, -1, -1]]
    )
    sharpened_roi = cv2.filter2D(enhanced_roi, -1, kernel_sharpen)

    # Apply dilation to make characters thicker
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated_roi = cv2.dilate(sharpened_roi, kernel_dilate, iterations=1)

    # Display the preprocessed ROI
    plt.figure(figsize=(6, 6))
    plt.imshow(morph_roi, cmap='gray')
    plt.axis('off')
    plt.title("Postprocess ROI - Time")
    plt.show()

    return dilated_roi


def ocr_time(enhanced_image, image_rgb):
    """
    Perform OCR on the enhanced image to extract times, and on the full image to extract Spanish text.

    Parameters:
        enhanced_image (numpy.ndarray): The preprocessed image for OCR.
        image_rgb (numpy.ndarray): The original RGB image.

    Returns:
        dict: A dictionary containing 'spanish_text' and 'times' extracted from the image.
    """
    # Initialize EasyOCR Reader for handwritten times (assuming English digits)
    easyocr_reader = easyocr.Reader(['en'])

    # Configure Tesseract to use Spanish language
    tesseract_config = '--oem 3 --psm 6 -l spa'

    # Perform OCR on the entire image using Tesseract
    spanish_text = pytesseract.image_to_string(image_rgb, config=tesseract_config).strip()

    # Perform OCR on the preprocessed ROI using EasyOCR
    extracted_times = []
    results_easyocr = easyocr_reader.readtext(enhanced_image)
    for _, text, _ in results_easyocr:
        cleaned = clean_time(text)
        if cleaned:
            extracted_times.append(cleaned)

    ocr_results = {
        'spanish_text': spanish_text,
        'times': extracted_times
    }

    return ocr_results


def clean_time(text, assume_pm=True):
    """
    Clean and correct the OCR result for time.
    Replaces common misreads and extracts time in HH:MM format.
    Appends 'pm' if neither 'am' nor 'pm' is detected and assume_pm is True.

    Parameters:
        text (str): The OCR-extracted text string.
        assume_pm (bool): Whether to append 'pm' if AM/PM is missing.

    Returns:
        str or None: The cleaned time string, or None if no valid time is found.
    """
    # Define a mapping of common misrecognized characters to their correct counterparts
    replacements = {
        '.': ':',  # Replace periods with colons
        'I': ':',  # Replace uppercase 'I' with colon
        'i': ':',  # Replace lowercase 'i' with colon
        'L': '1',  # Replace uppercase 'L' with '1'
        'l': '1',  # Replace lowercase 'l' with '1'
        '|': '1',  # Replace pipe character with '1'
        'O': '0',  # Replace uppercase 'O' with '0'
        'o': '0'   # Replace lowercase 'o' with '0'
    }

    # Apply the replacements
    for key, value in replacements.items():
        text = text.replace(key, value)

    # Remove any characters that are not digits, colon, or am/pm indicators
    text = re.sub(r'[^0-9:apmAPM]', '', text)

    # Replace multiple colons with a single colon
    text = re.sub(r':{2,}', ':', text)

    # Extract time using regex
    time_pattern = r'(\d{1,2}:\d{2}(?:\s*[ap]m)?)'
    match = re.search(time_pattern, text, re.IGNORECASE)

    if match:
        time_str = match.group(1).lower().strip()

        # Append 'pm' if neither 'am' nor 'pm' is present and assume_pm is True
        if 'am' not in time_str and 'pm' not in time_str and assume_pm:
            time_str += ' pm'

        return time_str
    else:
        return None


def get_ocr_results(image, roi_coords):
    """
    Get OCR results from the specified image and ROI coordinates.

    Parameters:
        image (numpy.ndarray): The input image.
        roi_coords (tuple): A tuple (roi_x, roi_y, roi_w, roi_h) specifying the ROI.

    Returns:
        dict: OCR results containing 'spanish_text' and 'times'.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    preprocessed_region = preprocess_roi_for_time(image_rgb, roi_coords)
    ocr_results = ocr_time(preprocessed_region, image_rgb)
    return ocr_results
