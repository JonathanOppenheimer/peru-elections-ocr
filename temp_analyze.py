import pdf2image
import pytesseract
import cv2
import numpy as np
import json

def extract_document_type(image, bbox):
    img_width, img_height = image.size
    left = int(bbox['left_pct'] * img_width)
    top = int(bbox['top_pct'] * img_height)
    right = int(bbox['right_pct'] * img_width)
    bottom = int(bbox['bottom_pct'] * img_height)
    
    cropped = image.crop((left, top, right, bottom))
    img_np = np.array(cropped)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(thresh, config='--oem 3 --psm 6', lang='spa')
    return ' '.join(text.strip().split())

with open('templates/erm/bounding_boxes.json', 'r') as f:
    bbox_config = json.load(f)
doc_type_bbox = bbox_config['default']['document_type']

pdfs = [
    'data/input/testing/ACERMC3709407734907.PDF',
    'data/input/testing/ACERMC3709307844823.PDF',
    'data/input/testing/ACERMC3709207405405.PDF',
    'data/input/testing/ACERMC3709107311512.PDF'
]

for pdf_path in pdfs:
    print(f'\nAnalyzing {pdf_path}:')
    images = pdf2image.convert_from_path(pdf_path)
    doc_type = extract_document_type(images[0], doc_type_bbox)
    print(f'Document type: {doc_type}') 