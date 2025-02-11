# OCR Signature Detection Project

This project focuses on detecting and counting handwritten signatures from specific areas on standardized Peru ONPE election records (Actas). The system processes PDF documents containing multiple signature boxes across different pages, using advanced image processing and OCR techniques to accurately detect signatures and extract relevant metadata such as table numbers and document types.

## Table of Contents
1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Configuration](#configuration)
5. [Template System](#template-system)
6. [Image Processing Techniques](#image-processing-techniques)
7. [Contributing](#contributing)
8. [License](#license)

---

## Project Structure

```
.
├── config/                    # Configuration files
│   └── secret.json            # API keys and credentials
├── data/                     # Data directory
│   ├── input/               # Input PDFs for processing
│   │   └── testing/        # Test PDF files
│   ├── downloaded_pdfs/     # PDFs downloaded from Dropbox
│   └── output/              # Processing results
│       └── csv/            # CSV output files
├── src/                      # Source code
│   ├── ocr/                 # OCR-related modules
│   │   ├── __init__.py
│   │   ├── ocr_signature_detection.py  # Main signature detection logic
│   │   ├── template_generation.py      # Template generation utilities
│   │   └── time_extraction.py          # Time extraction functionality
│   └── utils/               # Utility modules
│       ├── __init__.py
│       └── dropbox_downloader.py  # Dropbox integration
├── templates/                # Template files
│   └── erm/                 # ERM template directory
│       ├── bounding_boxes.json     # Box coordinates
│       ├── empty_numobs1.png       # Empty signature templates
│       ├── empty_numobs2.png
│       └── empty_numobs3.png
├── main.py                   # Main entry point
├── setup.py                  # Package installation
├── requirements.txt          # Project dependencies
└── README.md                # Project documentation
```

## Installation

To set up this project locally, follow these steps:

### 1. Clone the Repository
First, clone the repository to your local machine:
```bash
git clone https://github.com/JonathanOppenheimer/peru-elections-ocr
cd peru-elections-ocr
```

### 2. Set Up a Virtual Environment (Optional but recommended)
Create a virtual environment to isolate the project's dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows, use .venv\Scripts\activate
```

### 3. Install Package
Install the package in development mode:
```bash
pip install -e .
```

### 4. Install Required System Dependencies

#### Install Tesseract OCR:

- **macOS**: 
  ```bash
  brew install tesseract
  ```

- **Ubuntu/Debian**:
  ```bash
  sudo apt install tesseract-ocr
  ```

- **Windows**:
  Download and install Tesseract from [here](https://github.com/tesseract-ocr/tesseract/wiki).

#### Install Poppler:

Poppler is required for PDF processing. Install it as follows:

- **macOS**:
  ```bash
  brew install poppler
  ```

- **Ubuntu/Debian**:
  ```bash
  sudo apt-get install poppler-utils
  ```

- **Windows**:
  1. Download the latest binary from [here](http://blog.alivate.com.au/poppler-windows/)
  2. Extract to a folder (e.g., `C:\Program Files\poppler-xx\`)
  3. Add the bin folder to your system PATH

### 5. Verify Installation
To verify that everything is installed correctly, try running the example script:
```bash
python ocr_signature_detection.py
```

---

## Usage

### Basic Usage

1. Configure your environment:
   ```bash
   # Set up config/secret.json with your credentials
   {
     "dropbox_access_token": "your_access_token",
     "dropbox_shared_url": "your_shared_folder_url"
   }
   ```

2. Run the main script:
   ```bash
   python main.py
   ```

3. For processing a specific folder of PDFs:
   ```python
   from src.ocr import process_folder
   
   process_folder(
       input_folder="data/input/testing",
       empty_template_paths={
           "numobs1": "templates/erm/empty_numobs1.png",
           "numobs2": "templates/erm/empty_numobs2.png",
           "numobs3": "templates/erm/empty_numobs3.png"
       },
       csv_output_path="data/output/csv/results.csv",
       bounding_boxes_path="templates/erm/bounding_boxes.json"
   )
   ```

### Output Format

The program generates a CSV file with the following columns:
- `table_number`: Extracted table number from the document
- `document_type`: Type of the document (extracted via OCR)
- `numobs1`: Number of signatures detected in the first observation box
- `numobs2`: Number of signatures detected in the second observation box
- `numobs3`: Number of signatures detected in the third observation box


## Configuration

### Bounding Boxes Configuration

The `bounding_boxes.json` file defines the regions of interest for signature detection and metadata extraction:

```json
{
    "numobs1": {
        "left_pct": 0.1,
        "top_pct": 0.2,
        "right_pct": 0.3,
        "bottom_pct": 0.4,
        "grid": {
            "rows": 3,
            "columns": 2
        }
    },
    "mesa_sufragio": {
        "left_pct": 0.5,
        "top_pct": 0.1,
        "right_pct": 0.7,
        "bottom_pct": 0.15
    }
}
```

Each box is defined by:
- Percentage-based coordinates (`left_pct`, `top_pct`, `right_pct`, `bottom_pct`)
- Optional grid configuration for signature boxes
- Optional ROI (Region of Interest) for specific areas within boxes

## Template System

The project uses a template-based approach for signature detection:

### Template Generation

Templates can be generated using the template generation utility:

```bash
python -m src.ocr.template_generation
```

This will:
1. Process an empty form PDF
2. Extract signature box regions
3. Generate preprocessed templates for each box type
4. Save templates in the specified output directory

### Template Matching

---

## Image Processing Techniques

This project leverages several image processing techniques to reliably detect signatures within predefined regions of interest (bounding boxes). Below is a brief explanation of the techniques used:

### 1. **Grayscale Conversion**
Images are converted from RGB to grayscale. This helps simplify the image and reduces the complexity for further processing. Grayscale conversion is useful for working with binary thresholding and other pixel intensity-based techniques.

### 2. **Median Blurring**
We apply a **median blur** to the image, which is particularly useful for reducing noise (like ink spots or specks of dust). This type of blur helps remove salt-and-pepper noise while preserving the edges of the content (like handwritten signatures).

### 3. **Otsu's Thresholding**
Otsu's method is an automatic thresholding technique that determines the optimal threshold value to convert the grayscale image into a binary image (black and white). This is critical for differentiating between the background and the actual signatures.

### 4. **Structural Similarity Index (SSIM)**
For signature detection, we use the **SSIM** algorithm to compare each signature box with a predefined empty template. SSIM measures the similarity between two images and helps us identify whether a signature is present in a given box. If the similarity score falls below a certain threshold, we assume a signature is present.

---

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

For major changes, please open an issue first to discuss the proposed changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### Additional Notes

- **Template Quality**: The quality of signature detection heavily depends on the templates used. Ensure templates are generated from clean, empty forms.
- **Performance Tuning**: Adjust the SSIM threshold in `template_match_signature_area()` if experiencing false positives/negatives.
- **Batch Processing**: For large datasets, use the batch processing feature with appropriate batch sizes to optimize memory usage.

