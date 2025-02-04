# OCR Signature Detection Project

This project focuses on detecting and counting handwritten signatures from specific areas on standardized Peru ONPE election records. The PDFs contain multiple signature boxes on different pages, and the project applies various image processing techniques to accurately detect these signatures.

## Table of Contents
1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Main Script (main.py) Explanation](#main-script-mainpy-explanation)
5. [Image Processing Techniques](#image-processing-techniques)
6. [Contributing](#contributing)
7. [License](#license)

---

## Project Structure

```
.
├── src/                        # Source code
│   ├── ocr/                    # OCR-related modules
│   │   ├── __init__.py
│   │   ├── template_generation.py
│   │   └── ocr_signature_detection.py
│   └── utils/                  # Utility modules
│       ├── __init__.py
│       └── dropbox_downloader.py
├── config/                     # Configuration files
│   └── secret.json            # API keys and credentials
├── data/                      # Input PDFs and output CSV files
│   ├── downloaded_pdfs/       # Downloaded PDF files
│   └── output/                # Processing results
├── templates/                 # Empty signature box templates
│   └── r2/                    # Example template
│       └── bounding_boxes.json 
├── main.py                    # Main entry point
├── setup.py                   # Package installation
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Installation

To set up this project locally, follow these steps:

### 1. Clone the Repository
First, clone the repository to your local machine:
```bash
git clone https://github.com/your-repo-url.git
cd your-repo-url
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

To run the program:

1. Add your Dropbox credentials to `config/secret.json`
2. Run the main script:
   ```bash
   python main.py
   ```

3. The results, including the table number and the count of signatures from the different boxes (`numobs1`, `numobs2`, `numobs3`), will be saved to a CSV file in the `./data/output/` directory.

### Directory Structure
The key directories used by this project are:
- `./data/downloaded_pdfs/` - For storing downloaded PDFs.
- `./data/output/` - For storing processed results.
- `./templates/` - For storing bounding box templates.

## Main Script (main.py) Explanation

The `main.py` script serves as the entry point for the signature detection process. Here's how it works:

### Configuration Loading
- Loads secret configuration (Dropbox credentials) from `config/secret.json`
- Initializes necessary directories:
  - `data/downloaded_pdfs/`: For storing downloaded PDF files
  - `data/output/`: For storing processing results

### Process Flow
1. **Dropbox Integration**
   - Initializes Dropbox client using access token
   - Downloads PDF files from specified shared folder

2. **Document Processing**
   - After downloading, processes each PDF document to:
     - Extract table numbers
     - Detect signatures in specified regions
     - Count signatures in each region
     - Generate output CSV with results

### Usage Example

```bash
python main.py
```

This will:
1. Download PDFs from Dropbox
2. Process each PDF for signature detection
3. Save results to the output directory

### Required Configuration
Before running, ensure you have:
1. Created `config/secret.json` with:
   ```json
   {
     "dropbox_access_token": "your_access_token",
     "dropbox_shared_url": "your_shared_folder_url"
   }
   ```
2. Set up the necessary directory structure
3. Installed all required dependencies

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

Contributions are welcome! If you have suggestions or find issues, feel free to open a pull request or issue on the repository.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### Additional Notes
- **Templates**: To improve signature detection, it's important to have accurate templates of the empty signature boxes for `numobs1`, `numobs2`, and `numobs3`. These templates can be manually selected using the post-processed images generated by the system.
- **Performance Considerations**: If you notice any false positives or issues with the SSIM similarity score, try adjusting the SSIM threshold in the `template_match_signature_area` function.

