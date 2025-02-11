import dropbox
import json
import os

from src.utils.dropbox_downloader import download_pdfs
from src.ocr.ocr_signature_detection import process_folder

def main():
    # Load configuration
    config_path = os.path.join('config', 'secret.json')
    with open(config_path) as f:
        secrets = json.load(f)
    
    # Initialize paths
    project_root = os.path.dirname(__file__)
    download_path = os.path.join('data', 'input', 'downloaded_pdfs')
    output_path = os.path.join('data', 'output', 'csv')
    bounding_boxes_path = os.path.join(project_root, 'templates', 'erm', 'bounding_boxes.json')
    
    # Ensure directories exist
    os.makedirs(download_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    
    # Initialize Dropbox and download files
    dbx = dropbox.Dropbox(secrets['dropbox_access_token'])
    download_pdfs(dbx, secrets['dropbox_shared_url'], download_path)
    
    # Process the documents
    empty_template_paths = {
        "numobs1": os.path.join(project_root, 'templates', 'erm', 'empty_numobs1.png'),
        "numobs2": os.path.join(project_root, 'templates', 'erm', 'empty_numobs2.png'),
        "numobs3": os.path.join(project_root, 'templates', 'erm', 'empty_numobs3.png')
    }
    
    csv_output_path = os.path.join(output_path, 'signature_results.csv')
    print(f"Processing PDFs from {download_path}")
    print(f"Output will be saved to {csv_output_path}")
    
    process_folder(download_path, empty_template_paths, csv_output_path, bounding_boxes_path, debug=True)

if __name__ == "__main__":
    main() 