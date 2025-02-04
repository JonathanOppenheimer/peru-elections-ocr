import os
from src.utils.dropbox_downloader import download_pdfs
from src.ocr.ocr_signature_detection import process_documents  # You'll need to import your main OCR function
import dropbox
import json

def main():
    # Load configuration
    config_path = os.path.join('config', 'secret.json')
    with open(config_path) as f:
        secrets = json.load(f)
    
    # Initialize paths
    download_path = os.path.join('data', 'downloaded_pdfs')
    output_path = os.path.join('data', 'output')
    
    # Ensure directories exist
    os.makedirs(download_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    
    # Initialize Dropbox and download files
    dbx = dropbox.Dropbox(secrets['dropbox_access_token'])
    download_pdfs(dbx, secrets['dropbox_shared_url'], download_path)
    
    # Process the documents
    # Uncomment and modify based on your OCR function
    # process_documents(download_path, output_path)

if __name__ == "__main__":
    main() 