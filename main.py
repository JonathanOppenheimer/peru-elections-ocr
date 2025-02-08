import dropbox
import json
import os

from src.utils.dropbox_downloader import download_pdfs

def main():
    # Load configuration
    config_path = os.path.join('config', 'secret.json')
    with open(config_path) as f:
        secrets = json.load(f)
    
    # Initialize paths
    download_path = os.path.join('data', 'input', 'downloaded_pdfs')
    output_path = os.path.join('data', 'output')
    
    # Ensure directories exist
    os.makedirs(download_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    
    # Initialize Dropbox and download files
    dbx = dropbox.Dropbox(secrets['dropbox_access_token'])
    download_pdfs(dbx, secrets['dropbox_shared_url'], download_path)
    
    # Process the documents
    #TODO add main running OCR function 

if __name__ == "__main__":
    main() 