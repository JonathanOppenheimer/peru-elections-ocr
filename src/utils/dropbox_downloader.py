import dropbox
from dropbox.exceptions import ApiError
import json
import os
from dropbox.files import FileMetadata, SharedLink

def download_pdfs(dbx, shared_url, download_path):
    """Download PDF files from a flat shared folder"""
    try:
        # Create SharedLink object
        shared_link = SharedLink(url=shared_url)
        
        # Ensure download directory exists
        os.makedirs(download_path, exist_ok=True)
        
        print("Listing files in shared folder...")
        result = dbx.files_list_folder(
            path='',  # Use empty path for root of shared folder
            shared_link=shared_link
        )
        
        # Download PDF files
        successful = 0
        total_pdfs = 0
        
        for entry in result.entries:
            if isinstance(entry, FileMetadata) and entry.name.lower().endswith('.pdf'):
                total_pdfs += 1
                print(f"Downloading: {entry.name}")
                try:
                    # Try direct download with sharing_get_shared_link_file
                    metadata, res = dbx.sharing_get_shared_link_file(
                        url=shared_url,
                        path=f"/{entry.name}"  # Add leading slash to path
                    )
                    
                    # Save the file
                    file_path = os.path.join(download_path, entry.name)
                    with open(file_path, 'wb') as f:
                        f.write(res.content)
                    print(f"Successfully downloaded: {entry.name}")
                    successful += 1
                except Exception as e:
                    print(f"Error downloading {entry.name}: {e}")
                    print(f"Full error details for {entry.name}:", str(e))
        
        print(f"Download complete. Successfully downloaded {successful} of {total_pdfs} PDF files.")
        
    except ApiError as e:
        print(f"Error accessing folder: {e}")
        print("Full error details:", str(e))

def main():
    # Update config path
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'secret.json')
    
    # Load credentials
    with open(config_path) as f:
        secrets = json.load(f)
    
    # Initialize Dropbox client    
    dbx = dropbox.Dropbox(secrets['dropbox_access_token'])
    shared_url = secrets['dropbox_shared_url']
    
    # Set download path relative to project root
    download_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "downloaded_pdfs")
    
    # Download the PDFs
    download_pdfs(dbx, shared_url, download_path)

if __name__ == "__main__":
    main()
