import dropbox
from dropbox.exceptions import ApiError
import json
import os
from dropbox.files import FileMetadata, SharedLink
import time
from typing import Set
import pickle

def load_downloaded_files(download_path: str) -> Set[str]:
    """Load the set of already downloaded files"""
    tracking_file = os.path.join(download_path, '.downloaded_files.pkl')
    if os.path.exists(tracking_file):
        with open(tracking_file, 'rb') as f:
            return pickle.load(f)
    return set()

def save_downloaded_files(download_path: str, downloaded_files: Set[str]):
    """Save the set of downloaded files"""
    tracking_file = os.path.join(download_path, '.downloaded_files.pkl')
    with open(tracking_file, 'wb') as f:
        pickle.dump(downloaded_files, f)

def download_with_retry(dbx, shared_url: str, entry_name: str, file_path: str, max_retries: int = 3) -> bool:
    """Download a single file with retry logic"""
    for attempt in range(max_retries):
        try:
            metadata, res = dbx.sharing_get_shared_link_file(
                url=shared_url,
                path=f"/{entry_name}"
            )
            
            with open(file_path, 'wb') as f:
                f.write(res.content)
            return True
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Attempt {attempt + 1} failed for {entry_name}. Retrying in {wait_time} seconds...")
                print(f"Error: {e}")
                time.sleep(wait_time)
            else:
                print(f"All attempts failed for {entry_name}")
                print(f"Final error: {e}")
                return False

def download_pdfs(dbx, shared_url: str, download_path: str):
    """Download PDF files from a flat shared folder with resume capability"""
    try:
        # Ensure download directory exists
        os.makedirs(download_path, exist_ok=True)
        
        # Load set of already downloaded files
        downloaded_files = load_downloaded_files(download_path)
        print(f"Found {len(downloaded_files)} previously downloaded files")
        
        # Create SharedLink object
        shared_link = SharedLink(url=shared_url)
        
        print("Listing files in shared folder...")
        result = dbx.files_list_folder(
            path='',
            shared_link=shared_link
        )
        
        successful = len(downloaded_files)
        total_pdfs = 0
        failed_files = []
        
        try:
            while True:  # Loop to handle pagination
                for entry in result.entries:
                    if isinstance(entry, FileMetadata) and entry.name.lower().endswith('.pdf'):
                        total_pdfs += 1
                        
                        # Skip if already downloaded
                        if entry.name in downloaded_files:
                            print(f"Skipping already downloaded: {entry.name}")
                            continue
                            
                        print(f"Downloading ({total_pdfs}): {entry.name}")
                        file_path = os.path.join(download_path, entry.name)
                        
                        if download_with_retry(dbx, shared_url, entry.name, file_path):
                            print(f"Successfully downloaded: {entry.name}")
                            successful += 1
                            downloaded_files.add(entry.name)
                            # Save progress periodically
                            if successful % 10 == 0:
                                save_downloaded_files(download_path, downloaded_files)
                        else:
                            failed_files.append(entry.name)
                
                # Check if there are more files to process
                if result.has_more:
                    print("\nGetting next batch of files...")
                    result = dbx.files_list_folder_continue(result.cursor)
                else:
                    break
                
        except KeyboardInterrupt:
            print("\nDownload interrupted by user. Progress has been saved.")
        finally:
            # Save final progress
            save_downloaded_files(download_path, downloaded_files)
        
        print(f"\nDownload complete. Successfully downloaded {successful} of {total_pdfs} PDF files.")
        if failed_files:
            print(f"Failed to download {len(failed_files)} files:")
            for failed_file in failed_files:
                print(f"- {failed_file}")
        
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
