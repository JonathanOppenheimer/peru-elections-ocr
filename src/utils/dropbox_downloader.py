import dropbox
import json
import os
import pickle
import time
import threading

from concurrent.futures import ThreadPoolExecutor, as_completed
from dropbox.exceptions import ApiError
from dropbox.files import FileMetadata, SharedLink
from ratelimit import limits, sleep_and_retry
from typing import Set

# Rate limit: 1000 calls per minute
CALLS = 1000
RATE_LIMIT_PERIOD = 60  # 1 minute in seconds

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

@sleep_and_retry
@limits(calls=CALLS, period=RATE_LIMIT_PERIOD)
def list_folder_with_retry(dbx, shared_link, cursor=None, max_retries=3):
    """List folder contents with retry logic"""
    for attempt in range(max_retries):
        try:
            if cursor:
                return dbx.files_list_folder_continue(cursor)
            else:
                return dbx.files_list_folder(path='', shared_link=shared_link)
        except ApiError as e:
            if e.error.is_rate_limited():
                # If we hit the rate limit, wait for the time specified in the error
                time.sleep(e.error.get_retry_after() or 60)
                continue
            elif attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"Listing attempt {attempt + 1} failed. Retrying in {wait_time} seconds...")
                print(f"Error: {e}")
                time.sleep(wait_time)
            else:
                raise
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"Listing attempt {attempt + 1} failed. Retrying in {wait_time} seconds...")
                print(f"Error: {e}")
                time.sleep(wait_time)
            else:
                raise

def download_pdfs(dbx, shared_url: str, download_path: str, max_workers: int = 4, save_frequency: int = 50, batch_size: int = 1000):
    """Download PDF files from a flat shared folder with resume capability using parallel downloads"""
    try:
        os.makedirs(download_path, exist_ok=True)
        downloaded_files = load_downloaded_files(download_path)
        print(f"Found {len(downloaded_files)} previously downloaded files")
        
        shared_link = SharedLink(url=shared_url)
        print("Starting file download process...")
        
        # Track progress with thread-safe counter
        successful = len(downloaded_files)
        failed_files = []
        counter_lock = threading.Lock()
        total_pdfs = 0
        
        def download_file(entry):
            nonlocal successful
            if entry.name in downloaded_files:
                return None
            
            file_path = os.path.join(download_path, entry.name)
            if download_with_retry(dbx, shared_url, entry.name, file_path):
                with counter_lock:
                    successful += 1
                    downloaded_files.add(entry.name)
                    if successful % save_frequency == 0:
                        save_downloaded_files(download_path, downloaded_files)
                print(f"Successfully downloaded ({successful}): {entry.name}")
                return entry.name
            else:
                failed_files.append(entry.name)
                return None

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                cursor = None
                while True:
                    try:
                        # Get batch with retry logic
                        result = list_folder_with_retry(dbx, shared_link, cursor)
                        cursor = result.cursor
                        
                        # Filter PDF files from current batch
                        batch_files = [
                            entry for entry in result.entries 
                            if isinstance(entry, FileMetadata) and 
                            entry.name.lower().endswith('.pdf')
                        ]
                        total_pdfs += len(batch_files)
                        
                        if batch_files:
                            # Process current batch with parallel downloads
                            futures = [
                                executor.submit(download_file, entry)
                                for entry in batch_files
                                if entry.name not in downloaded_files
                            ]
                            
                            # Wait for current batch to complete
                            for future in as_completed(futures):
                                try:
                                    future.result()  # This ensures we catch any exceptions
                                except Exception as e:
                                    print(f"Error processing file: {e}")
                            
                            # Save progress after each batch
                            save_downloaded_files(download_path, downloaded_files)
                        
                        if not result.has_more:
                            break
                            
                        print(f"\nProcessed {total_pdfs} files so far. Getting next batch...")
                        
                    except Exception as e:
                        print(f"Error processing batch: {e}")
                        print("Waiting 60 seconds before retrying...")
                        time.sleep(60)  # Wait a minute before retrying the batch
                    
        except KeyboardInterrupt:
            print("\nDownload interrupted by user. Progress has been saved.")
        finally:
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
    
    # Download the PDFs with 4 parallel workers
    download_pdfs(dbx, shared_url, download_path, max_workers=4)

if __name__ == "__main__":
    main()
