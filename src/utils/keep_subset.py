import os
import shutil

def keep_only_listed_files(file_list_path, folder_path, backup=True):
    """
    Removes all files from folder except those listed in the file.
    
    Parameters:
        file_list_path (str): Path to text file containing filenames to keep
        folder_path (str): Path to folder containing files
        backup (bool): If True, creates backup folder with removed files
    """
    # Read the list of files to keep, removing "- " prefix
    with open(file_list_path, 'r') as f:
        files_to_keep = {
            line.strip().removeprefix('- ') 
            for line in f 
            if line.strip()
        }
    
    print(f"Found {len(files_to_keep)} files to keep")
    
    # Create backup folder if requested
    if backup:
        backup_folder = os.path.join(folder_path, 'backup')
        os.makedirs(backup_folder, exist_ok=True)
    
    # Process files in the folder
    files_processed = 0
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Skip if it's a directory or in the keep list
        if os.path.isdir(file_path) or filename in files_to_keep:
            continue
            
        if backup:
            # Move to backup
            shutil.move(file_path, os.path.join(backup_folder, filename))
        else:
            # Delete the file
            os.remove(file_path)
        files_processed += 1
            
    print(f"Finished. Processed {files_processed} files, kept {len(files_to_keep)} files.")

def main():
    # Define paths
    file_list_path = 'files_to_process.txt'  # File containing list of PDFs to keep
    folder_path = 'data/downloaded_pdfs'      # Folder containing all PDFs
    
    # Ensure paths exist
    if not os.path.exists(file_list_path):
        raise FileNotFoundError(f"File list not found at: {file_list_path}")
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"PDF folder not found at: {folder_path}")
    
    # Process the files
    print(f"Starting to process files...")
    print(f"File list: {file_list_path}")
    print(f"PDF folder: {folder_path}")
    
    keep_only_listed_files(file_list_path, folder_path, backup=True)

if __name__ == "__main__":
    main() 