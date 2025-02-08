import csv
import os

def find_unmatched_files(csv_path, folder_path):
    """
    Reads the CSV file at csv_path, extracts the filenames from the first column (ignoring the header),
    and then checks the files in folder_path. Returns a list of files in the folder (by filename) that 
    do not have a corresponding row in the CSV.
    """
    # Read the CSV file and collect the filenames from the first column.
    csv_filenames = set()
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        # Skip the header row.
        next(reader, None)
        for row in reader:
            if row:
                # Assuming the first column contains the filename (without extension).
                csv_filenames.add(row[0].strip())

    # List all files in the folder.
    unmatched_files = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # Process only files (ignore subdirectories).
        if os.path.isfile(file_path):
            # Remove the file extension so that "filename.pdf" becomes "filename".
            base_name, _ = os.path.splitext(filename)
            # If the file’s base name is not in the CSV list, add it to the result.
            if base_name not in csv_filenames:
                unmatched_files.append(filename)

    return unmatched_files

def main():
    # Define the path to the CSV file and the folder containing the files.
    csv_file = "path/to/yourfile.csv"   # Replace with the path to your CSV file
    folder = "path/to/your/folder"        # Replace with the path to your folder

    missing_files = find_unmatched_files(csv_file, folder)
    if missing_files:
        print("The following files are in the folder but not in the CSV:")
        for file in missing_files:
            print(f" - {file}")
    else:
        print("Every file in the folder has a corresponding row in the CSV.")

if __name__ == '__main__':
    main()
