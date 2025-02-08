import csv
import os

def find_unmatched_files(csv_path, folder_path):
    """
    Reads the CSV file at csv_path, extracts the filenames from the first column (ignoring the header),
    strips any leading zeros, and then checks the files in folder_path (also stripping their leading zeros).
    Returns a list of files in the folder (with their original names, including extensions) that do not have
    a corresponding row in the CSV.
    """
    # Read the CSV file and collect the filenames from the first column.
    csv_filenames = set()
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        # Skip the header row.
        next(reader, None)
        for row in reader:
            if row:
                # Remove any surrounding whitespace and strip leading zeros.
                filename = row[0].strip().lstrip('0')
                csv_filenames.add(filename)

    unmatched_files = []
    # List all files in the folder.
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # Process only files (ignore subdirectories).
        if os.path.isfile(file_path):
            # Remove the file extension so that "filename.pdf" becomes "filename"
            base_name, _ = os.path.splitext(filename)
            # Strip leading zeros from the folder filename.
            base_name_nozeros = base_name.lstrip('0')
            # If the processed base name is not in the CSV set, add the original filename to the result.
            if base_name_nozeros not in csv_filenames:
                unmatched_files.append(filename)
    return unmatched_files

def main():
    # Define the path to the CSV file and the folder containing the files.
    csv_file = "./data/output/csv/Actas2018rd2.csv"   # Replace with the path to your CSV file
    folder = "./data/Actas2018rd2"        # Replace with the path to your folder
    # Define the output file path.
    output_file = "./data/output/dif"

    # Get the list of unmatched files.
    missing_files = find_unmatched_files(csv_file, folder)
    
    # Ensure the output directory exists.
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the results to the output file.
    with open(output_file, 'w', encoding='utf-8') as f:
        if missing_files:
            f.write("The following files are in the folder but not in the CSV:\n")
            for file in missing_files:
                f.write(f" - {file}\n")
        else:
            f.write("Every file in the folder has a corresponding row in the CSV.\n")
    
    print(f"Results saved to {output_file}")

if __name__ == '__main__':
    main()
