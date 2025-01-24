import os
import zipfile

# Define paths
input_dir = "data/era5"  # Directory containing the .nc files (zip archives)
output_dir = "data/era5_new"  # Directory to store unzipped files

year = range(1940, 2025)

for yr in year:
    # Check if the file is a ZIP archive
    zip_path = os.path.join(input_dir, f'era5_{yr}.nc')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Extract all contents to the output directory
        new_output_dir = os.path.join(output_dir, f'era5_{yr}')
        zip_ref.extractall(new_output_dir)
        
# # Iterate through all files in the input directory
# for file_name in os.listdir(input_dir):
#     # Check if the file is a ZIP archive
#     if file_name.endswith(".nc"):
#         zip_path = os.path.join(input_dir, file_name)
#         with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#             # Extract all contents to the output directory
#             zip_ref.extractall(output_dir)
#             print(f"Extracted: {file_name} to {output_dir}")
