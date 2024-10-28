import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

folder = r"C:\Users\pietr\Documents\studia\machine learning\cut_audio"
output_folder = r"C:\Users\pietr\Documents\studia\machine learning\cut_audio_spectrogram2"

# Function to process a single file
def process_file(file):
    if file.endswith(".wav"):
        command = f'python -m helper_scripts.spectrogram_script "{folder}\\{file}" --output "{output_folder}\\{file[:-4]}.png" --mel'
        os.system(command)  # You might want to consider using subprocess.run instead
        print(f"Done with {file}")
        return f"Done with {file}"
    return None

# Get all files in the directory
files = os.listdir(folder)

# Use ThreadPoolExecutor to process files in parallel
with ThreadPoolExecutor(max_workers=5) as executor:
    results = list(executor.map(process_file, files))

# Print the results
for result in results:
    if result:
        print(result)
