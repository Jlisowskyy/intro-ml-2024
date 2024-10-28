import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

folder = r"C:\Users\pietr\Documents\studia\machine learning\cut_audio_spectrogram2"
output_file=r"C:\Users\pietr\Documents\studia\machine learning\histogram.txt"

files= os.listdir(folder)

# Open the output file in append mode
with open(output_file, 'a') as file:
    for index, file_name in enumerate(files):
        image = Image.open(os.path.join(folder, file_name))
        image_array = np.array(image)

        # Separate the color channels
        r_channel = image_array[:, :, 0].flatten()
        g_channel = image_array[:, :, 1].flatten()
        b_channel = image_array[:, :, 2].flatten()

        counts_r = np.bincount(r_channel, minlength=256)
        counts_g = np.bincount(g_channel, minlength=256)
        counts_b = np.bincount(b_channel, minlength=256)

        output_string = f"{file_name};"
        for i in range(256):  # Use a different variable name for the loop
            output_string += f"{i}: {counts_r[i]}, {counts_g[i]}, {counts_b[i]};"
        print(index)
        #print(output_string)  # Print to console
        file.write(output_string + '\n')  # Append to the file and add a newline
        