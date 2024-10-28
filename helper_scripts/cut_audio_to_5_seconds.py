folder="C:/Users/pietr/Desktop/Nowy folder"
output_folder="C:/Users/pietr/Desktop/Nowy folder (2)"

#load files
import os
import soundfile as sf
import numpy as np

files = os.listdir(folder)
for index,file in enumerate(files):
    if file.endswith(".wav"):
        data, samplerate = sf.read(os.path.join(folder, file))
        print(f"Loaded {file} with {len(data)} samples and samplerate {samplerate}")
        #cut into 5 seconds parts
        length=len(data)
        for i in range(0,length,samplerate*5):
            name=os.path.join(output_folder, file[:-4]+"_"+str(i)+".wav")
            data_part=data[i:i+samplerate*5]
            if(len(data_part)<samplerate*5):
                break
            sf.write(name,  data_part, samplerate)
            #print(f"Cut {name} to 5 seconds")
    
    print(f"Done with {file}, {index}")
        
print("Done")

