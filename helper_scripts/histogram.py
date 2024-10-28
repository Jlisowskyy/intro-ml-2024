import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class1 = ["f1", "f7", "f8", "m3", "m6", "m8"]
file = r"C:\Users\pietr\Documents\studia\machine learning\histogram.txt"


array_class1 = [[0] * 256 for _ in range(3)]  
array_class0 = [[0] * 256 for _ in range(3)]  

num_class1 = 0
num_class0 = 0

with open(file, "r") as f:
    
    for line in f:
        
        splited = line.split(";")
        
        speaker = splited[0].split("_")[0]
        for i in range(0, 255):  
           
            nums = splited[i+1].split(":")[1].strip()  
            nums = nums.split(",") 
            
            if speaker in class1:
                num_class1 += 1
                array_class1[0][i] += int(nums[0])  
                array_class1[1][i] += int(nums[1])  
                array_class1[2][i] += int(nums[2])  
            else:
                num_class0 += 1
                array_class0[0][i] += int(nums[0])  
                array_class0[1][i] += int(nums[1])  
                array_class0[2][i] += int(nums[2])

for i in range(0, 255):  
    for j in range(3):  
        array_class1[j][i] /= num_class1
        array_class0[j][i] /= num_class0
        
print(array_class1)
print(array_class0)



intensity_levels = np.arange(256)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.bar(intensity_levels, array_class0[0], color='red', alpha=0.5, label='Red')
plt.bar(intensity_levels, array_class0[1], color='green', alpha=0.5, label='Green')
plt.bar(intensity_levels, array_class0[2], color='blue', alpha=0.5, label='Blue')
plt.title('RGB Histogram - Class 0')
plt.xlabel('Intensity Level')
plt.ylabel('Frequency')
plt.xlim(0, 255)
plt.legend()
plt.grid(axis='y')

plt.subplot(1, 2, 2)
plt.bar(intensity_levels, array_class1[0], color='red', alpha=0.5, label='Red')
plt.bar(intensity_levels, array_class1[1], color='green', alpha=0.5, label='Green')
plt.bar(intensity_levels, array_class1[2], color='blue', alpha=0.5, label='Blue')
plt.title('RGB Histogram - Class 1')
plt.xlabel('Intensity Level')
plt.ylabel('Frequency')
plt.xlim(0, 255)
plt.legend()
plt.grid(axis='y')

plt.tight_layout()
plt.show()