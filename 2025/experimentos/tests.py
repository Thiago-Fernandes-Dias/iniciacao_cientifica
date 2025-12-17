import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import signal

# Open the TIFF image file
# Use the correct path to your image file
image_path = "101_5.tif"
img = Image.open(image_path)

# Convert the image to a NumPy array for processing (optional, but common)
img_array = np.array(img)
M = np.mean(img_array)
V = np.var(img_array)
M0 = M * 0.005
V0 = V * 0.005

for i, p in np.ndenumerate(img_array):
    if img_array[i] > M:
        img_array[i] = M0 + ((V0 * (img_array[i] - M) ** 2) / V) ** 0.5
    else:
        img_array[i] = M0 - ((V0 * (img_array[i] - M) ** 2) / V) ** 0.5

# Define Sobel Kernels
Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

Gx = signal.convolve2d(img_array, Kx, boundary="symm", mode="same")
Gy = signal.convolve2d(img_array, Ky, boundary="symm", mode="same")

# Calculate Gradient Magnitude
# Formula: G = sqrt(Gx^2 + Gy^2)
G = np.sqrt(np.square(Gx) + np.square(Gy))

# Normalize to 0-255 for visualization
G = (G / G.max()) * 255
edge_img = Image.fromarray(G.astype(np.uint8))
edge_img.show()

img.close()
