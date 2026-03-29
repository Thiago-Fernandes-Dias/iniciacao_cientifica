import os
from PIL import Image

input_folder = "datasets/FVC/FVC2004/Dbs/DB1_B"
output_folder = "datasets/FVC/FVC2004/Dbs/DB1_B"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".tif", ".tiff")):
        input_path = os.path.join(input_folder, filename)
        output_filename = os.path.splitext(filename)[0] + ".bmp"
        output_path = os.path.join(output_folder, output_filename)

        try:
            with Image.open(input_path) as img:
                img.convert("RGB").save(output_path, "BMP")
            print(f"Converted: {filename}")
        except Exception as e:
            print(f"Error converting {filename}: {e}")

print("Batch conversion completed!")
