import os
import shutil
import pandas as pd

# Paths
dataset1_images = "datasetss - Copy"
dataset2_images = "images"
dataset1_captions = "captions_architecture.csv"  # CSV format
dataset2_captions = "captions.txt"  # TXT format
combined_images = "combined_dataset"
combined_captions_file = "combined_dataset captionsss.csv"

os.makedirs(combined_images, exist_ok=True)

# --- Load CSV captions ---
try:
    df1 = pd.read_csv(dataset1_captions, encoding="utf-8")
except UnicodeDecodeError:
    df1 = pd.read_csv(dataset1_captions, encoding="cp1252")

# Rename CSV columns: 'image_name' -> 'filename'
df1 = df1.rename(columns={"image_name": "filename"})

# --- Load TXT captions ---
txt_data = []
with open(dataset2_captions, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if "|" in line:
            img, cap = line.split("|", 1)
        else:
            parts = line.split(" ", 1)
            img, cap = parts[0], parts[1] if len(parts) > 1 else ""
        txt_data.append({"filename": img, "caption": cap})

df2 = pd.DataFrame(txt_data)

# --- Copy dataset1 images ---
for img_file in os.listdir(dataset1_images):
    shutil.copy(os.path.join(dataset1_images, img_file),
                os.path.join(combined_images, img_file))

# --- Copy dataset2 images with prefix ---
for img_file in os.listdir(dataset2_images):
    new_name = "ds2_" + img_file
    shutil.copy(os.path.join(dataset2_images, img_file),
                os.path.join(combined_images, new_name))
    # Update df2 filenames
    df2.loc[df2['filename'] == img_file, 'filename'] = new_name

# --- Combine captions ---
combined_df = pd.concat([df1, df2], ignore_index=True)

# --- Save combined captions ---
combined_df.to_csv(combined_captions_file, index=False, encoding="utf-8")

print("Datasets merged successfully!")
