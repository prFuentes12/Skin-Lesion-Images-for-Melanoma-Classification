import kagglehub
import shutil
import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# Download the latest version of the dataset
path = kagglehub.dataset_download("andrewmvd/isic-2019")
print("Path to dataset files:", path)

# Set destination folder
dest_path = "../dataset"
os.makedirs(dest_path, exist_ok=True)

# Move all files and folders from the downloaded path to the destination
for item in os.listdir(path):
    src = os.path.join(path, item)
    dst = os.path.join(dest_path, item)
    shutil.move(src, dst)

print("All content moved to:", dest_path)


# ----------------------------------------------------------
# CSV's analysis
# ----------------------------------------------------------

# Load metadata and ground truth CSVs
metadata_path = "../dataset/ISIC_2019_Training_Metadata.csv"
labels_path = "../dataset/ISIC_2019_Training_GroundTruth.csv"

# Read both CSVs
metadata_df = pd.read_csv(metadata_path)
labels_df = pd.read_csv(labels_path)

# Add .jpg extension to match image filenames
metadata_df['image'] = metadata_df['image'].astype(str) + '.jpg'
labels_df['image'] = labels_df['image'].astype(str) + '.jpg'

# Identify the diagnosis by taking the column with value 1
diagnosis_cols = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
labels_df['diagnosis'] = labels_df[diagnosis_cols].idxmax(axis=1)

# Merge on 'image' column
combined_df = pd.merge(metadata_df, labels_df, on='image')

# Save to new CSV
combined_df.to_csv("../dataset/combined.csv", index=False)

# Show sample
print("Combined DataFrame (sample):")
print(combined_df.head())



# ----------------------------------------------------------
# EDA
# ----------------------------------------------------------

# Load combined CSV
df = pd.read_csv("../dataset/combined.csv")

# Set plot style
sns.set(style="whitegrid")

# -----------------------------
# 1. Check for missing values
# -----------------------------
print("\nMissing values per column:")
print(df.isnull().sum())

# -----------------------------
# 2. Count of each diagnosis
# -----------------------------
diagnosis_counts = df['diagnosis'].value_counts()
print("\nNumber of images per diagnosis:")
print(diagnosis_counts)

# Bar plot: count per diagnosis
plt.figure(figsize=(10, 5))
sns.barplot(x=diagnosis_counts.index, y=diagnosis_counts.values)
plt.title("Number of Images per Diagnosis")
plt.xlabel("Diagnosis")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------
# 3. Age distribution by diagnosis
# -----------------------------
plt.figure(figsize=(12, 6))
sns.boxplot(x='diagnosis', y='age_approx', data=df)
plt.title("Age Distribution by Diagnosis")
plt.xlabel("Diagnosis")
plt.ylabel("Approximate Age")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------
# 4. Sex distribution by diagnosis
# -----------------------------
plt.figure(figsize=(12, 6))
sns.countplot(x='diagnosis', hue='sex', data=df)
plt.title("Sex Distribution per Diagnosis")
plt.xlabel("Diagnosis")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------
# 5. Anatomical site distribution
# -----------------------------
plt.figure(figsize=(12, 6))
sns.countplot(y='anatom_site_general', hue='diagnosis', data=df, order=df['anatom_site_general'].value_counts().index)
plt.title("Anatomical Site vs. Diagnosis")
plt.xlabel("Count")
plt.ylabel("Anatomical Site")
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# Image resolution analysis
# ----------------------------------------------------------

# # Path to the folder containing the images
# image_folder = "..\dataset\ISIC_2019_Training_Input"

# # Verify the path
# print("Absolute path to image folder:", os.path.abspath(image_folder))

# if not os.path.exists(image_folder):
#     print("Image folder does not exist:", image_folder)
#     exit()

# # Display the first few files to confirm content
# print("First files in folder:", os.listdir(image_folder)[:10])

# # Dictionary to store unique resolutions and their counts
# resolutions = {}
# valid_extensions = ('.jpg', '.jpeg', '.png')

# # Iterate through the files in the folder
# for img_name in os.listdir(image_folder):
#     if not img_name.lower().endswith(valid_extensions):
#         continue  # Skip non-image files

#     img_path = os.path.join(image_folder, img_name)

#     try:
#         with Image.open(img_path) as img:
#             width, height = img.size
#             res = (height, width)  # Store as (height, width)
#             resolutions[res] = resolutions.get(res, 0) + 1
#     except Exception as e:
#         print(f"Error opening {img_name}: {e}")

# # Display the unique resolutions and how many images match each
# if resolutions:
#     print("\nUnique resolutions found (Height x Width) and number of images:")
#     for res, count in resolutions.items():
#         print(f"{res}: {count} images")
# else:
#     print("No valid image files found.")


