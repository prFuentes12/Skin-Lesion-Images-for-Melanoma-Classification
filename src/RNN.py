import kagglehub
import shutil
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator



# ----------------------------------------------------------
# Download and move dataset
# ----------------------------------------------------------

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
# Load and merge CSVs
# ----------------------------------------------------------

# Load metadata and ground truth CSVs
metadata_path = os.path.join(dest_path, "ISIC_2019_Training_Metadata.csv")
labels_path = os.path.join(dest_path, "ISIC_2019_Training_GroundTruth.csv")

metadata_df = pd.read_csv(metadata_path)
labels_df = pd.read_csv(labels_path)

# Add .jpg extension to match image filenames
metadata_df['image'] = metadata_df['image'].astype(str) + '.jpg'
labels_df['image'] = labels_df['image'].astype(str) + '.jpg'

# Identify the diagnosis column with value 1
diagnosis_cols = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
labels_df['diagnosis'] = labels_df[diagnosis_cols].idxmax(axis=1)

# Merge metadata and labels
combined_df = pd.merge(metadata_df, labels_df, on='image')
combined_df.to_csv(os.path.join(dest_path, "combined.csv"), index=False)

print("Combined DataFrame (sample):")
print(combined_df.head())



# ----------------------------------------------------------
# EDA (Exploratory Data Analysis)
# ----------------------------------------------------------

# Load the combined CSV
csv_path = os.path.join(dest_path, "combined.csv")
df = pd.read_csv(csv_path)

# Set plot style
sns.set_theme(style="whitegrid")

# 1. Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# 2. Count of each diagnosis
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

# 3. Age distribution by diagnosis
plt.figure(figsize=(12, 6))
sns.boxplot(x='diagnosis', y='age_approx', data=df)
plt.title("Age Distribution by Diagnosis")
plt.xlabel("Diagnosis")
plt.ylabel("Approximate Age")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4. Sex distribution by diagnosis
plt.figure(figsize=(12, 6))
sns.countplot(x='diagnosis', hue='sex', data=df)
plt.title("Sex Distribution per Diagnosis")
plt.xlabel("Diagnosis")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 5. Anatomical site distribution
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



# ----------------------------------------------------------
# Images Resampling
# ----------------------------------------------------------

# # Output folder for resized images
# output_folder = "../dataset/resized_512"
# os.makedirs(output_folder, exist_ok=True)

# # Desired size
# target_size = (512, 512)

# # Image extensions to include
# valid_extensions = ('.jpg', '.jpeg', '.png')

# # Process each image
# count = 0
# for filename in os.listdir(image_folder):
#     if not filename.lower().endswith(valid_extensions):
#         continue

#     input_path = os.path.join(image_folder, filename)
#     output_path = os.path.join(output_folder, filename)

#     try:
#         with Image.open(input_path) as img:
#             # Resize and save
#             resized = img.resize(target_size, Image.Resampling.LANCZOS)
#             resized.save(output_path)
#             count += 1
#     except Exception as e:
#         print(f"Failed to process {filename}: {e}")

# print(f"\n {count} images resized and saved to '{output_folder}'")

# ----------------------------------------------------------
# Count images per diagnosis in the resized folder
# ----------------------------------------------------------

resized_folder = "../dataset/resized_512"

# Train-validation split
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['diagnosis'], random_state=42)

# ImageDataGenerators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    vertical_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Load images from dataframe
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=resized_folder,
    x_col='image',
    y_col='diagnosis',
    target_size=(512, 512),
    batch_size=32,
    class_mode='categorical'
)
val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=resized_folder,
    x_col='image',
    y_col='diagnosis',
    target_size=(512, 512),
    batch_size=32,
    class_mode='categorical'
)

num_classes = len(train_generator.class_indices)

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(512, 512, 3)), 
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),  # Required before Dense
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

tensorboard_callback = TensorBoard(log_dir="logs/exp", histogram_freq=1)

# Model training
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=[tensorboard_callback]
)


