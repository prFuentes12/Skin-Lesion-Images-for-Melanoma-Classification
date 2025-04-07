import kagglehub
import shutil
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
import numpy as np

# ----------------------------------------------------------
# Download and move dataset
# ----------------------------------------------------------

# Download the latest version of the dataset
path = kagglehub.dataset_download("andrewmvd/isic-2019")
print("Path to dataset files:", path)

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
# Resize Images
# ----------------------------------------------------------

# image_folder = os.path.join(dest_path, "ISIC_2019_Training_Input")
# output_folder = os.path.join(dest_path, "resized_255")
# os.makedirs(output_folder, exist_ok=True)

# target_size = (255, 255)
# valid_extensions = ('.jpg', '.jpeg', '.png')
# count = 0

# for filename in os.listdir(image_folder):
#     if not filename.lower().endswith(valid_extensions):
#         continue
#     input_path = os.path.join(image_folder, filename)
#     output_path = os.path.join(output_folder, filename)
#     try:
#         with Image.open(input_path) as img:
#             resized = img.resize(target_size, Image.Resampling.LANCZOS)
#             resized.save(output_path)
#             count += 1
#     except Exception as e:
#         print(f"Failed to process {filename}: {e}")

# print(f"\n{count} images resized and saved to '{output_folder}'")

# ----------------------------------------------------------
# Preprocessing & Class Weight
# ----------------------------------------------------------

resized_folder = "../dataset/resized_255"


# Train-validation split
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['diagnosis'], random_state=42)

le = LabelEncoder()
train_df['diagnosis_encoded'] = le.fit_transform(train_df['diagnosis'])

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df['diagnosis_encoded']),
    y=train_df['diagnosis_encoded']
)
class_weights_dict = {label: weight for label, weight in zip(np.unique(train_df['diagnosis_encoded']), class_weights)}

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    brightness_range=[0.85, 1.15],
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
    target_size=(255, 255),
    batch_size=32,
    class_mode='categorical'
)
val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=resized_folder,
    x_col='image',
    y_col='diagnosis',
    target_size=(255, 255),
    batch_size=32,
    class_mode='categorical'
)

# ----------------------------------------------------------
# Model Definition
# ----------------------------------------------------------

num_classes = len(train_generator.class_indices)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(255, 255, 3)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ----------------------------------------------------------
# Train Model with Class Weights
# ----------------------------------------------------------

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    class_weight=class_weights_dict
)

# ----------------------------------------------------------
# Plot Accuracy & Loss
# ----------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
