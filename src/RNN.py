import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ------------------------------
# Load and merge CSVs
# ------------------------------
dest_path = "../dataset"
metadata_path = os.path.join(dest_path, "ISIC_2019_Training_Metadata.csv")
labels_path = os.path.join(dest_path, "ISIC_2019_Training_GroundTruth.csv")

metadata_df = pd.read_csv(metadata_path)
labels_df = pd.read_csv(labels_path)

metadata_df['image'] = metadata_df['image'].astype(str) + '.jpg'
labels_df['image'] = labels_df['image'].astype(str) + '.jpg'

diagnosis_cols = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
labels_df['diagnosis'] = labels_df[diagnosis_cols].idxmax(axis=1)

combined_df = pd.merge(metadata_df, labels_df, on='image')
combined_df.to_csv(os.path.join(dest_path, "combined.csv"), index=False)

# ------------------------------
# Exploratory Data Analysis (EDA)
# ------------------------------
csv_path = os.path.join(dest_path, "combined.csv")
df = pd.read_csv(csv_path)
sns.set_theme(style="whitegrid")

# Check missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Count of each diagnosis
diagnosis_counts = df['diagnosis'].value_counts()
print("\nNumber of images per diagnosis:")
print(diagnosis_counts)

# Bar plot of diagnosis distribution
plt.figure(figsize=(10, 5))
sns.barplot(x=diagnosis_counts.index, y=diagnosis_counts.values)
plt.title("Number of Images per Diagnosis")
plt.xlabel("Diagnosis")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Age distribution by diagnosis
plt.figure(figsize=(12, 6))
sns.boxplot(x='diagnosis', y='age_approx', data=df)
plt.title("Age Distribution by Diagnosis")
plt.xlabel("Diagnosis")
plt.ylabel("Approximate Age")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Sex distribution by diagnosis
plt.figure(figsize=(12, 6))
sns.countplot(x='diagnosis', hue='sex', data=df)
plt.title("Sex Distribution per Diagnosis")
plt.xlabel("Diagnosis")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Anatomical site distribution
plt.figure(figsize=(12, 6))
sns.countplot(y='anatom_site_general', hue='diagnosis', data=df,
              order=df['anatom_site_general'].value_counts().index)
plt.title("Anatomical Site vs. Diagnosis")
plt.xlabel("Count")
plt.ylabel("Anatomical Site")
plt.tight_layout()
plt.show()

# ------------------------------
# Data preparation
# ------------------------------
resized_folder = "../dataset/resized_512"
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['diagnosis'], random_state=42)

# Image augmentation for training
data_augmentation = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.85, 1.15],
    fill_mode='nearest'
)
val_generator = ImageDataGenerator(rescale=1./255)

# Data generators
train_generator = data_augmentation.flow_from_dataframe(
    dataframe=train_df,
    directory=resized_folder,
    x_col='image',
    y_col='diagnosis',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
val_generator = val_generator.flow_from_dataframe(
    dataframe=val_df,
    directory=resized_folder,
    x_col='image',
    y_col='diagnosis',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Compute class weights to handle class imbalance
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df['diagnosis']),
    y=train_df['diagnosis']
)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

# ------------------------------
# Build CNN Model from scratch
# ------------------------------
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(train_generator.class_indices), activation='softmax')
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Model training
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15,
    class_weight=class_weights_dict
)

# ------------------------------
# Evaluate model performance
# ------------------------------
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()