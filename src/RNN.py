import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import kagglehub
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
import numpy as np
import gc
from tensorflow.keras.regularizers import l2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# ----------------------------------------------------------
# Download and move dataset
# ----------------------------------------------------------
path = kagglehub.dataset_download("andrewmvd/isic-2019")
dest_path = "../dataset"
os.makedirs(dest_path, exist_ok=True)
for item in os.listdir(path):
    shutil.move(os.path.join(path, item), os.path.join(dest_path, item))

# ----------------------------------------------------------
# Load and merge CSVs
# ----------------------------------------------------------
metadata = pd.read_csv(os.path.join(dest_path, "ISIC_2019_Training_Metadata.csv"))
labels = pd.read_csv(os.path.join(dest_path, "ISIC_2019_Training_GroundTruth.csv"))
metadata['image'] = metadata['image'].astype(str) + '.jpg'
labels['image'] = labels['image'].astype(str) + '.jpg'
diagnosis_cols = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
labels['diagnosis'] = labels[diagnosis_cols].idxmax(axis=1)
df = pd.merge(metadata, labels, on='image')

# ----------------------------------------------------------
# EDA (Exploratory Data Analysis)
# ----------------------------------------------------------
sns.set_theme(style="whitegrid")
print(df.isnull().sum())

plt.figure(figsize=(10, 5))
sns.countplot(x='diagnosis', data=df)
plt.title("Image count per diagnosis")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='diagnosis', y='age_approx', data=df)
plt.title("Age Distribution by Diagnosis")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(x='diagnosis', hue='sex', data=df)
plt.title("Sex Distribution by Diagnosis")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(y='anatom_site_general', hue='diagnosis', data=df, order=df['anatom_site_general'].value_counts().index)
plt.title("Anatomical Site vs. Diagnosis")
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# Preprocessing and Augmentation
# ----------------------------------------------------------
img_size = 500
batch_size = 32
label_encoder = LabelEncoder()
df['diagnosis_encoded'] = label_encoder.fit_transform(df['diagnosis'])

class_weights_array = compute_class_weight(class_weight='balanced', classes=np.unique(df['diagnosis_encoded']), y=df['diagnosis_encoded'])
class_weights_dict = {int(i): float(w) for i, w in enumerate(class_weights_array)}

image_folder = os.path.join(dest_path, "ISIC_2019_Training_Input")
df['image_path'] = df['image'].apply(lambda x: os.path.join(image_folder, x))

train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['diagnosis_encoded'], random_state=42)

# Image processing and augmentation
def preprocess_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_size, img_size])
    img = preprocess_input(img)
    return img, label

def preprocess_and_augment(path, label):
    img, label = preprocess_image(path, label)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    return img, label

train_dataset = tf.data.Dataset.from_tensor_slices((train_df['image_path'].values, train_df['diagnosis_encoded'].values))
train_dataset = train_dataset.map(preprocess_and_augment, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((val_df['image_path'].values, val_df['diagnosis_encoded'].values))
test_dataset = test_dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# ----------------------------------------------------------
# Model Definition from scratch
# ----------------------------------------------------------
tf.keras.backend.clear_session()
gc.collect()

inputs = tf.keras.Input(shape=(img_size, img_size, 3))
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling2D(2, 2)(x)
x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling2D(2, 2)(x)
x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling2D(2, 2)(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=l2(0.005))(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.0005, momentum=0.9),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ----------------------------------------------------------
# Callbacks and Training
# ----------------------------------------------------------
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("model_checkpoint.keras", monitor="val_accuracy", mode="max", save_best_only=True, verbose=1)
early_stop_cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=20,
    class_weight=class_weights_dict,
    callbacks=[checkpoint_cb, early_stop_cb]
)

# ----------------------------------------------------------
# Plot Training Curves
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

# ----------------------------------------------------------
# Confusion Matrix + Classification Report
# ----------------------------------------------------------
y_true = val_df['diagnosis_encoded'].values
y_pred_probs = model.predict(test_dataset)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_true, y_pred)
labels = label_encoder.classes_

fig, ax = plt.subplots(figsize=(10, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(ax=ax, xticks_rotation=45, cmap='Blues')
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=labels))