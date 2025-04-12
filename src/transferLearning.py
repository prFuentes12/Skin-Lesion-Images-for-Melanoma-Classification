import kagglehub
import shutil
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder, label_binarize
import numpy as np
import gc
from tensorflow.keras.applications import EfficientNetB5
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, regularizers, models
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# ----------------------------------------------------------
# Download and move dataset
# ----------------------------------------------------------
path = kagglehub.dataset_download("andrewmvd/isic-2019")
dest_path = "../dataset"
os.makedirs(dest_path, exist_ok=True)
for item in os.listdir(path):
    shutil.move(os.path.join(path, item), os.path.join(dest_path, item))

# ----------------------------------------------------------
# Load and prepare CSVs
# ----------------------------------------------------------
metadata = pd.read_csv(os.path.join(dest_path, "ISIC_2019_Training_Metadata.csv"))
labels = pd.read_csv(os.path.join(dest_path, "ISIC_2019_Training_GroundTruth.csv"))
metadata['image'] = metadata['image'].astype(str) + '.jpg'
labels['image'] = labels['image'].astype(str) + '.jpg'
diagnosis_cols = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
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
# Parameters and preprocessing
# ----------------------------------------------------------
img_size = 512
batch_size = 16
epochs = 20
labels_list = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']

df = df[df['diagnosis'].isin(labels_list)].reset_index(drop=True)
label_encoder = LabelEncoder()
df['diagnosis_encoded'] = label_encoder.fit_transform(df['diagnosis'])

# Safe float conversion
y_labels = list(map(int, df['diagnosis_encoded'].values.tolist()))
unique_classes = np.unique(y_labels)
weights_array = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_labels)
class_weights_dict = {
    int(cls): float(w.numpy()) if hasattr(w, 'numpy') else float(w)
    for cls, w in zip(unique_classes, weights_array)
}

# Image paths
image_folder = os.path.join(dest_path, "ISIC_2019_Training_Input")
df['image_path'] = df['image'].apply(lambda x: os.path.join(image_folder, x))

# Split
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['diagnosis_encoded'], random_state=42)

# ----------------------------------------------------------
# TF dataset
# ----------------------------------------------------------
def preprocess_image(file_path, label):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_size, img_size])
    img = preprocess_input(img)
    return img, label

def preprocess_and_augment(file_path, label):
    img, label = preprocess_image(file_path, label)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    return img, label

train_dataset = tf.data.Dataset.from_tensor_slices((train_df['image_path'].values, train_df['diagnosis_encoded'].values))
train_dataset = train_dataset.map(preprocess_and_augment, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((val_df['image_path'].values, val_df['diagnosis_encoded'].values))
val_dataset = val_dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# ----------------------------------------------------------
# Model
# ----------------------------------------------------------
base_model = EfficientNetB5(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))
base_model.trainable = True
for layer in base_model.layers[:300]:
    layer.trainable = False

inputs = tf.keras.Input(shape=(img_size, img_size, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.005))(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(len(label_encoder.classes_), activation="softmax")(x)
model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.0005, momentum=0.9),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ----------------------------------------------------------
# Training (NO CALLBACKS)
# ----------------------------------------------------------
tf.keras.backend.clear_session()
gc.collect()

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    class_weight=class_weights_dict  # safe now
)

# ----------------------------------------------------------
# Simple Plot
# ----------------------------------------------------------
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.grid(True)
plt.title("Accuracy")
plt.show()
