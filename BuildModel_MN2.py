# Importieren der erforderlichen Bibliotheken
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

IMG_SIZE = 224  # Optimiert für MobileNetV2

def extract_features(img_path):
    """
    Lädt ein Bild, führt Resizing durch und konvertiert Color Channels.
    
    Args:
        img_path (str): Pfad zum Bildfile
    
    Returns:
        np.array: Preprocesssed Image oder None bei Fehler
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARNING] Bildladefehler: {img_path}")
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV lädt Bilder standardmäßig in BGR
    return img

# Daten laden und vorbereiten
data = []
labels = []

# Iteration durch die Klassenordner
for label in ["chihuahua", "muffin"]:
    folder_path = os.path.join("dataset", label)
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        features = extract_features(file_path)
        if features is not None:
            data.append(features)
            labels.append(label)

# Konvertierung in NumPy Arrays
X = np.array(data)
y = np.array([0 if label == "chihuahua" else 1 for label in labels])  # Numerisches Labeln

# MobileNetV2-spezifisches Preprocessing
# Die features werden in float32 konvertiert, um die Vorverarbeitung zu ermöglichen
# MobileNetV2 erwartet Bild Featuredaten im Format von -1 bis 1
X = tf.keras.applications.mobilenet_v2.preprocess_input(X.astype('float32')) # Normalisierung: (x / 127.5) - 1

# Train-Test-Split mit Stratifizierung
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# Class Weight Berechnung für unbalancierte Daten
class_weights = class_weight.compute_class_weight(
    'balanced', classes=np.unique(y_train), y=y_train) # Berechnung der Gewichtung für jede Klasse: GesamtMenge / 2 * Anzahl der Klasse
# Umwandlung in Dictionary für Keras
class_weights = dict(enumerate(class_weights))

# Data Augmentation Konfiguration
train_datagen = ImageDataGenerator(
    rotation_range=20,      # Zufällige Rotation (±20°) x' = x·cosθ - y·sinθ
    width_shift_range=0.2,  # Horizontale Verschiebung
    height_shift_range=0.2, # Vertikale Verschiebung x' = x + dx, y' = y + dy
    shear_range=0.2,        # Scherverzerrung
    zoom_range=0.2,         # Zufälliges Zoomen x' = x·sx, y' = y·sy
    horizontal_flip=True,   # Horizontales Spiegeln
    fill_mode='nearest'     # Füllmethode für leere Pixel
)

# Transfer Learning mit MobileNetV2
base_model = MobileNetV2(
    weights='imagenet',     # Pre-trained Gewichte
    include_top=False,      # Eigenes Top Layer wird hinzugefügt
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False  # Freeze Base Model

# Modellarchitektur
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),  # Reduziert räumliche Dimensionen
    Dense(256, activation='relu'),  # Fully Connected Layer
    Dropout(0.5),             # Regularisierung
    Dense(1, activation='sigmoid')  # Binary Classification Output
])

# Kompilierung mit angepasster Learning Rate
optimizer = Adam(learning_rate=1e-3)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Callbacks für Training
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),  # Vermeidet Overfitting
    ModelCheckpoint('models/best_model_MN2.h5', save_best_only=True)   # Speichert beste Modelweights
]

# Initiales Training
history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=32),
    epochs=10,
    validation_data=(X_test, y_test),
    class_weight=class_weights,
    callbacks=callbacks
)

# Evaluierung
y_pred = (model.predict(X_test) > 0.5).astype(int)
print(classification_report(y_test, y_pred))

# Fine-Tuning: Teilweises Unfreezen
base_model.trainable = True
for layer in base_model.layers[:100]:  # Erste 100 Layer bleiben gefrozen
    layer.trainable = False

# Rekompilierung mit kleinerer Learning Rate
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Feinabstimmung des Modells
history_fine = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=32),
    epochs=20,
    validation_data=(X_test, y_test),
    class_weight=class_weights,
    callbacks=callbacks
)

# Finale Evaluierung
model.evaluate(X_test, y_test)

# Modellspeicherung
model.save("chihuahua_muffin_modelMN2.h5")