import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set image size
IMG_SIZE = 250

# Function to load and preprocess images
def load_and_preprocess_image(img_path):
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARNING] Could not load image: {img_path}")
            return None
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception as e:
        print(f"[ERROR] Error processing image {img_path}: {e}")
        return None

# Load data without augmentation (we'll use ImageDataGenerator for that)
print("[INFO] Loading and preprocessing images...")
data = []
labels = []

# Track counts for each class
chihuahua_count = 0
muffin_count = 0

for label in ["chihuahua", "muffin"]:
    folder = f"dataset/{label}"
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        features = load_and_preprocess_image(path)
        if features is None:
            continue
        
        data.append(features)
        labels.append(label)
        
        # Update counts
        if label == "chihuahua":
            chihuahua_count += 1
        else:
            muffin_count += 1

print(f"[INFO] Loaded {chihuahua_count} chihuahua images and {muffin_count} muffin images")

# Convert to numpy arrays
X = np.array(data, dtype='float32')
y = np.array(labels)

# Normalize images
X = X / 255.0

# Convert labels to numeric (Chihuahua = 0, Muffin = 1)
y = np.where(y == "chihuahua", 0, 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"[INFO] Training set: {X_train.shape[0]} images")
print(f"[INFO] Test set: {X_test.shape[0]} images")

# Create ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# CNN model
print("[INFO] Building model...")
model = Sequential([
    # First Convolutional Block 32 Filters, 3x3 Kernel, Leaky ReLU Activation
    # Input shape is (IMG_SIZE, IMG_SIZE, 3) for RGB images
    # Padding is 'same' to keep the output size same as input size
    Conv2D(32, (3, 3), activation='leaky_relu', input_shape=(IMG_SIZE, IMG_SIZE, 3), padding='same'),
    MaxPooling2D(pool_size=(2, 2)), # Max pooling to reduce spatial dimensions (2x2 pooling)
    
    # Second Convolutional Block 64 Filters, 4x4 Kernel, Leaky ReLU Activation
    # Padding is 'same' to keep the output size same as input size
    Conv2D(64, (4, 4), activation='leaky_relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)), # Max pooling to reduce spatial dimensions (2x2 pooling)
    
    # Third Convolutional Block 128 Filters, 5x5 Kernel, Leaky ReLU Activation
    # Padding is 'same' to keep the output size same as input size
    Conv2D(128, (5, 5), activation='leaky_relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)), # Max pooling to reduce spatial dimensions (2x2 pooling)
    
    # Flatten layer to convert 2D feature maps to 1D feature vectors
    Flatten(),
    
    # Fully connected layers with Dropout for regularization
    Dense(128, activation='leaky_relu'), # Fully connected layer with 128 neurons and Leaky ReLU activation (to avoid dead neurons)
    Dropout(0.5),  # Dropout to prevent overfitting
    Dense(1, activation='sigmoid')  # Binary classification output (0 or 1)
])

# Compile model with Adam optimizer and binary crossentropy loss function (suitable for binary classification)
model.compile(
    optimizer='adam', # Adam optimizer for adaptive learning rate (good for most cases)
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Display model summary
model.summary()

# Calculate proper batch size and steps per epoch
batch_size = 32


# Fit model with data augmentation using flow()
print("[INFO] Training model with data augmentation...")
# Note: when using flow(), we don't need to call fit() on the datagen
train_generator = datagen.flow(X_train, y_train, batch_size=batch_size)
history = model.fit(
    train_generator,
    # No need to specify steps_per_epoch when using fit_generator - it will use the entire dataset
    epochs=15,
    validation_data=(X_test, y_test),
    verbose=1, # Verbose output during training
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
)

# Evaluate model
print("[INFO] Evaluating model...")
score = model.evaluate(X_test, y_test, verbose=1) # Evaluate on test set
print(f"Test accuracy: {score[1]:.4f}")

# Generate predictions
print("[INFO] Generating predictions...")
y_pred_proba = model.predict(X_test) # Get predicted probabilities
y_pred = (y_pred_proba > 0.5).astype("int32")

# Print classification report
print("\n📋 Classification Report:\n")
print(classification_report(y_test, y_pred))

# Test sample images
print("[INFO] Testing sample images...")
test_images = {
    "Chihuahua": "dataset/chihuahua/chihuahua_25.JPG",
    "Muffin": "dataset/muffin/img_0_74.jpg"
}

for name, path in test_images.items():
    # Load and preprocess the image
    img = load_and_preprocess_image(path)
    
    if img is not None:
        # Convert to float and normalize
        img = img.astype('float32') / 255.0
        
        # Make prediction
        prediction = model.predict(np.expand_dims(img, axis=0)) # Predict on the image
        
        # Determine class and confidence
        pred_class = "Chihuahua" if prediction < 0.5 else "Muffin"
        confidence = 1 - prediction[0][0] if prediction < 0.5 else prediction[0][0]
        
        print(f"→ Prediction for '{name}': {pred_class} (confidence: {confidence*100:.2f}%)")

# Save model
print("[INFO] Saving model...")
model.save('models/chihuahua_muffin_modelV3.h5')
print("[SUCCESS] Model saved!")


# Save training history plot
print("[INFO] Saving training history plot...")

try:
    import matplotlib.pyplot as plt
    
    # Plot accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'])
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])
    
    # Save plot
    plt.tight_layout()
    plt.savefig('models/training_history.png')
    print("[INFO] Training history plot saved to 'models/training_history.png'")
except:
    print("[WARNING] Could not create training history plot")