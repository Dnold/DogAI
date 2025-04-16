import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import os

# Configuration
IMG_SIZE = 250
MODEL_PATH = 'models/chihuahua_muffin_modelV3.h5'
TEST_IMG_PATH = 'dataset/muffin/muffin_191.JPG'
SAVE_DIR = 'visualizations'
os.makedirs(SAVE_DIR, exist_ok=True)

# 1. Network Architecture Diagram
def plot_model_architecture():
    print("[INFO] Generating model architecture diagram...")
    try:
        plot_model(model, to_file=os.path.join(SAVE_DIR, 'model_architecture.png'),
                   show_shapes=True,
                   show_layer_names=True,
                   rankdir='TB',
                   expand_nested=True,
                   dpi=96)
        print("[SUCCESS] Saved model_architecture.png")
    except ImportError as e:
        print(f"[WARNING] Could not generate architecture diagram: {e}")
        print("Install required packages: pip install pydot graphviz")

def visualize_feature_maps():
    print("[INFO] Plotting feature maps...")

    # Load and preprocess test image
    img = cv2.imread(TEST_IMG_PATH)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # First, we need to make sure the model is built by running a prediction
    # This ensures the model's input is defined
    _ = model.predict(np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype='float32'))
    
    # Find the first conv layer
    layer_name = [layer.name for layer in model.layers if 'conv' in layer.name][0]
    
    # Now create the visualization model using the functional API
    from tensorflow.keras.models import Model
    try:
        # Method 1: Standard way
        intermediate_model = Model(inputs=model.input, 
                                  outputs=model.get_layer(layer_name).output)
    except AttributeError:
        # Method 2: Alternative for newer Keras versions
        # This creates a new model that directly accesses the layer we want
        intermediate_model = Model(
            inputs=[model.layers[0].input],
            outputs=[model.get_layer(layer_name).output]
        )

    # Get feature maps
    feature_maps = intermediate_model.predict(img)

    # Plot first 16 feature maps
    num_filters = feature_maps.shape[-1]
    plt.figure(figsize=(15, 15))
    for i in range(min(num_filters, 16)):
        plt.subplot(4, 4, i + 1)
        plt.imshow(feature_maps[0, :, :, i], cmap='viridis')
        plt.axis('off')
        plt.title(f'Filter {i+1}')
    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, 'feature_maps_first_conv.png')
    plt.savefig(save_path)
    print(f"[SUCCESS] Saved '{save_path}'")
    plt.close()
# 3. Confusion Matrix
def plot_confusion_matrix():
    print("\n[INFO] Generating confusion matrix...")
    
    # Reload and preprocess data (same as training)
    def load_data():
        data, labels = [], []
        for label in ["chihuahua", "muffin"]:
            folder = f"dataset/{label}"
            for filename in os.listdir(folder):
                path = os.path.join(folder, filename)
                img = cv2.imread(path)
                if img is None: continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                data.append(img.astype('float32') / 255.0)
                labels.append(0 if label == "chihuahua" else 1)
        return np.array(data), np.array(labels)

    # Load and split data
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype("int32")

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                 display_labels=['Chihuahua', 'Muffin'])
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    plt.title('Model Confusion Matrix')
    plt.savefig(os.path.join(SAVE_DIR, 'confusion_matrix.png'), bbox_inches='tight')
    print("[SUCCESS] Saved confusion_matrix.png")
    plt.close()

def save_model_for_netron():
    print("[INFO] Saving model in formats compatible with Netron...")
    
    # Save in TensorFlow SavedModel format
    saved_model_path = os.path.join(SAVE_DIR, 'saved_model')
    model.save(saved_model_path)
    print(f"[SUCCESS] Saved model in TensorFlow SavedModel format to {saved_model_path}")
    
    # Save in ONNX format (if onnx is installed)
    try:
        import tf2onnx
        import onnx
        
        # Convert to ONNX
        onnx_path = os.path.join(SAVE_DIR, 'model.onnx')
        
        # Create a concrete function
        spec = tf.TensorSpec((None, IMG_SIZE, IMG_SIZE, 3), tf.float32)
        onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=[spec], opset=13)
        
        # Save the ONNX model
        onnx.save(onnx_model, onnx_path)
        print(f"[SUCCESS] Saved model in ONNX format to {onnx_path}")
        print("[INFO] You can visualize this model using Netron: https://netron.app/")
        
    except ImportError:
        print("[WARNING] Could not save ONNX model. To enable ONNX export, install: pip install tf2onnx onnx")

if __name__ == "__main__":
    # Load model
    model = load_model(MODEL_PATH)
    
    # Generate all visualizations
    plot_model_architecture()
    visualize_feature_maps()
    plot_confusion_matrix()
    save_model_for_netron
    print("\n[COMPLETE] All visualizations saved to", SAVE_DIR)