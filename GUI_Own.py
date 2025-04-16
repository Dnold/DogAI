import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Reduce TensorFlow logging
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk

# ===== CONFIGURATION =====
TERMINAL_THEME = {
    "bg": "#121212",
    "fg": "#00FF00",
    "accent": "#1F1F1F",
    "font": ("Consolas", 12),
    "title_font": ("Consolas", 16, "bold"),
}

print("[BOOTING UP] Initializing visual cortex...")

# Configure GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[OK] {len(gpus)} GPU(s) configured")
    except RuntimeError as e:
        print(f"[WARNING] GPU configuration failed: {e}")

# ===== MODEL LOADING =====
def load_model_safely(model_path):
    try:
        model = load_model(model_path, compile=False)
        print(f"[OK] Model loaded from {model_path}")
        # Try a dummy forward pass
        dummy_input = np.zeros((1, 250, 250, 3), dtype=np.float32)
        model.predict(dummy_input)
        print("[OK] Model forward pass successful")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        print("[INFO] Creating a new model instead")
        return create_new_model()


def create_new_model():
    """Create a new model with the expected architecture"""
    IMG_SIZE = 250
    model = tf.keras.Sequential([
        Conv2D(32, (3, 3), activation='leaky_relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(64, (3, 3), activation='leaky_relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(128, (3, 3), activation='leaky_relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Flatten(),
        Dense(128, activation='leaky_relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load or create model
model_path = 'models/chihuahua_muffin_modelV3.h5'
model = load_model_safely(model_path)

# Initialize model with dummy input
try:
    model(np.zeros((1, 250, 250, 3)))
    print("[OK] Model initialized successfully")
except Exception as e:
    print(f"[WARNING] Model initialization failed: {e}")

# ===== IMAGE PROCESSING =====
IMG_SIZE = 250

def extract_features(img_path):
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype('float32') / 255.0
    except Exception as e:
        print(f"[ERROR] Image processing failed: {e}")
        return None

# ===== FEATURE VISUALIZATION =====
def get_feature_maps(img, model):
    try:
        # Get all convolutional layers
        conv_layers = [layer for layer in model.layers 
                      if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.SeparableConv2D))]
        
        # Create a new model that outputs feature maps
        layer_outputs = [layer.output for layer in conv_layers]
        feature_model = Model(inputs=model.inputs, outputs=layer_outputs)
        
        # Process image and get features
        img_array = np.expand_dims(img, axis=0)
        feature_maps = feature_model.predict(img_array)
        
        # Prepare visualizations
        visualizations = []
        for i, fmap in enumerate(feature_maps):
            # Normalize and resize feature maps
            n_filters = fmap.shape[-1]
            size = fmap.shape[1]
            
            # Create 4x4 grid of first 16 filters
            grid_size = 4
            display_grid = np.zeros((size * grid_size, size * grid_size))
            for j in range(min(grid_size * grid_size, n_filters)):
                x, y = j // grid_size, j % grid_size
                channel_image = fmap[0, :, :, j]
                channel_image = cv2.normalize(channel_image, None, 0, 255, cv2.NORM_MINMAX)
                display_grid[x*size:(x+1)*size, y*size:(y+1)*size] = channel_image
            
            # Resize for display
            display_image = cv2.resize(display_grid, (400, 400), interpolation=cv2.INTER_NEAREST)
            visualizations.append(display_image.astype('uint8'))
            
        layer_info = [f"Layer {i+1}\n{fmap.shape[-1]} filters" for i, fmap in enumerate(feature_maps)]
        return visualizations, layer_info
        
    except Exception as e:
        print(f"[ERROR] Feature extraction failed: {e}")
        return None, None

# ===== GUI IMPLEMENTATION =====
class SnackClassifierApp:
    def __init__(self, master):
        self.master = master
        self.setup_gui()
        
    def setup_gui(self):
        self.master.title("Snack Classifier v3.0")
        self.master.geometry("1000x800")
        self.master.configure(bg=TERMINAL_THEME["bg"])
        
        # Main frame
        main_frame = tk.Frame(self.master, bg=TERMINAL_THEME["bg"])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header
        header = tk.Label(
            main_frame,
            text="🐾 Chihuahua or Muffin? 🧁\nAdvanced Snack Classifier",
            bg=TERMINAL_THEME["bg"],
            fg=TERMINAL_THEME["fg"],
            font=TERMINAL_THEME["title_font"])
        header.pack(pady=(0, 20))
        
        # Image display
        self.image_frame = tk.Frame(main_frame, bg=TERMINAL_THEME["accent"], padx=5, pady=5)
        self.image_frame.pack()
        
        self.image_label = tk.Label(
            self.image_frame, 
            text="[ No image selected ]",
            bg=TERMINAL_THEME["bg"],
            fg=TERMINAL_THEME["fg"])
        self.image_label.pack()
        
        # Controls
        control_frame = tk.Frame(main_frame, bg=TERMINAL_THEME["bg"])
        control_frame.pack(pady=10)
        
        self.style = ttk.Style()
        self.style.configure("Terminal.TButton", 
                           font=TERMINAL_THEME["font"],
                           background="#00FF00",
                           foreground="black",
                           padding=10)
        
        ttk.Button(
            control_frame, 
            text="> Load Image", 
            command=self.load_image,
            style="Terminal.TButton").pack(side=tk.LEFT, padx=5)
            
        ttk.Button(
            control_frame,
            text="> Clear",
            command=self.clear_display,
            style="Terminal.TButton").pack(side=tk.LEFT, padx=5)
        
        # Results
        self.result_var = tk.StringVar()
        self.result_label = tk.Label(
            main_frame,
            textvariable=self.result_var,
            bg=TERMINAL_THEME["bg"],
            fg="#FF5555",
            font=TERMINAL_THEME["title_font"])
        self.result_label.pack()
        
        self.confidence_var = tk.StringVar()
        self.confidence_label = tk.Label(
            main_frame,
            textvariable=self.confidence_var,
            bg=TERMINAL_THEME["bg"],
            fg=TERMINAL_THEME["fg"],
            font=TERMINAL_THEME["font"])
        self.confidence_label.pack()
        
        # Feature maps with scrollbar
        canvas_frame = tk.Frame(main_frame, bg=TERMINAL_THEME["bg"])
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, bg=TERMINAL_THEME["bg"], highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        
        self.feature_frame = tk.Frame(self.canvas, bg=TERMINAL_THEME["bg"])
        self.feature_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.feature_frame, anchor=tk.NW)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.feature_labels = []
        self.layer_labels = []
    
    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not path:
            return
            
        self.clear_display()
        self.result_var.set("[Processing...]")
        self.master.update()
        
        try:
            # Display input image
            img = Image.open(path)
            img.thumbnail((300, 300))
            self.tk_img = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.tk_img)
            
            # Process and classify
            processed_img = extract_features(path)
            if processed_img is None:
                raise ValueError("Image processing failed")
                
            prediction = model.predict(np.expand_dims(processed_img, axis=0))[0][0]
            result = "Muffin 🧁" if prediction >= 0.5 else "Chihuahua 🐕"
            confidence = f"Confidence: {max(prediction, 1-prediction)*100:.1f}%"
            
            self.result_var.set(result)
            self.confidence_var.set(confidence)
            
            # Show feature maps
            feature_maps, layer_info = get_feature_maps(processed_img, model)
            if feature_maps:
                self.show_feature_maps(feature_maps, layer_info)
                
        except Exception as e:
            self.result_var.set("ERROR")
            self.confidence_var.set(str(e))
            print(f"[ERROR] {e}")
    
    def show_feature_maps(self, feature_maps, layer_info):
        for i, (fmap, info) in enumerate(zip(feature_maps, layer_info)):
            # Layer info
            lbl = tk.Label(
                self.feature_frame,
                text=info,
                bg=TERMINAL_THEME["bg"],
                fg=TERMINAL_THEME["fg"],
                font=TERMINAL_THEME["font"])
            lbl.grid(row=i*2, column=0, padx=5, pady=5, sticky=tk.W)
            self.layer_labels.append(lbl)
            
            # Feature map visualization
            img = Image.fromarray(fmap).convert("L")
            tk_img = ImageTk.PhotoImage(img)
            
            lbl = tk.Label(
                self.feature_frame,
                image=tk_img,
                bg=TERMINAL_THEME["bg"])
            lbl.image = tk_img
            lbl.grid(row=i*2+1, column=0, padx=5, pady=(0, 20), sticky=tk.N)  # Centered vertically
            self.feature_labels.append(lbl)
    
    def clear_display(self):
        self.result_var.set("")
        self.confidence_var.set("")
        self.image_label.config(image=None)
        self.image_label.config(text="[ No image selected ]")
        
        for widget in self.feature_labels + self.layer_labels:
            widget.destroy()
        
        self.feature_labels = []
        self.layer_labels = []

# Run application
if __name__ == "__main__":
    print("\n[SYSTEM] Starting Snack Classifier...")
    root = tk.Tk()
    app = SnackClassifierApp(root)
    root.mainloop()