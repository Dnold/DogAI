import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# ===== CONFIGURATION =====
TERMINAL_THEME = {
    "bg": "#121212",
    "fg": "#00FF00",
    "accent": "#1F1F1F",
    "font": ("Consolas", 12),
    "title_font": ("Consolas", 16, "bold"),
    "warning_font": ("Consolas", 10)
}

# Load model with terminal-style loading message
print("[SYSTEM] Initializing snack recognition protocol...")
model = load_model('models/best_model.h5')
print("[SYSTEM] Muffin detection matrix ready!")

# ===== IMAGE PROCESSING =====
IMG_SIZE = 224

def extract_features(img_path):
    try:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        return img.astype('float32')
    except Exception as e:
        print(f"[ERROR] Image processing failed: {str(e)}")
        return None

# ===== FEATURE MAP VISUALIZATION =====
def get_feature_maps(img, model, layer_names):
    """
    Extrahiert Feature-Maps aus bestimmten Schichten
    Args:
        img: Vorverarbeitetes Eingabebild
        model: Hauptmodell
        layer_names: Liste von Layer-Namen aus dem Base-Model
    Returns:
        List of processed feature maps
    """
    # Erstelle Teilmodell für Feature-Map Extraktion
    base_model = model.layers[0]
    outputs = [base_model.get_layer(name).output for name in layer_names]
    feature_model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    
    # Berechne Feature-Maps
    feature_maps = feature_model.predict(np.expand_dims(img, axis=0))
    
    # Verarbeite jede Feature-Map
    processed = []
    for fm in feature_maps:
        # Reduziere auf 2D durch Kanal-Durchschnitt
        fm_2d = np.mean(fm[0], axis=-1)
        
        # Normalisierung für die Darstellung
        fm_norm = cv2.normalize(fm_2d, None, 0, 255, cv2.NORM_MINMAX)
        fm_resized = cv2.resize(fm_norm, (200, 200))  # Increased size to 200x200
        
        processed.append(fm_resized.astype('uint8'))
    
    return processed

# ===== AI CLASSIFICATION =====
def classify_image(img_path):
    img = extract_features(img_path)
    if img is not None:
        # Hauptvorhersage
        prediction = model.predict(np.expand_dims(img, axis=0))[0][0]
        
        # Feature-Maps von wichtigen Schichten
        layer_names = [
            'block_1_expand_relu',   # Frühe Kantenerkennung
            'block_3_expand_relu',   # Texturmuster
            'block_6_expand_relu'    # Objektteile
        ]
        feature_maps = get_feature_maps(img, model, layer_names)
        
        # Ergebnisformatierung
        if prediction >= 0.5:
            result = "Muffin 🧁 (You can probably eat that!)"
            confidence = prediction
        else:
            result = "Chihuahua 🐕 (Do NOT eat!)"
            confidence = 1 - prediction
            
        return result, f"Certainty: {max(confidence, 0.01)*100:.1f}%", feature_maps
    
    return "ERROR", "Failed to digest image", None

# ===== TERMINAL-STYLE GUI =====
class CyberPupApp:
    def __init__(self, master):
        self.master = master
        master.configure(bg=TERMINAL_THEME["bg"])
        
        # Main container
        self.frame = tk.Frame(master, bg=TERMINAL_THEME["bg"])
        self.frame.pack(padx=20, pady=20)
        
        # Header with funny text
        self.header = tk.Label(self.frame, 
            text="🐕¿Chihuahua or Muffin?🧁\n"
                 "v2.3.1 - Snack Identification System\n"
                 "WARNING: May confuse baked goods with small dogs",
            bg=TERMINAL_THEME["bg"],
            fg=TERMINAL_THEME["fg"],
            font=TERMINAL_THEME["title_font"])
        self.header.grid(row=0, column=0, pady=(0, 20))

        # Image display with terminal border
        self.img_frame = tk.Frame(self.frame, 
            bg=TERMINAL_THEME["accent"], 
            padx=5, pady=5)
        self.panel = tk.Label(self.img_frame, 
            bg=TERMINAL_THEME["bg"], 
            text="[Awaiting nutritional analysis...]")
        self.panel.pack()
        self.img_frame.grid(row=1, column=0, pady=10)

        # Stylish button
        self.btn = ttk.Button(self.frame, 
            text="SCAN EDIBLE ITEM", 
            command=self.open_file,
            style="Terminal.TButton")
        self.btn.grid(row=2, column=0, pady=10)

        # Results display
        self.result_text = tk.StringVar()
        self.result_label = tk.Label(self.frame,
            textvariable=self.result_text,
            bg=TERMINAL_THEME["bg"],
            fg=TERMINAL_THEME["fg"],
            font=TERMINAL_THEME["title_font"])
        self.result_label.grid(row=3, column=0, pady=(10, 5))

        self.confidence_text = tk.StringVar()
        self.confidence_label = tk.Label(self.frame,
            textvariable=self.confidence_text,
            bg=TERMINAL_THEME["bg"],
            fg=TERMINAL_THEME["fg"],
            font=TERMINAL_THEME["font"])
        self.confidence_label.grid(row=4, column=0)

        # Feature map display
        self.feature_frame = tk.Frame(self.frame, bg=TERMINAL_THEME["bg"])
        self.feature_frame.grid(row=5, column=0, pady=10)
        
        self.feature_labels = []
        self.layer_title_labels = []

        # Configure styles
        self.style = ttk.Style()
        self.style.configure("Terminal.TButton",
            font=TERMINAL_THEME["font"],
            foreground="black",
            background="#00FF00",
            bordercolor="#00FF00",
            relief="raised",
            padding=10)

    def open_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            try:
                # Update UI during processing
                self.result_text.set("[ANALYZING...]")
                self.confidence_text.set("Executing muffin detection protocol...")
                self.master.update()

                # Load and display image
                img = Image.open(file_path)
                img.thumbnail((400, 400))  # Increased thumbnail size to 400x400
                photo = ImageTk.PhotoImage(img)
                self.panel.config(image=photo)
                self.panel.image = photo

                # Get classification with feature maps
                result, confidence, feature_maps = classify_image(file_path)
                self.result_text.set(result)
                self.confidence_text.set(confidence)

                # Clear previous feature maps
                for label in self.feature_labels:
                    label.destroy()
                for label in self.layer_title_labels:
                    label.destroy()
                self.feature_labels = []
                self.layer_title_labels = []

                # Display new feature maps if available
                if feature_maps:
                    layer_names = ["Edge Detection", "Texture Patterns", "Object Parts"]
                    
                    # Create title labels
                    for i, name in enumerate(layer_names):
                        title = tk.Label(
                            self.feature_frame,
                            text=name,
                            bg=TERMINAL_THEME["bg"],
                            fg=TERMINAL_THEME["fg"],
                            font=TERMINAL_THEME["font"]
                        )
                        title.grid(row=0, column=i, padx=5)
                        self.layer_title_labels.append(title)
                    
                    # Create feature map images
                    for i, fm in enumerate(feature_maps):
                        img = Image.fromarray(fm).convert('L')
                        img = ImageTk.PhotoImage(img)
                        
                        label = tk.Label(
                            self.feature_frame,
                            image=img,
                            bg=TERMINAL_THEME["bg"]
                        )
                        label.image = img
                        label.grid(row=1, column=i, padx=5)
                        self.feature_labels.append(label)

            except Exception as e:
                self.result_text.set("SYSTEM ERROR")
                self.confidence_text.set(f"Failed to process: {str(e)}")
                print(f"[ERROR] {str(e)}")

# Initialize the cyberpunk interface
root = tk.Tk()
root.title("CyberPup 9000")
app = CyberPupApp(root)

# ASCII art splash screen
splash_text = """                
 ____          _   _ 
|    \ ___ ___| |_| |
|  |  |   | . | | . |
|____/|_|_|___|_|___|                   
"""
print(splash_text)

root.mainloop()