import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2

# ===== CONFIGURATION =====
TERMINAL_THEME = {
    "bg": "#121212",
    "fg": "#00FF00",
    "accent": "#1F1F1F",
    "font": ("Consolas", 12),
    "title_font": ("Consolas", 16, "bold"),
}

IMG_SIZE = 250
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("[BOOTING UP] Initializing visual cortex...")

# ===== MODEL ARCHITECTURE (MUST MATCH TRAINING SCRIPT) =====
class CNNClassifier(nn.Module):
    def __init__(self, img_size=250):  # Added default img_size to match your training
        super(CNNClassifier, self).__init__()
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            # Fourth convolutional block
            nn.Conv2d(128, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            nn.Dropout(0.5)
        )

        # Calculate the flattened size after the feature extractor
        flattened_size = 256 * (img_size // 16) * (img_size // 16)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 512),
            nn.SiLU(),  # Swish activation
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.SiLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, 1)  # Binary classification
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Initialize model with the correct architecture
model = CNNClassifier(IMG_SIZE)
checkPoint = torch.load("E:/DogAI/models/chihuahua_muffin_modelV4.pth", map_location=DEVICE)
model.load_state_dict(checkPoint)
model.to(DEVICE)
model.eval()
print("[OK] Snack Discriminator v2 Ready.")

# ===== IMAGE TRANSFORM =====
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Added normalization to match training
    std=[0.229, 0.224, 0.225])
])

def load_image(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(DEVICE)
        return tensor, img
    except Exception as e:
        print(f"[ERROR] Image loading failed: {e}")
        return None, None

# ===== FEATURE MAPS =====
def get_feature_maps(model, img_tensor):
    fmap_list = []
    layer_info = []
    hooks = []

    def hook_fn(module, input, output):
        fmap = output.detach().cpu().squeeze(0)
        fmap_list.append(fmap)

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(hook_fn))
    
    with torch.no_grad():
        _ = model(img_tensor)
    
    for hook in hooks:
        hook.remove()

    display_maps = []
    for fmap in fmap_list:
        n_filters = min(64, fmap.shape[0])
        grid = np.zeros((8 * fmap.shape[1], 8 * fmap.shape[2]))
        for i in range(n_filters):
            row = i // 8
            col = i % 8
            filter_img = fmap[i].numpy()
            filter_img = cv2.normalize(filter_img, None, 0, 255, cv2.NORM_MINMAX)
            grid[row*fmap.shape[1]:(row+1)*fmap.shape[1],
                 col*fmap.shape[2]:(col+1)*fmap.shape[2]] = filter_img
        resized = cv2.resize(grid.astype("uint8"), (200, 200))
        display_maps.append(resized)
        layer_info.append(f"Conv Layer\n{fmap.shape[0]} filters")

    return display_maps, layer_info

# ===== CLASSIFY IMAGE =====
def classify_image(img_path):
    img_tensor, pil_img = load_image(img_path)
    if img_tensor is not None:
        with torch.no_grad():
            output = model(img_tensor)
            prob = torch.sigmoid(output).item()  # Your model outputs single value for binary classification

        result = "Muffin 🧁 (Safe to munch)" if prob >= 0.5 else "Chihuahua 🐕 (No nibbling!)"
        confidence = f"Certainty: {max(prob, 1-prob)*100:.2f}%"
        feature_maps, layer_info = get_feature_maps(model, img_tensor)

        torch.cuda.empty_cache()
        return result, confidence, feature_maps, layer_info, pil_img

    return "ERROR", "Failed to classify image", None, None, None

# ===== GUI =====
class TerminalClassifierApp:
    def __init__(self, master):
        self.master = master
        master.configure(bg=TERMINAL_THEME["bg"])
        master.geometry("1000x800")

        self.frame = tk.Frame(master, bg=TERMINAL_THEME["bg"])
        self.frame.pack(padx=20, pady=20)

        self.header = tk.Label(self.frame, 
                             text="🐾 Chihuahua or Muffin? 🧁\nAdvanced Snack Classifier v2.0\nNeural Network Visualization",
                             bg=TERMINAL_THEME["bg"], fg=TERMINAL_THEME["fg"], 
                             font=TERMINAL_THEME["title_font"])
        self.header.pack(pady=(0, 20))

        self.img_frame = tk.Frame(self.frame, bg=TERMINAL_THEME["accent"], padx=5, pady=5)
        self.panel = tk.Label(self.img_frame, text="[ Awaiting image input... ]",
                            bg=TERMINAL_THEME["bg"], fg=TERMINAL_THEME["fg"])
        self.panel.pack()
        self.img_frame.pack(pady=10)

        self.btn_frame = tk.Frame(self.frame, bg=TERMINAL_THEME["bg"])
        self.btn_frame.pack(pady=10)

        self.btn = ttk.Button(self.btn_frame, text="> Analyze Snack", 
                            command=self.open_file, style="Terminal.TButton")
        self.btn.pack(side=tk.LEFT, padx=5)

        self.clear_btn = ttk.Button(self.btn_frame, text="> Clear", 
                                  command=self.clear_display, style="Terminal.TButton")
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        self.result_var = tk.StringVar()
        self.result_label = tk.Label(self.frame, textvariable=self.result_var,
                                   bg=TERMINAL_THEME["bg"], fg="#FF5555", 
                                   font=TERMINAL_THEME["title_font"])
        self.result_label.pack()

        self.confidence_var = tk.StringVar()
        self.confidence_label = tk.Label(self.frame, textvariable=self.confidence_var,
                                       bg=TERMINAL_THEME["bg"], fg=TERMINAL_THEME["fg"], 
                                       font=TERMINAL_THEME["font"])
        self.confidence_label.pack()

        self.canvas = tk.Canvas(self.frame, bg=TERMINAL_THEME["bg"], highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.frame, orient="vertical", command=self.canvas.yview)
        self.feature_frame = tk.Frame(self.canvas, bg=TERMINAL_THEME["bg"])

        self.feature_frame.bind("<Configure>", 
                              lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.feature_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.feature_labels = []
        self.layer_info_labels = []

        self.style = ttk.Style()
        self.style.configure("Terminal.TButton",
                            font=TERMINAL_THEME["font"],
                            background="#00FF00",
                            foreground="black",
                            padding=10)

    def clear_display(self):
        self.result_var.set("")
        self.confidence_var.set("")
        
        # Clear the image properly
        if hasattr(self, 'panel'):
            self.panel.image = None  # Clear the reference
            # Create a new label instead of configuring the existing one
            self.panel.destroy()
            self.panel = tk.Label(self.img_frame, text="[ Awaiting image input... ]",
                                bg=TERMINAL_THEME["bg"], fg=TERMINAL_THEME["fg"])
            self.panel.pack()

        # Clear feature maps and layer info
        for label in self.feature_labels + self.layer_info_labels:
            label.destroy()

        self.feature_labels = []
        self.layer_info_labels = []
        
        # Clear the canvas scroll region
        self.canvas.configure(scrollregion=(0, 0, 0, 0))

    def open_file(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if path:
            try:
                # Clear everything first
                self.clear_display()
                
                # Update status
                self.result_var.set("[ Processing Image... ]")
                self.confidence_var.set("Running advanced snack cognition...")
                self.master.update()
                
                # Process the image
                result, confidence, feature_maps, layer_info, pil_img = classify_image(path)

                # Display input image
                if pil_img:  # Only proceed if we got a valid image
                    pil_img.thumbnail((300, 300))
                    tk_img = ImageTk.PhotoImage(pil_img)
                    
                    # Get rid of text label and replace with image label
                    self.panel.destroy()
                    self.panel = tk.Label(self.img_frame, image=tk_img,
                                        bg=TERMINAL_THEME["bg"])
                    self.panel.image = tk_img  # Keep reference
                    self.panel.pack()

                    self.result_var.set(result)
                    self.confidence_var.set(confidence)

                    if feature_maps:
                        for i, (fmap, info) in enumerate(zip(feature_maps, layer_info)):
                            info_label = tk.Label(self.feature_frame, text=info,
                                                bg=TERMINAL_THEME["bg"], fg=TERMINAL_THEME["fg"],
                                                font=TERMINAL_THEME["font"])
                            info_label.grid(row=i*2, column=0, padx=5, pady=5, sticky="w")
                            self.layer_info_labels.append(info_label)

                            map_img = Image.fromarray(fmap).convert("L")
                            tk_map_img = ImageTk.PhotoImage(map_img)
                            label = tk.Label(self.feature_frame, image=tk_map_img, bg=TERMINAL_THEME["bg"])
                            label.image = tk_map_img  # Keep reference
                            label.grid(row=i*2+1, column=0, padx=5, pady=(0, 20), sticky="w")
                            self.feature_labels.append(label)
                            
                        # Update scroll region after adding new content
                        self.feature_frame.update_idletasks()
                        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

            except Exception as e:
                self.result_var.set("ERROR")
                self.confidence_var.set(f"Processing failed: {str(e)}")
                print(f"[ERROR] {str(e)}")

# ===== LAUNCH =====
root = tk.Tk()
root.title("Advanced Snack-Terminal v2")
app = TerminalClassifierApp(root)

ascii_banner = """                   
 ____          _   _ 
|    \\ ___ ___| |_| |
|  |  |   | . | | . |
|____/|_|_|___|_|___|  
"""
print(ascii_banner)
print("[SYSTEM] Welcome to Advanced Snack-Terminal v2 — Now with neural visualization.")

root.mainloop()