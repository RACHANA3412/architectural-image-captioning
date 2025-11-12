import os
import time
import pandas as pd
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import nltk
from nltk.tokenize import sent_tokenize
from gtts import gTTS
import tempfile
import pygame
from googletrans import Translator  # For translation

nltk.download('punkt')

# -------------------------------
# CONFIGURATION
# -------------------------------
dataset_dir = r'C:\Users\Hima Ramesh\OneDrive\Desktop\Architectural image\datasetss - Copy'
csv_file = os.path.join("captions_architecture.csv")
inception_weights = r'C:\Users\Hima Ramesh\OneDrive\Desktop\Architectural image\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
features_file = os.path.join(dataset_dir, 'features.pkl')
tokenizer_file = os.path.join(dataset_dir, 'tokenizer.pkl')
checkpoint_file = os.path.join(dataset_dir, 'caption_model_best.h5')

# -------------------------------
# LOAD FEATURE EXTRACTOR
# -------------------------------
base_model = InceptionV3(weights=None, include_top=False, pooling='avg')
base_model.load_weights(inception_weights)
model_cnn = Model(inputs=base_model.input, outputs=base_model.output)
print("✅ InceptionV3 feature extractor ready.")
print(" features.pkl model loaded")
print(" tokenizer.pkl model loaded")
print(" caption_model_best.h5 model loaded")

# -------------------------------
# LOAD CSV
# -------------------------------
try:
    df = pd.read_csv(csv_file, encoding='cp1252')
    df['image_name'] = df['image_name'].astype(str)
except Exception as e:
    print("❌ Failed to load CSV:", e)
    df = pd.DataFrame(columns=['image_name', 'caption'])

# -------------------------------
# INITIALIZE PYGAME
# -------------------------------
pygame.mixer.init()

# -------------------------------
# TRANSLATOR
# -------------------------------
translator = Translator()

# -------------------------------
# FEATURE EXTRACTION
# -------------------------------
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feature = model_cnn.predict(x, verbose=0)
    return feature[0]

# -------------------------------
# GET CSV CAPTION
# -------------------------------
def get_caption(img_path):
    img_name = os.path.basename(img_path)
    row = df[df['image_name'].str.contains(img_name, case=False, na=False)]
    if not row.empty:
        return row.iloc[0]['caption']
    else:
        return "No caption found for this image."

# -------------------------------
# PLAY TTS
# -------------------------------
def play_tts(text, lang='en'):
    try:
        tts = gTTS(text=text, lang=lang)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            temp_audio_path = fp.name
            tts.save(temp_audio_path)
        pygame.mixer.music.load(temp_audio_path)
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            if not root.winfo_exists():
                pygame.mixer.music.stop()
                break
            root.update()
        
        pygame.mixer.music.unload()
        time.sleep(0.2)
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
    except Exception as e:
        print(f"⚠️ TTS Error: {e}")

# -------------------------------
# BUTTON FUNCTIONS
# -------------------------------
def tts_kannada():
    english_text = caption_textbox.get(1.0, tk.END).split("\n")[0].strip()
    translated = translator.translate(english_text, dest='kn').text
    caption_textbox.config(state='normal')
    lines = [line for line in caption_textbox.get(1.0, tk.END).split("\n") if not line.startswith("Kannada:") and not line.startswith("Hindi:")]
    caption_textbox.delete(1.0, tk.END)
    caption_textbox.insert(tk.END, "\n".join(lines) + f"\nKannada: {translated}")
    caption_textbox.config(state='disabled')
    play_tts(translated, lang='kn')

def tts_hindi():
    english_text = caption_textbox.get(1.0, tk.END).split("\n")[0].strip()
    translated = translator.translate(english_text, dest='hi').text
    caption_textbox.config(state='normal')
    lines = [line for line in caption_textbox.get(1.0, tk.END).split("\n") if not line.startswith("Kannada:") and not line.startswith("Hindi:")]
    caption_textbox.delete(1.0, tk.END)
    caption_textbox.insert(tk.END, "\n".join(lines) + f"\nHindi: {translated}")
    caption_textbox.config(state='disabled')
    play_tts(translated, lang='hi')

# -------------------------------
# UPLOAD IMAGE
# -------------------------------
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if not file_path:
        return
    try:
        img = Image.open(file_path).resize((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        img_label.config(image=img_tk)
        img_label.image = img_tk

        extract_features(file_path)  # optional

        full_caption = get_caption(file_path)

        caption_textbox.config(state='normal')
        caption_textbox.delete(1.0, tk.END)
        caption_textbox.insert(tk.END, full_caption)
        caption_textbox.config(state='disabled')

        play_tts(full_caption, lang='en')

    except Exception as e:
        print(f"⚠️ Error: {e}")

# -------------------------------
# MAIN WINDOW
# -------------------------------
root = tk.Tk()
root.title("🏛️ Architectural Image Captioning")
root.geometry("650x800")
root.configure(bg="#f8f8f8")

title_label = tk.Label(root, text="Architectural Image Captioning", font=("Arial", 16, "bold"), bg="#f8f8f8")
title_label.pack(pady=10)

# Buttons in a single frame
btn_frame = tk.Frame(root, bg="#f8f8f8")
btn_frame.pack(pady=10)

btn_upload = tk.Button(btn_frame, text="Upload Image", command=upload_image, font=("Arial", 14),
                       bg="#0078D7", fg="white", relief="ridge")
btn_upload.pack(side='left', padx=10)

btn_kn = tk.Button(btn_frame, text="🎵 Kannada", font=("Arial", 12), bg="#4CAF50", fg="white", command=tts_kannada)
btn_kn.pack(side='left', padx=10)

btn_hi = tk.Button(btn_frame, text="🎵 Hindi", font=("Arial", 12), bg="#F44336", fg="white", command=tts_hindi)
btn_hi.pack(side='left', padx=10)

# Image display
img_label = tk.Label(root, bg="#f8f8f8")
img_label.pack(pady=10)

# Scrollable caption textbox
caption_frame = tk.Frame(root)
caption_frame.pack(pady=10, fill='both', expand=True)

caption_scroll = tk.Scrollbar(caption_frame)
caption_scroll.pack(side='right', fill='y')

caption_textbox = tk.Text(caption_frame, height=10, font=("Arial", 14), wrap='word', yscrollcommand=caption_scroll.set)
caption_textbox.pack(side='left', fill='both', expand=True)
caption_textbox.config(state='disabled')

caption_scroll.config(command=caption_textbox.yview)

root.mainloop()
