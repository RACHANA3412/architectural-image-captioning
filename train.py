# ==========================================
# ARCHITECTURAL IMAGE CAPTIONING - FULL TRAINING
# ==========================================

import os
import string
import numpy as np
import pandas as pd
from PIL import Image
import pickle
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint

# -------------------------------
# CONFIGURATION
# -------------------------------
dataset_dir = r'D:\champa\projects\Architectural image\datasetss - Copy'
resized_dir = os.path.join(dataset_dir, 'resized')
captions_file = os.path.join( 'captions_architecture.csv')
inception_weights = r'D:\champa\projects\Architectural image\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

os.makedirs(resized_dir, exist_ok=True)

# -------------------------------
# STEP 1: Load Captions
# -------------------------------
try:
    df = pd.read_csv(captions_file, encoding='ISO-8859-1')  
except:
    df = pd.read_csv(captions_file, encoding='cp1252')

captions_dict = {}
all_captions = []

for _, row in df.iterrows():
    img_name = row['image_name']
    caption = row['caption']
    if pd.isna(caption):
        continue
    caption = str(caption).lower()
    caption = caption.translate(str.maketrans('', '', string.punctuation))
    caption = '<start> ' + caption + ' <end>'
    captions_dict[img_name] = caption
    all_captions.append(caption)

print(f"✅ Loaded {len(captions_dict)} captions.")

# -------------------------------
# STEP 2: Resize Images
# -------------------------------
for img_name in os.listdir(dataset_dir):
    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(dataset_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        img = img.resize((299, 299))
        img.save(os.path.join(resized_dir, img_name))
print("✅ Images resized.")

# -------------------------------
# STEP 3: Extract CNN Features
# -------------------------------
base_model = InceptionV3(weights=None, include_top=False, pooling='avg')
base_model.load_weights(inception_weights)
model_cnn = Model(inputs=base_model.input, outputs=base_model.output)

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model_cnn.predict(x, verbose=0)
    return features[0]

features_file = os.path.join(dataset_dir,'features.pkl')
if os.path.exists(features_file):
    features = pickle.load(open(features_file,'rb'))
    print("✅ Features loaded from file.")
else:
    features = {}
    for img_name in os.listdir(resized_dir):
        if img_name in captions_dict:
            path = os.path.join(resized_dir, img_name)
            features[img_name] = extract_features(path)
    pickle.dump(features, open(features_file,'wb'))
    print("✅ Features extracted and saved.")

# -------------------------------
# STEP 4: Tokenize Captions
# -------------------------------
vocab_size = 5000
tokenizer_file = os.path.join(dataset_dir,'tokenizer.pkl')
if os.path.exists(tokenizer_file):
    tokenizer = pickle.load(open(tokenizer_file,'rb'))
    print("✅ Tokenizer loaded.")
else:
    tokenizer = Tokenizer(num_words=vocab_size, oov_token='<unk>')
    tokenizer.fit_on_texts(all_captions)
    pickle.dump(tokenizer, open(tokenizer_file,'wb'))
    print("✅ Tokenizer created and saved.")

max_length = max(len(c.split()) for c in all_captions)
print(f"Vocabulary size: {vocab_size}, Max caption length: {max_length}")

# -------------------------------
# STEP 5: Create Sequences
# -------------------------------
def create_sequences(tokenizer, max_length, desc, photo, vocab_size):
    X1, X2, y = [], [], []
    seq = tokenizer.texts_to_sequences([desc])[0]
    for i in range(1, len(seq)):
        in_seq, out_seq = seq[:i], seq[i]
        in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
        out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
        X1.append(photo)
        X2.append(in_seq)
        y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

X1_list, X2_list, y_list = [], [], []
for img_name, caption in captions_dict.items():
    if img_name in features:
        photo = features[img_name]
        X1, X2, y = create_sequences(tokenizer, max_length, caption, photo, vocab_size)
        X1_list.append(X1)
        X2_list.append(X2)
        y_list.append(y)

X1 = np.vstack(X1_list)
X2 = np.vstack(X2_list)
y = np.vstack(y_list)
print(f"✅ Training sequences prepared: {X1.shape}, {X2.shape}, {y.shape}")

# -------------------------------
# STEP 6: Build CNN + LSTM Model
# -------------------------------
# Image branch
inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

# Text branch
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 512, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)

# Merge
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# -------------------------------
# STEP 7: Train Model with Checkpoints
# -------------------------------
checkpoint = ModelCheckpoint(os.path.join(dataset_dir,'caption_model_best.h5'),
                             monitor='val_loss', verbose=1, save_best_only=True, mode='min')

history = model.fit([X1, X2], y,
                    epochs=10,  
                    batch_size=64,
                    validation_split=0.1,
                    callbacks=[checkpoint])

print("✅ Model training completed and best model saved.")
