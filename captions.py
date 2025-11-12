import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model

# -------------------------------
# Step 1: Feature Extractor (InceptionV3)
# -------------------------------
def build_feature_extractor():
    model_incep = InceptionV3(weights=None, include_top=False, pooling='avg')
    model_incep.load_weights(r'D:\champa\projects\Architectural image\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
    model = Model(model_incep.input, model_incep.output)
    return model

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feature = model.predict(x, verbose=0)
    return feature[0]

# -------------------------------
# Step 2: Load Tokenizer and Trained Model
# -------------------------------
tokenizer_file = r'D:\champa\projects\Architectural image\combined_dataset\tokenizer.pkl'
model_file = r'D:\champa\projects\Architectural image\combined_dataset\caption_model_best.h5'

with open(tokenizer_file, 'rb') as f:
    tokenizer = pickle.load(f)

caption_model = load_model(model_file)

max_length = 164  # same as training

# -------------------------------
# Step 3: Generate Clean Caption
# -------------------------------
def generate_caption_clean(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    generated_words = []

    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')

        # Fix: expand dims for single image
        yhat = model.predict([np.expand_dims(photo, axis=0), sequence], verbose=0)
        yhat = np.argmax(yhat)

        word = tokenizer.index_word.get(yhat, None)

        if word is None or word == "<unk>" or word in generated_words:
            continue

        in_text += ' ' + word
        generated_words.append(word)

        if word == 'endseq':
            break

    final_caption = ' '.join(generated_words)
    return final_caption

# -------------------------------
# Step 4: Test on New Images
# -------------------------------
if __name__ == "__main__":
    feature_model = build_feature_extractor()

    test_images = [
        r'D:\champa\projects\Architectural image\combined_dataset\resized\0_20.jpg',
        r'D:\champa\projects\Architectural image\combined_dataset\resized\0_21.jpg',
        r'D:\champa\projects\Architectural image\combined_dataset\resized\0_22.jpg'
    ]

    for img_path in test_images:
        photo_feature = extract_features(img_path, feature_model)
        caption = generate_caption_clean(caption_model, tokenizer, photo_feature, max_length)
        print(f"Image: {img_path}")
        print("Caption:", caption)
        print("-" * 80)
