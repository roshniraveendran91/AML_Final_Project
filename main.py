from flask import Flask, render_template, request
import numpy as np
import os

import pickle
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.sequence import pad_sequences


app = Flask(__name__)

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for captioning the uploaded image
@app.route('/caption', methods=['POST'])
def caption():
    img_size = 224
    features = {}

    # Get the uploaded image file from the request
    uploaded_image = request.files['imageUpload']

    def read_image(path, img_size=224):
        img = load_img(path, color_mode='rgb', target_size=(img_size, img_size))
        img = img_to_array(img)
        img = img / 255.

        return img

    # Define the path for saving the uploaded image
    image_path = 'uploads/uploaded_image.jpg'
    uploaded_image.save(image_path)
    image = read_image(image_path)
    plt.imshow(image)

    import os
    import pickle

    # Get the directory path of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the paths to the pickle files in the same folder as main.py
    tokenizer_path = os.path.join(script_dir, 'tokenizer.pkl')
    DenseNet201_path = os.path.join(script_dir, 'DenseNet201.pkl')

    # Load the tokenizer from the pickle file
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    print("Model loaded successfully.")

    # Load the DenseNet201 from the pickle file
    with open(DenseNet201_path, 'rb') as f:
        DenseNet201 = pickle.load(f)

    def feature_extraction(image_path, model1, img_size):
        img = load_img(os.path.join(image_path), target_size=(img_size, img_size))
        img = img_to_array(img)
        img = img / 255.
        img = np.expand_dims(img, axis=0)
        feature = model1.predict(img, verbose=0)

        return feature

    # Extract features from the uploaded image
    features = feature_extraction(image_path, DenseNet201, img_size)
    model_path = os.path.join(script_dir, 'model.h5')
    end_model = tf.keras.models.load_model(model_path)

    def idx_to_word(integer, tokenizer):

        for word, index in tokenizer.word_index.items():
            if index == integer:
                return word
        return None

    def predict_caption(model, tokenizer, max_length, feature):

        in_text = "startseq"
        for i in range(max_length):
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], max_length)

            y_pred = model.predict([feature, sequence])
            y_pred = np.argmax(y_pred)

            word = idx_to_word(y_pred, tokenizer)

            if word is None:
                break

            in_text += " " + word

            if word == 'endseq':
                break

        return in_text

    # Perform captioning on the uploaded image
    max_length = 34
    caption_text = predict_caption(end_model, tokenizer, max_length, features)
    return render_template('index.html', caption=caption_text)

if __name__ == '__main__':
    app.run(debug=True)