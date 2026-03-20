from flask import Flask, render_template, request, send_file
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

encoder = load_model("encoder.h5")
decoder = load_model("decoder.h5")

UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def text_to_binary(text):
    return ''.join(format(ord(c), '08b') for c in text)

def binary_to_text(binary):
    chars = [binary[i:i+8] for i in range(0, len(binary), 8)]
    return ''.join(chr(int(b, 2)) for b in chars if int(b, 2) < 128)

def binary_to_matrix(binary):
    matrix = np.zeros((64, 64, 1), dtype=np.float32)
    for i in range(min(len(binary), 64*64)):
        r, c = divmod(i, 64)
        matrix[r, c, 0] = int(binary[i]) * 2 - 1  # Scale to [-1, 1]
    return matrix

def matrix_to_binary(matrix):
    matrix = np.squeeze(matrix)
    return ''.join(['1' if x > 0 else '0' for x in matrix.flatten()])

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/encode', methods=['POST'])
def encode():
    file = request.files['image']
    message = request.form['message']

    if file and message:
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        img = cv2.imread(path)
        img = cv2.resize(img, (64, 64)) / 255.0
        img = np.expand_dims(img, axis=0)

        binary_msg = text_to_binary(message)
        msg_matrix = binary_to_matrix(binary_msg)
        msg_matrix = np.expand_dims(msg_matrix, axis=0)

        stego_features = encoder.predict([img, msg_matrix])
        np.save(os.path.join(RESULTS_FOLDER, "stego_features.npy"), stego_features)
        np.save(os.path.join(RESULTS_FOLDER, "original_message.npy"), msg_matrix)

        return "Message encoded successfully!"

@app.route('/decode')
def decode():
    stego_features = np.load(os.path.join(RESULTS_FOLDER, "stego_features.npy"))
    reconstructed_img, extracted_msg = decoder.predict(stego_features)

    binary = matrix_to_binary(extracted_msg)
    text = binary_to_text(binary)

    img = (reconstructed_img[0] * 255).astype(np.uint8)
    path = os.path.join(RESULTS_FOLDER, "reconstructed.png")
    cv2.imwrite(path, img)

    return f"Decoding complete! Extracted message: {text}"

@app.route('/download')
def download():
    return send_file(os.path.join(RESULTS_FOLDER, "reconstructed.png"), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
