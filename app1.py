from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import cv2
import pywt
import json

app = Flask(__name__)

# Load the model and the class dictionary
model = joblib.load('Model/saved_model (1).pkl')

with open("Model/class_dictionary (1).json", "r") as f:
    class_dict = json.load(f)

# Reverse the class dictionary
class_names = {v: k for k, v in class_dict.items()}


# Define wavelet transformation function
def w2d(img, mode='haar', level=1):
    imArray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    imArray = np.float32(imArray)
    imArray /= 255
    coeffs = pywt.wavedec2(imArray, mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)
    return imArray_H


# Preprocessing function
def preprocess_image(image):
    img = cv2.imdecode(np.fromstring(image.read(), np.uint8), 1)
    scalled_raw_img = cv2.resize(img, (32, 32))
    img_har = w2d(img, 'db1', 5)
    scalled_img_har = cv2.resize(img_har, (32, 32))
    combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))
    return combined_img.flatten().reshape(1, -1)


# Classification function
def classify_image(image):
    preprocessed_image = preprocess_image(image)
    probabilities = model.predict_proba(preprocessed_image)[0]
    return probabilities


# Route for home page
@app.route('/')
def index():
    return render_template('index.html')


# Route for image upload and classification
@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400


    probabilities = classify_image(file)

    # Prepare response data
    results = {}
    d2 = {
        'C:\\chat-data\\py\\DataScience\\CelebrityFaceRecognition\\model\\dataset\\cropped\\lionel_messi': 'Lionel Messi',
        'C:\\chat-data\\py\\DataScience\\CelebrityFaceRecognition\\model\\dataset\\cropped\\maria_sharapova': 'Maria Sharapova',
        'C:\\chat-data\\py\\DataScience\\CelebrityFaceRecognition\\model\\dataset\\cropped\\roger_federer': 'Roger Federer',
        'C:\\chat-data\\py\\DataScience\\CelebrityFaceRecognition\\model\\dataset\\cropped\\serena_williams': 'Serena Williams',
        'C:\\chat-data\\py\\DataScience\\CelebrityFaceRecognition\\model\\dataset\\cropped\\virat_kohli': 'Virat Kohli'}

    for idx, (image_path_key, celebrity_name) in enumerate(d2.items()):
        prob = probabilities[idx]
        results[celebrity_name] = f"{prob * 100:.2f}%"

    # Find the predicted class using the index of the maximum probability
    predicted_class_idx = np.argmax(probabilities)
    predicted_class = list(d2.values())[predicted_class_idx]

    # Return results as JSON
    return jsonify({"probabilities": results, "predicted": predicted_class})


if __name__ == "__main__":
    app.run(debug=True)