import joblib
import numpy as np
import cv2
import json
import os
from sklearn.preprocessing import StandardScaler
import numpy as np
import pywt
import cv2
import matplotlib.pyplot as plt

def w2d(img, mode='haar', level=1):
    imArray = img
    #Datatype conversions
    #convert to grayscale
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    #convert to float
    imArray =  np.float32(imArray)
    imArray /= 255;
    # compute coefficients
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    coeffs_H=list(coeffs)
    coeffs_H[0] *= 0;

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255;
    imArray_H =  np.uint8(imArray_H)

    return imArray_H
# Load the trained model
model = joblib.load('Model/saved_model (1).pkl')

# Load the class dictionary
with open("Model/class_dictionary (1).json", "r") as f:
    class_dict = json.load(f)

# Reverse the class dictionary to map class indices to names
class_names = {v: k for k, v in class_dict.items()}


# Preprocessing function for the image
def preprocess_image(image_path):
    """
    Preprocess the image for the SVM classifier: resizing, extracting wavelet features,
    and flattening the image.
    """
    img = cv2.imread(image_path)
    scalled_raw_img = cv2.resize(img, (32, 32))  # Resize image to 32x32
    img_har = w2d(img, 'db1', 5)  # Apply wavelet transform (you already have w2d implemented)
    scalled_img_har = cv2.resize(img_har, (32, 32))  # Resize wavelet-transformed image

    # Combine raw image and wavelet transformed image
    combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))

    # Return as a flattened feature vector
    return combined_img.flatten().reshape(1, -1)


# Function to classify the image
def classify_image(image_path):
    """
    Classify an image and return the probabilities for each celebrity.
    """
    preprocessed_image = preprocess_image(image_path)

    # Predict probabilities using the trained model
    probabilities = model.predict_proba(preprocessed_image)[0]

    return probabilities


# Function to display classification results
def display_classification_results(image_path):
    """
    Display the classification results with probabilities for each celebrity.
    """
    probabilities = classify_image(image_path)

    # Display the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    # Display probabilities for each celebrity
    print("Class probabilities:")
    #   for idx, prob in enumerate(probabilities):
    #         if idx == 0:
    #             print(f"Lionel Messi: {prob * 100:.2f}%")
    #         if idx == 1:
    #             print(f"Maria Sharapova: {prob * 100:.2f}%")
    #         if idx == 2:
    #             print(f"Roger Federer: {prob * 100:.2f}%")
    #         if idx == 3:
    #             print(f"Serena Williams: {prob * 100:.2f}%")
    #         if idx == 4:
    #             print(f"Virat Kohli: {prob * 100:.2f}%")
    for idx, (image_path_key, celebrity_name) in enumerate(d2.items()):
        prob = probabilities[idx]
        print(f"{celebrity_name}: {prob * 100:.2f}%")

    # Get the predicted class
    predicted_class_idx = np.argmax(probabilities)
    print(f"\nPredicted Celebrity: {d2[class_names[predicted_class_idx]]}")


d2 = {'C:\\chat-data\\py\\DataScience\\CelebrityFaceRecognition\\model\\dataset\\cropped\\lionel_messi': 'Lionel Messi',
      'C:\\chat-data\\py\\DataScience\\CelebrityFaceRecognition\\model\\dataset\\cropped\\maria_sharapova': 'Maria Sharapova',
      'C:\\chat-data\\py\\DataScience\\CelebrityFaceRecognition\\model\\dataset\\cropped\\roger_federer': 'Roger Federer',
      'C:\\chat-data\\py\\DataScience\\CelebrityFaceRecognition\\model\\dataset\\cropped\\serena_williams': 'Serena Williams',
      'C:\\chat-data\\py\\DataScience\\CelebrityFaceRecognition\\model\\dataset\\cropped\\virat_kohli': 'Virat Kohli'}

# Provide the path to the image you want to classify
image_path = r'C:\chat-data\py\DataScience\CelebrityFaceRecognition\model\test_images\sharapova2.JPG'

# Display classification results
display_classification_results(image_path)
