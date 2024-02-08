from flask import Flask, render_template, request, redirect
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__, template_folder='template')

model = load_model('/Users/shitleshbakshi/Library/CloudStorage/OneDrive-UniversityofSouthWales/Deep Learning Assignment Code/age_model.h5')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']

    # Read and preprocess the image
    image = Image.open(file.stream).convert('L')  # Convert to grayscale
    image = image.resize((224, 224))
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Make predictions
    predictions = model.predict(img_array)

    # Extract age and gender predictions
    age_prediction = np.round(predictions[0][0],2)
    gender_prediction = "Male" if predictions[1][0] < 0.8 else "Female"

    return render_template('index.html', age=age_prediction, gender=gender_prediction)


if __name__ == '__main__':
    app.run(debug=False)
