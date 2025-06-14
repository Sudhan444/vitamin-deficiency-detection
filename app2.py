from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import uuid

app = Flask(__name__)
model = load_model('hypervision_vitamin_model.h5')

# Update image size here
img_size = (299, 299)  # Change this to your desired size
class_dict = {0: 'Normal Nail', 1: 'Normal Skin', 2: 'Vitamin B12 Deficiency Skin', 3: 'Vitamin C Deficiency Nail'}
classes = list(class_dict.values())
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure upload directory exists


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    # Save image with a unique filename to avoid conflicts
    filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[-1]
    img_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(img_path)

    try:
        img = image.load_img(img_path, target_size=img_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array)[0]  # Get probabilities for all classes
        predicted_class_index = np.argmax(prediction)  # Get highest probability index
        predicted_label = classes[predicted_class_index]  # Get class name
        confidence_score = round(float(prediction[predicted_class_index]) * 100, 2)  # Convert to percentage

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'})
    finally:
        os.remove(img_path)  # Remove the temporary file after processing

    return jsonify({'predicted_class': predicted_label, 'confidence_score': confidence_score})


if __name__ == '__main__':
    app.run(debug=True)
