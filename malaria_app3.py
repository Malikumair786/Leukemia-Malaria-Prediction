from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
# from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
import os
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

# Load your trained model
# model = load_model('model_vgg19 (1)-Malaria_version2.h5')
model = load_model('malariaModel_vgg19V11.h5')
# E:\FYP\Implementation\concluded\Predictions\malariaModel_vgg19V11.h5

# Function to predict the class of an image
def model_predict(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    result = model.predict(img_array)
    print(result)
    predicted_class = np.argmax(result, axis=1)
    print("Predicted class: ",predicted_class)
    # print(predicted_class)
    return predicted_class[0]

@app.route('/api/classifyMalaria', methods=['POST'])
def classify_image():
    if request.method == 'POST':
        # Get the uploaded image file
        f = request.files['file']
        # Save the file to a temporary location
        file_path = 'temp.png'
        f.save(file_path)
        # Make a prediction
        predicted_class = model_predict(file_path)
        # print(predicted_class)
        # Determine the class based on the predicted class index
        prediction = 'Uninfected' if predicted_class == 1 else 'Infected'
        # Delete the temporary file
        os.remove(file_path)
        
        # Return the prediction class as JSON
        result = {'prediction': prediction}
        return jsonify(result)
    return 'Invalid Request'

if __name__ == '__main__':
    app.run(debug=True)
