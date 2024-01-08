# # from flask import Flask, request, jsonify, render_template
# # from keras.models import load_model
# # from PIL import Image
# # from tensorflow.keras.preprocessing.image import img_to_array, load_img
# # import numpy as np
# # # from keras.preprocessing import image
# # from keras.applications.vgg19 import preprocess_input
# # import os
# # from flask_cors import CORS
# # app = Flask(__name__)
# # CORS(app)

# # # Load your trained model
# # # model = load_model('model_vgg19 (1)-Malaria_version2.h5')
# # # model = load_model('Leukemia_prediction.keras')
# # # model = load_model('model_vgg19_Leukemia.h5')
# # # E:\FYP\Implementation\concluded\version 2\Predictions\efficientnetb3_notop.h5
# # model = load_model('efficientnetb3_notop.h5')

# # # Function to predict the class of an image
# # def model_predict(image_path):
# #     img = load_img(image_path, target_size=(224, 224))
# #     img_array = img_to_array(img)
# #     img_array = np.expand_dims(img_array, axis=0)
# #     img_array /= 255.0
# #     result = model.predict(img_array)
# #     print(result)
# #     predicted_class = np.argmax(result, axis=1)
# #     # print(predicted_class)
# #     return predicted_class[0]

# # # Function to predict the class of an image
# # # def model_predict(image_path):
# # #     img = load_img(image_path, target_size=(224, 224))
# # #     img_array = img_to_array(img)
# # #     img_array = np.expand_dims(img_array, axis=0)
# # #     img_array /= 255.0
# # #     result = model.predict(img_array)
# # #     predicted_class = np.argmax(result, axis=1)
# # #     return predicted_class[0]


# # @app.route('/api/classifyLeukemia', methods=['POST'])
# # def classify_image():
# #     if request.method == 'POST':
# #         # Get the uploaded image file
# #         f = request.files['file']
        
# #         # Save the file to a temporary location
# #         file_path = 'temp.png'
# #         f.save(file_path)
        
# #         # Make a prediction
# #         predicted_class = model_predict(file_path)
# #         print(predicted_class)
        
# #         # Determine the class based on the predicted class index
# #         prediction = 'Uninfected' if predicted_class == 1 else 'Infected with Leukemia'
        
# #         # Delete the temporary file
# #         os.remove(file_path)
        
# #         # Return the prediction class as JSON
# #         result = {'prediction': prediction}
# #         return jsonify(result)
# #     return 'Invalid Request'

# # if __name__ == '__main__':
# #     app.run(debug=True, port=5001)



# from flask import Flask, request, jsonify
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array
# import tensorflow as tf
# from flask_cors import CORS
# from PIL import Image
# import numpy as np

# import io

# app = Flask(__name__)
# CORS(app)
# # Load the model
# model = load_model('./model.h5')
# @app.route("/")
# def home():
#     return "<h1>Server is running</h1>"
 
# @app.route('/api/classifyLeukemia', methods=['POST'])
# def predict():
#     # Load the image from the request
#     image = Image.open(io.BytesIO(request.files['file'].read()))

#     # Preprocess the image so it matches the input format of your model
#     image = preprocess_image(image)

#     # Make a prediction
#     prediction = model.predict(image)


#     # Convert the prediction to a JSON-compatible format
#     prediction = prediction.tolist()
#     if(prediction[0][0] < 0.5):
#         return "Cancer"
#     elif(prediction[0][0] > 0.5):
#         return "Normal"
#     else:
#         return "Unknown"


 

   
# def preprocess_image(image):
#     # Convert the PIL Image to a numpy array
#     image = img_to_array(image)

#     # Resize the image
#     image = tf.image.resize(image, (224, 224))

#     # Expand the dimensions to match the input shape of your model
#     image = np.expand_dims(image, axis=0)

#     return image

# if __name__ == '__main__':
#     # app.run(debug=True, port=os.getenv("PORT", default=5001))
#     app.run(debug=True, port=5001)




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

model = load_model('Leukemia_effecientnetb3_NEW version.h5')

# Function to predict the class of an image
def model_predict(image_path):
    img = load_img(image_path, target_size=(224, 224,3))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # img_array /= 255.0
    result = model.predict(img_array)
    print(result)
    predicted_class = np.argmax(result, axis=1)
    print("Hello world",predicted_class)
    return predicted_class[0]


@app.route('/api/classifyLeukemia', methods=['POST'])
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
        prediction = 'Uninfected' if predicted_class == 1 else 'Infected with Leukemia'
        
        # Delete the temporary file
        os.remove(file_path)
        
        # Return the prediction class as JSON
        result = {'prediction': prediction}
        return jsonify(result)
    return 'Invalid Request'

if __name__ == '__main__':
    app.run(debug=True, port=5001)
