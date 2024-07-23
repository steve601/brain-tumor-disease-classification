from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)
model = load_model(r'artifacts\braintm_model.keras')

UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

disease = {'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route("/detect", methods=['POST'])
def recognize():
    imgfile = request.files['imag']
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], imgfile.filename)
    imgfile.save(image_path)
    
    img = load_img(image_path, target_size=(224, 224))
    img_arr = img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    
    pred = model.predict(img_arr)
    score = tf.nn.softmax(pred)
    
    for key, val in disease.items():
        if val == np.argmax(score):
            msg = f"This is a {key} condition"
    
    return render_template('index.html', text=msg, img_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
