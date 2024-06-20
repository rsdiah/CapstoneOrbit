from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__, static_url_path='/static/assets')
app.static_folder = 'static'
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load model
model = load_model('model.h5')

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    preds = model.predict(img_array)
    return preds

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            preds = model_predict(filepath, model)
            result = np.argmax(preds, axis=1)

            return render_template('index.html', result=result, img_path=filepath)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
