from flask import Flask, request, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename
import numpy as np
import cv2
from tensorflow.keras.models import load_model
app = Flask(__name__)
# Set upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads/'
STATIC_FOLDER = 'static/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Load the model
model = load_model('my_model.h5')
# Ensure the upload and static folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def prep_image(image, crop_size, size_y, size_x):
    # Resize the image
    prepd_image = cv2.resize(image, (size_y, size_x))
    # Crop the image to remove the border black pixels
    prepd_image = prepd_image[crop_size:-crop_size, crop_size:-crop_size]
    return prepd_image
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # Process the image
            img = cv2.imread(filepath, cv2.IMREAD_COLOR)
            size_x, size_y, crop_size = 148, 148, 10
            img = prep_image(img, crop_size, size_y, size_x)
            img = np.array(img) / 255.0
            img = np.expand_dims(img, axis=0)
            predicted_mask = model.predict(img)
            predicted_mask = predicted_mask.reshape((size_x - 2 * crop_size, size_y - 2 * crop_size))
            # Save the original and predicted mask images
            original_image_path = os.path.join(STATIC_FOLDER, 'original_' + filename)
            predicted_mask_path = os.path.join(STATIC_FOLDER, 'mask_' + filename)
            cv2.imwrite(original_image_path, img[0] * 255)
            cv2.imwrite(predicted_mask_path, predicted_mask * 255)
            return render_template('result.html', original_image='original_' + filename, mask_image='mask_' + filename)
    return render_template('upload.html')
if __name__ == "__main__":
    app.run(debug=True)