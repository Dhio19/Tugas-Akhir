from flask import Flask, request, render_template, redirect, url_for, Response
import os
from werkzeug.utils import secure_filename
import torch
import cv2
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['RESULT_FOLDER'] = 'static/results/'

# Ensure the upload and result directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path="./models_train/best-32", force_reload=True)
model.conf = 0.7

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return render_template ('Nofile.html')
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Run YOLOv5 inference
            results = model(filepath)
            render_and_save_results(results)

            return redirect(url_for('result'))
    return render_template('upload.html')

@app.route('/result')
def result():
    result_image_url = url_for('static', filename='results/results.jpg')
    return render_template('result.html', result_image_url=result_image_url)

@app.route('/webcam')
def webcam():
    return render_template('livecam.html')

def generate():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv5 inference
        results = model(frame)
        results.render()

        # Convert to JPEG format
        ret, buffer = cv2.imencode('.jpg', results.ims[0])
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def render_and_save_results(results):
    # Render the results on the image
    results.render()

    # Save the rendered image to the result folder
    output_image_path = os.path.join(app.config['RESULT_FOLDER'], 'results.jpg')
    rendered_image = results.ims[0]
    rendered_image = cv2.cvtColor(rendered_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_image_path, rendered_image)

    # Debug: Print result file path
    print(f"Result file saved to {output_image_path}")
    print(f"Files in result folder: {os.listdir(app.config['RESULT_FOLDER'])}")

@app.route('/alfabet')
def alfabet():
    return render_template('alfabet.html')

if __name__ == '__main__':
    app.run(debug=True)
