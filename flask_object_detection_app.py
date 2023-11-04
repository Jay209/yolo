from flask import Flask, render_template, request, session, Response

from werkzeug.utils import secure_filename
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from yolov3.utils import detect_image, detect_realtime, detect_video, Load_Yolo_model, detect_video_realtime_mp
from yolov3.configs import *


# *** Backend operation

# WSGI Application
# Provide template folder name
# The default folder name should be "templateFiles" else need to mention custom folder name

# Accepted image for to upload for object detection model
UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'PNG', 'jpg', 'jpeg', 'gif', 'bmp'}

app = Flask(__name__, template_folder='templateFiles', static_folder='staticFiles')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.secret_key = 'You Will Never Guess'


# YOLO object detection function
def detect_object(uploaded_image_path):

    yolo = Load_Yolo_model()

    #classification_model = load_model(CLASSIFICATION_MODEL)

    # Write output image (object detection output)
    output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output_image.jpg')
    detect_image(yolo, None, uploaded_image_path, output_image_path, input_size=YOLO_INPUT_SIZE, show=False,
                     CLASSES=TRAIN_CLASSES, rectangle_colors=(255, 0, 0))
    return (output_image_path)


@app.route('/')
def index():
    return render_template('index_upload_and_display_image.html')


@app.route('/', methods=("POST", "GET"))
def uploadFile():
    if request.method == 'POST':
        uploaded_img = request.files['uploaded-file']
        img_filename = secure_filename(uploaded_img.filename)
        uploaded_img.save(os.path.join(app.config['UPLOAD_FOLDER'], img_filename))

        session['uploaded_img_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)

        return render_template('index_upload_and_display_image_page2.html')


@app.route('/show_image')
def displayImage():
    img_file_path = session.get('uploaded_img_file_path', None)
    return render_template('show_image.html', user_image=img_file_path)


@app.route('/detect_object')
def detectObject():
    uploaded_image_path = session.get('uploaded_img_file_path', None)
    output_image_path = detect_object(uploaded_image_path)
    print(output_image_path)
    return render_template('show_image.html', user_image=output_image_path)


# flask clear browser cache (disable cache)
# Solve flask cache images issue
@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0")